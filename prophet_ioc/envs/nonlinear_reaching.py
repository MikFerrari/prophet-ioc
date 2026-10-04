from typing import NamedTuple

import jax.numpy as jnp

from prophet_ioc.envs.base import Env


class NonlinearReachingParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.


class NonlinearReaching(Env):
    def __init__(self, dt=0.01, target=jnp.array([0.05, 0.5]), x0=jnp.array([jnp.pi / 4, jnp.pi / 2, 0., 0.]),
                 indep_noise=0., upper_arm_length=0.3, forearm_length=0.33, I1=0.025, I2=0.045):
        """ Non-linear reaching task from Weiwei Li's PhD thesis

        Args:
            dt (float): time step duration
            target (jnp.array): target position (x, y)
            x0 (jnp.array): initial state (theta1, theta2, theta1_dot, theta2_dot)
            indep_noise (float): state-independent noise on dynamics
            upper_arm_length (float): upper arm length in meters
            forearm_length (float): forearm length in meters
            I1, I2 (float): moments of inertia of the joints (kg / m**2)
        """
        self.dt = dt

        self.target = target
        self.x0 = x0

        self.l1 = upper_arm_length
        self.l2 = forearm_length

        # setting the centers of mass of the arm links
        # based on the original values in Li (2006)
        # s1 = 0.11 / 0.3 * self.l1
        self.s2 = 0.16 / 0.33 * self.l2

        # scale the masses of the arm segments (assumed cylinders)
        # according to their lengths
        # self.m1 = 1.4 / 0.3 * self.l1
        self.m2 = 1.1 / 0.33 * self.l2

        # eqn 3.5
        self.d1 = I1 + I2 + self.m2 * self.l1 ** 2
        self.d2 = self.m2 * self.l1 * self.s2
        self.d3 = I2

        self.bii = 0.05
        self.bij = 0.025

        self.v = indep_noise
        super().__init__(state_shape=(4,), action_shape=(2,), observation_shape=(4,))

    def _dynamics(self, state, action, noise, params):
        """Discrete-time stochastic manipulator dynamics for 2-link planar arm.
        
        Follows Euler-Lagrange manipulator equations:
            M(theta) * ddot_theta + C(theta, dot_theta) * dot_theta + B * dot_theta = u + w_motor
        which transforms into state-space form dot_x = [dot_theta, ddot_theta]^T.
        
        Args:
            state (jnp.ndarray): State x = [theta1, theta2, dot_theta1, dot_theta2] in R^4.
            action (jnp.ndarray): Control torques u = [tau1, tau2] in R^2.
            noise (jnp.ndarray): Standard Gaussian noise vector [xi_indep(2), xi_motor(2)] in R^4.
            params (NonlinearReachingParams): Dynamic parameters including motor_noise (sigma_m).
        Returns:
            jnp.ndarray: Next state x_{t+1} = x_t + dt * (dx + du) + w via Euler-Maruyama discretization.
        """
        # Determinant of the 2x2 symmetric inertia matrix M(theta2)
        det = self.d1 * self.d3 - self.d3 ** 2 - (self.d2 * jnp.cos(state[1])) ** 2
        
        # Passive drift: [dot_theta1, dot_theta2, ddot_theta1_drift, ddot_theta2_drift]^T
        # includes Coriolis, centripetal, and joint cross-damping (-M^{-1} * (C * dot_theta + B * dot_theta))
        dx = jnp.array([state[2],
                        state[3],
                        1 / det * (-self.d2 * self.d3 * (state[2] + state[3]) ** 2 * jnp.sin(state[1]) - self.d2 ** 2 *
                                   state[2] ** 2 * jnp.sin(
                                    state[1]) * jnp.cos(state[1]) - self.d2 * (
                                           self.bij * state[2] + self.bii * state[3]) * jnp.cos(state[1]) + (
                                           self.d3 * self.bii - self.d3 * self.bij) * state[2] + (
                                           self.d3 * self.bij - self.d3 * self.bii) * state[3]),
                        1 / det * (self.d2 * self.d3 * state[3] * (2 * state[2] + state[3]) * jnp.sin(
                            state[1]) + self.d1 * self.d2 * state[
                                       2] ** 2 * jnp.sin(state[1]) + self.d2 ** 2 * (
                                           state[2] + state[3]) ** 2 * jnp.sin(state[1]) * jnp.cos(
                            state[1]) + self.d2 * (
                                           (2 * self.bij - self.bii) * state[2] + (2 * self.bii - self.bij) * state[
                                       3]) * jnp.cos(state[1]) + (
                                           self.d1 * self.bij - self.d3 * self.bii) * state[2] + (
                                           self.d1 * self.bii - self.d3 * self.bij) * state[3])])
        
        # Control input mapping G = [0; M^{-1}], mapping joint torques u to angular accelerations
        G = 1 / det * jnp.array([[0., 0.],
                                 [0., 0.],
                                 [self.d3, -(self.d3 + self.d2 * jnp.cos(state[1]))],
                                 [-(self.d3 + self.d2 * jnp.cos(state[1])), self.d1 + 2 * self.d2 * jnp.cos(state[1])]])

        # State-independent noise selection matrix H (acts directly on angular accelerations)
        H = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])

        du = G @ action
        f = self.dt * (dx + du)
        
        # Stochastic Wiener increment with signal-dependent motor noise (proportional to torque magnitude)
        # w = sqrt(dt) * [ G * (sigma_m * diag(u) * xi_motor) + H * (v * xi_indep) ]
        w = jnp.sqrt(self.dt) * (G @ (params.motor_noise * jnp.diag(action) @ noise[2:]) + H @ (self.v * noise[:2]))
        return state + f + w

    def _observation(self, state, noise, params):
        """Observation model: direct state measurement corrupted by additive Gaussian sensor noise.
        
        y_t = x_t + dt * sigma_obs * xi_obs
        """
        return state + self.dt * params.obs_noise * jnp.eye(self.observation_shape[0]) @ noise

    def _cost(self, state, action, params):
        """Running stage cost: quadratic penalty on instantaneous motor effort.
        
        c(x, u) = 0.5 * action_cost * ||u||_2^2
        """
        return 0.5 * params.action_cost * jnp.sum(action ** 2)

    def _final_cost(self, state, params):
        """Terminal boundary cost: penalizes target reaching error and residual hand velocity.
        
        Phi(x_T) = ||e(x_T) - target||_2^2 + velocity_cost * ||e_dot(x_T)||_2^2
        The velocity penalty ensures the arm decelerates to a complete stop at the target.
        """
        return jnp.sum((self.e(state) - self.target) ** 2) + params.velocity_cost * jnp.sum(self.edot(state) ** 2)

    def _reset(self, noise, params):
        """Resets environment state to deterministic nominal posture x0 = [pi/4, pi/2, 0, 0]."""
        x0 = self.x0
        return x0


    def e(self, x):
        """Forward Kinematics (FK): maps joint angles to 2D Cartesian hand position.
        
        Args:
            x (jnp.ndarray): State vector [theta1, theta2, theta1_dot, theta2_dot].
                             theta1 is shoulder angle, theta2 is elbow angle.
        Returns:
            jnp.ndarray: Cartesian end-effector / hand position [p_x, p_y] in meters:
                         p_x = l1 * cos(theta1) + l2 * cos(theta1 + theta2)
                         p_y = l1 * sin(theta1) + l2 * sin(theta1 + theta2)
        """
        return jnp.array([self.l1 * jnp.cos(x[0]) + self.l2 * jnp.cos(x[0] + x[1]),
                          self.l1 * jnp.sin(x[0]) + self.l2 * jnp.sin(x[0] + x[1])])

    def gamma(self, x):
        """Kinematic Jacobian matrix: partial derivative of hand position w.r.t. joint angles.
        
        Gamma(x) = d e(x) / d theta in R^(2 x 2).
        Maps joint angular velocities [theta1_dot, theta2_dot] to Cartesian hand velocities [p_x_dot, p_y_dot].
        
        Args:
            x (jnp.ndarray): State vector [theta1, theta2, theta1_dot, theta2_dot].
        Returns:
            jnp.ndarray: (2, 2) Jacobian matrix:
                         [[-l1*sin(theta1) - l2*sin(theta1+theta2),  -l2*sin(theta1+theta2)],
                          [ l1*cos(theta1) + l2*cos(theta1+theta2),   l2*cos(theta1+theta2)]]
        """
        return jnp.array([[-self.l1 * jnp.sin(x[0]) - self.l2 * jnp.sin(x[0] + x[1]), -self.l2 * jnp.sin(x[0] + x[1])],
                          [self.l1 * jnp.cos(x[0]) + self.l2 * jnp.cos(x[0] + x[1]), self.l2 * jnp.cos(x[0] + x[1])]])

    def edot(self, x):
        """Cartesian hand velocity: time derivative of hand position.
        
        Computed via the chain rule as the matrix-vector product of the kinematic
        Jacobian Gamma(x) and joint angular velocities x[2:] = [theta1_dot, theta2_dot]:
            e_dot = Gamma(x) @ theta_dot
        
        Args:
            x (jnp.ndarray): State vector [theta1, theta2, theta1_dot, theta2_dot].
        Returns:
            jnp.ndarray: Cartesian velocity of the hand [v_x, v_y] in m/s.
        """
        return self.gamma(x) @ x[2:]

    @staticmethod
    def get_params_type():
        """Returns the NamedTuple type representing learnable cost and noise parameters."""
        return NonlinearReachingParams

    @staticmethod
    def get_params_bounds():
        """Defines the lower and upper bounds for parameter inference (MLE / L-BFGS-B).
        
        Searches over 4 decades of action_cost, 2 decades of velocity_cost,
        motor noise sigma_m in [0.01, 1.0], and observation noise sigma_obs in [0.1, 100.0].
        """
        lo = NonlinearReachingParams(action_cost=1e-5, velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1)
        hi = NonlinearReachingParams(action_cost=1e-1, velocity_cost=1e-1, motor_noise=1., obs_noise=100.)
        return lo, hi

