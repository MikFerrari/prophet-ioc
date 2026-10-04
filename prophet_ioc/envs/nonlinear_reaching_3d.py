from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jacobian

from prophet_ioc.envs.base import Env


class NonlinearReaching3DParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.0


class NonlinearReaching3D(Env):
    """3D Human Arm Reaching Task (3-DOF Anthropomorphic Arm).

    Kinematics:
        - Shoulder yaw (q1, azimuth in horizontal plane)
        - Shoulder pitch (q2, elevation in vertical plane)
        - Elbow pitch (q3, forearm flexion in vertical plane)
        - Shoulder keypoint: [0, 0, 0] (origin)
        - Elbow keypoint: [l1 * cos(q2) * cos(q1), l1 * cos(q2) * sin(q1), l1 * sin(q2)]
        - Hand keypoint: Elbow + [l2 * cos(q2+q3) * cos(q1), l2 * cos(q2+q3) * sin(q1), l2 * sin(q2+q3)]

    Dynamics:
        State: [q1, q2, q3, dq1, dq2, dq3] in R^6
        Action: [u1, u2, u3] in R^3 (joint torques)
        Equation: M(q) * ddq + C(q, dq) * dq + B * dq = u
    """

    def __init__(
        self,
        dt: float = 0.01,
        target: Optional[jnp.ndarray] = None,
        x0: Optional[jnp.ndarray] = None,
        upper_arm_length: float = 0.30,
        forearm_length: float = 0.33,
        I1: float = 0.025,
        I2: float = 0.045,
        I_base: float = 0.05,
        indep_noise: float = 0.0,
        w_target: float = 10.0,
    ):
        self.dt = dt
        self.l1 = upper_arm_length
        self.l2 = forearm_length

        # Centers of mass of links
        self.s1 = 0.11 / 0.30 * self.l1
        self.s2 = 0.16 / 0.33 * self.l2

        # Masses of arm segments
        self.m1 = 1.4 / 0.30 * self.l1
        self.m2 = 1.1 / 0.33 * self.l2

        # 2D planar inertia constants (from Li 2006)
        self.d1 = I1 + I2 + self.m2 * self.l1**2
        self.d2 = self.m2 * self.l1 * self.s2
        self.d3 = I2
        self.I_base = I_base

        # Joint damping / friction constants
        self.b11 = 0.05
        self.b22 = 0.05
        self.b33 = 0.05
        self.b23 = 0.025
        self.b32 = 0.025

        self.v = indep_noise
        self.w_target = w_target

        if target is not None:
            self.target = jnp.asarray(target, dtype=jnp.float32)
        else:
            self.target = jnp.array([0.40, 0.12, 0.25], dtype=jnp.float32)

        self.q_target = self.ik(self.target)

        if x0 is not None:
            self.x0 = jnp.asarray(x0, dtype=jnp.float32)
        else:
            x0_pos = jnp.array([0.34, 0.02, 0.18], dtype=jnp.float32)
            q0 = self.ik(x0_pos)
            self.x0 = jnp.concatenate([q0, jnp.zeros(3, dtype=jnp.float32)])

        super().__init__(
            state_shape=(6,),
            action_shape=(3,),
            observation_shape=(6,),
            state_noise_shape=(6,),
            obs_noise_shape=(6,),
        )

    def ik(self, target_pos: jnp.ndarray) -> jnp.ndarray:
        """Closed-form analytical inverse kinematics for 3-DOF arm.

        Args:
            target_pos: [x, y, z] target coordinates in meters.

        Returns:
            [q1, q2, q3] joint angles in radians.
        """
        x, y, z = target_pos[0], target_pos[1], target_pos[2]
        theta1 = jnp.arctan2(y, x)
        r = jnp.sqrt(x**2 + y**2)
        D2 = r**2 + z**2
        cos_theta3 = jnp.clip(
            (D2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2), -1.0, 1.0
        )
        theta3 = jnp.arccos(cos_theta3)
        alpha = jnp.arctan2(z, r)
        beta = jnp.arctan2(self.l2 * jnp.sin(theta3), self.l1 + self.l2 * jnp.cos(theta3))
        theta2 = alpha - beta
        return jnp.array([theta1, theta2, theta3], dtype=jnp.float32)

    def keypoints(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Calculates 3D Cartesian coordinates of Shoulder, Elbow, and Hand keypoints.

        Args:
            state: [q1, q2, q3, dq1, dq2, dq3]

        Returns:
            Tuple (p_shoulder, p_elbow, p_hand), each a 3D vector [x, y, z] in meters.
        """
        q1, q2, q3 = state[0], state[1], state[2]
        c1, s1 = jnp.cos(q1), jnp.sin(q1)
        c2, s2 = jnp.cos(q2), jnp.sin(q2)
        c23, s23 = jnp.cos(q2 + q3), jnp.sin(q2 + q3)

        p_shoulder = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        p_elbow = jnp.array(
            [self.l1 * c2 * c1, self.l1 * c2 * s1, self.l1 * s2], dtype=jnp.float32
        )
        p_hand = jnp.array(
            [
                (self.l1 * c2 + self.l2 * c23) * c1,
                (self.l1 * c2 + self.l2 * c23) * s1,
                self.l1 * s2 + self.l2 * s23,
            ],
            dtype=jnp.float32,
        )
        return p_shoulder, p_elbow, p_hand

    def e(self, state: jnp.ndarray) -> jnp.ndarray:
        """Forward kinematics for Hand (end-effector) 3D position [x, y, z] in meters."""
        _, _, p_hand = self.keypoints(state)
        return p_hand

    def elbow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Forward kinematics for Elbow 3D position [x, y, z] in meters."""
        _, p_elbow, _ = self.keypoints(state)
        return p_elbow

    def gamma(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space 3D Jacobian J = de/dq (3x3)."""
        return jacobian(self.e)(state)[:3, :3]

    def gamma_elbow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space 3D Jacobian for Elbow J_elbow = d(elbow)/dq (3x3)."""
        return jacobian(self.elbow)(state)[:3, :3]

    def edot(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space 3D velocity de/dt = J(q) * dq."""
        return self.gamma(state) @ state[3:]

    def _dynamics(self, state, action, noise, params):
        q = state[:3]
        dq = state[3:]

        q1, q2, q3 = q[0], q[1], q[2]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]

        # 1. Inertia matrix M(q)
        # Yaw inertia about vertical z-axis
        r1 = self.s1 * jnp.cos(q2)
        r2 = self.l1 * jnp.cos(q2) + self.s2 * jnp.cos(q2 + q3)
        Iz = self.I_base + self.m1 * r1**2 + self.m2 * r2**2

        # 2D planar arm block
        det_planar = self.d1 * self.d3 - self.d3**2 - (self.d2 * jnp.cos(q3))**2
        inv_M22 = self.d3 / det_planar
        inv_M23 = -(self.d3 + self.d2 * jnp.cos(q3)) / det_planar
        inv_M33 = (self.d1 + 2 * self.d2 * jnp.cos(q3)) / det_planar

        inv_M = jnp.array(
            [
                [1.0 / Iz, 0.0, 0.0],
                [0.0, inv_M22, inv_M23],
                [0.0, inv_M23, inv_M33],
            ],
            dtype=jnp.float32,
        )

        # 2. Coriolis & centrifugal forces
        c2_term = -self.d2 * jnp.sin(q3) * (2.0 * dq2 * dq3 + dq3**2)
        c3_term = self.d2 * jnp.sin(q3) * dq2**2

        dIz_dt = 2.0 * (
            -self.m1 * r1 * self.s1 * jnp.sin(q2) * dq2
            - self.m2
            * r2
            * (
                self.l1 * jnp.sin(q2) * dq2
                + self.s2 * jnp.sin(q2 + q3) * (dq2 + dq3)
            )
        )
        c1_term = dIz_dt * dq1

        coriolis = jnp.array([c1_term, c2_term, c3_term], dtype=jnp.float32)

        # 3. Damping / joint friction
        friction = jnp.array(
            [
                self.b11 * dq1,
                self.b22 * dq2 + self.b23 * dq3,
                self.b32 * dq2 + self.b33 * dq3,
            ],
            dtype=jnp.float32,
        )

        # Joint acceleration: ddq = inv_M * (u - coriolis - friction)
        h_q = -inv_M @ (coriolis + friction)
        G_act = inv_M

        dx = jnp.concatenate([dq, h_q])
        G = jnp.vstack([jnp.zeros((3, 3), dtype=jnp.float32), G_act])
        H = jnp.vstack([jnp.zeros((3, 3), dtype=jnp.float32), jnp.eye(3, dtype=jnp.float32)])

        du = G @ action
        f = self.dt * (dx + du)

        # Stochastic noise injection
        motor_noise_mat = params.motor_noise * jnp.diag(action)
        w_motor = G @ (motor_noise_mat @ noise[3:])
        w_indep = H @ (self.v * noise[:3])
        w = jnp.sqrt(self.dt) * (w_motor + w_indep)

        return state + f + w

    def _observation(self, state, noise, params):
        return state + self.dt * params.obs_noise * jnp.eye(self.observation_shape[0]) @ noise

    def _cost(self, state, action, params):
        return 0.5 * params.action_cost * jnp.sum(action**2)

    def _final_cost(self, state, params):
        # Guaranteed positive-definite Hessian Qf for stable, non-divergent Riccati backward sweeps
        q_err = state[:3] - self.q_target
        v_err = state[3:]
        return self.w_target * jnp.sum(q_err**2) + params.velocity_cost * jnp.sum(v_err**2)

    def _reset(self, noise, params):
        return self.x0

    @staticmethod
    def get_params_type():
        return NonlinearReaching3DParams

    @staticmethod
    def get_params_bounds():
        lo = NonlinearReaching3DParams(
            action_cost=1e-5, velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1
        )
        hi = NonlinearReaching3DParams(
            action_cost=1e-1, velocity_cost=1e-1, motor_noise=1.0, obs_noise=100.0
        )
        return lo, hi
