from typing import Callable, Any, Tuple
import jax.numpy as jnp
from jax import vmap, jacobian, lax, scipy as js

from prophet_ioc.infer.base import InverseOptimalControl
from prophet_ioc.envs import Env
from prophet_ioc.belief import Belief, kf
from prophet_ioc.control import ilqr, ilqg_fixed, glqg, make_lqg_approx
from prophet_ioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from prophet_ioc.infer.utils import estimate_controls


def create_joint_dynamics(p: Env, K: jnp.ndarray) -> Callable:
    r"""Constructs the combined closed-loop dynamics of the true system and observer.

    In stochastic optimal control under partial observability (LQG framework), the agent
    does not directly access the true state $x_t \in \mathbb{R}^d$, but maintains an internal
    belief state estimate $\hat{x}_t \in \mathbb{R}^d$ via a Kalman filter.

    The joint state $z_t = \begin{bmatrix} x_t \\ \hat{x}_t \end{bmatrix} \in \mathbb{R}^{2d}$ evolves as:
        1. Control generation:
           $$u_t = \pi(t, \hat{x}_t, \xi_t) = \bar{u}_t - L_t (\hat{x}_t - \bar{x}_t) + \xi_t$$
        2. Real physical state transition:
           $$x_{t+1} = f(x_t, u_t, w_t, \theta)$$
           where $w_t \sim \mathcal{N}(0, \Sigma_w)$ is process/motor noise.
        3. Observation generation:
           $$y_t = h(x_t, v_t, \theta)$$
           where $v_t \sim \mathcal{N}(0, \Sigma_v)$ is sensory observation noise.
        4. Observer (Kalman filter) update:
           $$\hat{x}_{t+1} = f(\hat{x}_t, u_t, 0, \theta) + K_t \left( h(x_t, v_t, \theta) - h(\hat{x}_t, 0, \theta) \right)$$
           where $K_t$ is the Kalman observer gain computed from the Riccati filter equations.

    Args:
        p: Environment instance defining nonlinear dynamics $f$ and observation model $h$.
        K: Kalman gain schedule of shape $(T, d, d_{\text{obs}})$.

    Returns:
        Callable joint_dynamics(t, x, xhat, state_noise, obs_noise, policy_noise, policy, params)
        returning concatenated vector $[x_{t+1}^\top, \hat{x}_{t+1}^\top]^\top \in \mathbb{R}^{2d}$.
    """
    f = p._dynamics
    h = p._observation

    def joint_dynamics(t, x, xhat, state_noise, obs_noise, policy_noise, policy, params):
        u = policy(t, xhat, policy_noise)
        x_next = f(x, u, state_noise, params)
        xhat_next = f(xhat, u, jnp.zeros_like(state_noise), params) + K[t] @ (
                h(x, obs_noise, params) - h(xhat, jnp.zeros_like(obs_noise), params))

        return jnp.hstack((x_next, xhat_next))

    return joint_dynamics


class InverseILQG(InverseOptimalControl):
    r"""Probabilistic Inverse Optimal Control via Iterative Linear-Quadratic-Gaussian (iLQG).

    This estimator recovers unknown cost parameters and noise characteristics $\theta$
    from demonstrated trajectories $X = \{x_{0:T}^{(i)}\}_{i=1}^N$ by maximizing the
    marginal trajectory likelihood under the optimal closed-loop LQG controller:
        $$\hat{\theta} = \arg\max_\theta \sum_{i=1}^N \log p\left(x_{0:T}^{(i)} \mid \theta\right)$$

    Unlike traditional Inverse Reinforcement Learning (IRL) methods that treat action choices
    independently or assume fully observable open-loop state sequences, Inverse iLQG
    explicitly models the coupled closed-loop dynamics between the physical system and
    the actor's internal Bayesian observer (Extended Kalman Filter), capturing sensory delays,
    motor noise, and certainty-equivalence feedback.
    """

    def __init__(self, env: Env, b0: Belief, solve: Callable = ilqr.solve, kf=kf, maxent_temp: float = 0.,
                 max_iter: int = 10):
        self.env = env
        self.solve = solve
        self.kf = kf

        self.x0, self.Sigma0 = b0
        self.b0 = self.x0

        self.xdim = self.x0.shape[0]
        self.bdim = self.xdim

        if maxent_temp > 0:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar, maxent_temp)
        else:
            self.create_policy = create_lqr_policy

        self.max_iter = max_iter

    def moments(self, x: jnp.ndarray, joint_dynamics: Callable,
                policy: Callable, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""Propagates Gaussian belief moments and conditions on observed trajectory points.

        At each discrete time step $t$, the joint state $z_t = [x_t, \hat{x}_t]^\top$ has prior
        mean $\mu_z \in \mathbb{R}^{2d}$ and covariance $\Sigma_z \in \mathbb{R}^{2d \times 2d}$:
            $$\mu_z = \mathbb{E}[z_{t+1} \mid x_t, \mu_t]$$
            $$\Sigma_z = G_b \Sigma_t G_b^\top + G_w G_w^\top + G_v G_v^\top + G_u G_u^\top$$
        where $G_b, G_w, G_v, G_u$ are Jacobians with respect to belief state, process noise,
        observation noise, and policy exploration noise, respectively.

        Since the demonstrator's true physical state $x_{t+1}$ is observed, we compute the
        conditional posterior over the internal belief $\hat{x}_{t+1} \mid x_{t+1}$ using
        the standard multivariate Gaussian conditioning identity:
            $$\mu = \mu_z[\hat{x}] + \Sigma_{z, \hat{x}x} (\Sigma_{z, xx} + \epsilon I)^{-1} (x_{t+1} - \mu_{z, x})$$
            $$\Sigma = \Sigma_{z, \hat{x}\hat{x}} - \Sigma_{z, \hat{x}x} (\Sigma_{z, xx} + \epsilon I)^{-1} \Sigma_{z, x\hat{x}}$$

        Returns:
            mu: Sequence of joint mean vectors $\mu_z(t)$ of shape $(T, 2d)$.
            Sigma: Sequence of joint covariance matrices $\Sigma_z(t)$ of shape $(T, 2d, 2d)$.
        """
        d = self.xdim

        def step(carry, t):
            mu, Sigma = carry

            state_noise_zero = jnp.zeros(self.env.state_noise_shape)
            obs_noise_zero = jnp.zeros(self.env.obs_noise_shape)
            policy_noise_zero = jnp.zeros(self.env.action_shape)

            # 1. First-order Taylor propagation of joint mean
            mu_z = joint_dynamics(t, x[t], mu,
                                  state_noise_zero,
                                  obs_noise_zero,
                                  policy_noise_zero,
                                  policy, params)

            # 2. Linearized covariance propagation via automatic differentiation
            gb, gm, gn, go = jacobian(joint_dynamics, argnums=(2, 3, 4, 5))(t, x[t], mu,
                                                                            state_noise_zero,
                                                                            obs_noise_zero,
                                                                            policy_noise_zero,
                                                                            policy, params)
            Sigma_z = gb @ Sigma @ gb.T + gm @ gm.T + gn @ gn.T + go @ go.T

            # 3. Condition internal belief on the observed future physical state x[t+1]
            mu = mu_z[d:] + Sigma_z[d:, :d] @ jnp.linalg.solve(Sigma_z[:d, :d] + jnp.eye(d) * 1e-6,
                                                               x[t + 1] - mu_z[:d])
            Sigma = Sigma_z[d:, d:] - Sigma_z[d:, :d] @ jnp.linalg.solve(Sigma_z[:d, :d] + jnp.eye(d) * 1e-6,
                                                                         Sigma_z[:d, d:])

            return (mu, Sigma), (mu_z, Sigma_z)

        _, (mu, Sigma) = lax.scan(step, (self.b0, jnp.eye(self.bdim)), jnp.arange(x.shape[0] - 1))

        return mu, Sigma

    def loglikelihood(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        r"""Evaluates the conditional log-likelihood of demonstrator trajectories.

        Under the Markovian assumption conditioned on belief propagation:
            $$\log p(X \mid \theta) = \sum_{i=1}^N \sum_{t=0}^{T-1} \log \mathcal{N}\left(x_{t+1}^{(i)};\, \mu_z^{(i)}(t)_{[:d]},\, \Sigma_z^{(i)}(t)_{[:d, :d]} + \epsilon I\right)$$

        Args:
            x: Demonstrated batch of trajectories of shape $(N, T+1, d)$.
            params: Candidate cost and noise parameters $\theta$.

        Returns:
            Scalar sum of log-likelihood across all transitions and trials.
        """
        # get policy for current params
        policy, joint_dynamics = self.apply_solver(x, params)

        mu, Sigma = vmap(lambda xi: self.moments(xi, joint_dynamics, policy, params))(x)

        d = x.shape[-1]

        return jnp.sum(js.stats.multivariate_normal.logpdf(x[:, 1:],
                                                           mu[:, :, :d],
                                                           Sigma[:, :, :d, :d] + jnp.eye(d) * 1e-6))

    def apply_solver(self, x: jnp.ndarray, params: Any) -> Tuple[Callable, Callable]:
        r"""Solves the forward optimal control problem for given candidate parameters.

        Finds the nominal trajectory $(\bar{x}, \bar{u})$, backward Riccati feedback gains $L_t$,
        and forward Kalman filter observer gains $K_t$.
        """
        T = x.shape[1] - 1

        gains, xbar, ubar = self.solve(p=self.env, x0=self.x0, Sigma0=self.Sigma0,
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params, max_iter=self.max_iter)
        policy = self.create_policy(gains, xbar, ubar)

        lqgspec = make_lqg_approx(p=self.env, params=params)(xbar, ubar)
        K = self.kf.forward(spec=lqgspec, gains=gains, xhat0=self.x0, Sigma0=self.Sigma0)

        joint_dynamics = create_joint_dynamics(self.env, K)

        return policy, joint_dynamics


class FixedLinearizationInverseILQG(InverseILQG):
    r"""Fast Inverse iLQG with Fixed Linearization along the Demonstrated Trajectory.

    In standard Inverse iLQG, evaluating candidate parameters $\theta$ requires running
    the full iterative backward-forward Riccati solver (`ilqr.solve` or `ilqg.solve`) to
    find a new nominal trajectory $(\bar{x}, \bar{u})$ from scratch at every optimization step.

    Fixed Linearization exploits the demonstrator trajectory itself:
        1. Assume the observed demonstration $X$ is already close to the optimal trajectory $\bar{X}$.
        2. Invert the dynamics via Gauss-Newton to estimate nominal controls:
           $$u_t^{\text{est}} = \arg\min_u \|x_{t+1} - f(x_t, u, 0, \theta)\|^2$$
        3. Linearize the dynamics and quadratize the candidate cost function directly along $(X, U^{\text{est}})$.
        4. Execute a single backward Riccati pass to obtain feedback gains $L_t$ and Kalman gains $K_t$.

    This reduces the computational complexity from $O(K \cdot N_{\text{iter}})$ forward-backward
    passes to a single Riccati sweep, accelerating parameter recovery by orders of magnitude.
    """

    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0., max_iter: int = 0):
        super().__init__(env, b0, solve=ilqg_fixed.solve, maxent_temp=0., max_iter=max_iter)

        if maxent_temp > 0:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar, maxent_temp)
        else:
            self.create_policy = create_lqr_policy

    def apply_solver(self, x: jnp.ndarray, params: Any) -> Tuple[Callable, Callable]:
        # get policy for current params
        K, gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params),
                                          Sigma0=self.Sigma0, params=params)
        policy = self.create_policy(gains, xbar, ubar)
        joint_dynamics = create_joint_dynamics(self.env, K)

        return joint_dynamics, policy

    def loglikelihood(self, x: jnp.ndarray, params: Any):
        mu, Sigma = vmap(lambda xi: self.moments(xi, *self.apply_solver(xi, params), params))(x)

        d = self.xdim
        return jnp.sum(js.stats.multivariate_normal.logpdf(x[:, 1:],
                                                           mu[:, :, :d],
                                                           Sigma[:, :, :d, :d] + jnp.eye(d) * 1e-6))


class FixedLinearizationInverseGILQG(FixedLinearizationInverseILQG):
    r"""Fixed Linearization Inverse Generalized iLQG (with Signal-Dependent Motor Noise).

    Biological motor systems exhibit control-dependent (multiplicative) motor noise:
        $$w_t \sim \mathcal{N}\left(0,\, \Sigma_w + \sum_i u_{i, t}^2 C_i\right)$$
    which causes motor variance to scale with the magnitude of muscle activation (Harris & Wolpert 1998).

    Standard LQG assumes additive, state/control-independent noise and separates estimation from
    control (Certainty Equivalence). Under signal-dependent noise, separation fails: larger control
    efforts inject larger state variance into future steps.

    `FixedLinearizationInverseGILQG` applies Generalized LQG (`glqg`), modifying the Riccati backward
    pass:
        $$Q_{uu} = R_t + B_t^\top S_{t+1} B_t + \sum_i C_i^\top S_{t+1} C_i$$
    accounting for signal-dependent noise when computing optimal feedback gains $L_t$ and recovering
    noise parameter $\sigma_m$ alongside cost parameters.
    """

    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0., max_iter: int = 0):
        super().__init__(env, b0, maxent_temp=maxent_temp, max_iter=max_iter)

    def apply_solver(self, x: jnp.ndarray, params: Any) -> Tuple[Callable, Callable]:
        # get policy for current params
        K, gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params,
                                          Sigma0=self.Sigma0,
                                          lqg_module=glqg)
        policy = self.create_policy(gains, xbar, ubar)
        joint_dynamics = create_joint_dynamics(self.env, K)

        return joint_dynamics, policy

