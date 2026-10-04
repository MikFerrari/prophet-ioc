"""Inverse iLQG: the likelihood of the partially observed model (Straub et al., NeurIPS 2023, Algorithm 1).

Notation. theta = `params`: the cost weights (in the agent's cost c), the motor noise sigma_m (in the dynamics f) and
the observation noise sigma_o (in the observation model h). The agent (the demonstrator) does not know its state x_t;
it keeps a belief b_t (an EKF mean xhat_t) from noisy observations y_t = h(x_t, w_t) and acts with a time-varying
policy u_t = pi_t(b_t, xi_t). The policy depends on t because the horizon is finite: L_t, xbar_t, ubar_t of the
iLQG solution. Two different noises act on the action: the policy noise xi_t (decision: the max-ent policy samples
u_t ~ N(mean, temperature H_t^-1)) and the motor noise v_t (execution: inside f, std proportional to |u|).

The experimenter observes x_{1:T} but neither b_t nor u_t. x alone is not Markov (x_{t+1} depends on the past
through b_t), but the joint (x_t, b_t) is:  [x_{t+1}; b_{t+1}] = g(x_t, b_t, v_t, w_t, xi_t). Algorithm 1:
    lines 1-2  solve the control problem for theta: policy gains (L_t, l_t), nominal (xbar, ubar) and the filter
               gains K_t along the nominal                                    -> apply_solver, returns SolvedModel
    line 3     for t = 0 .. T-1 (sequential: the belief at t depends on the whole past) -> lax.scan in belief_moments
    line 4     joint predictive mean g(x_t, mu_b, 0, 0, 0) and covariance
               J_b Sigma_b J_b^T + J_v J_v^T + J_w J_w^T + J_xi J_xi^T (first-order propagation, Jacobians by autodiff)
    line 5     condition the belief on the recorded x_{t+1}:
               mu_{b|x} = mu_b + Sigma_bx Sigma_xx^-1 (x_{t+1} - mu_x),  Sigma_{b|x} = Sigma_bb - Sigma_bx Sigma_xx^-1
               Sigma_xb (only the observed block Sigma_xx is inverted)
    line 6     log p(x_{1:T}) = sum_t log N(x_{t+1}; mu_x, Sigma_xx): the marginal x block of each predictive
               Gaussian. The scoring is outside the loop and vectorized: it does not feed back into the recursion
               (the chain rule p(x_{1:T}) = prod_t p(x_{t+1} | x_{1:t}), whose log is a sum).
The internal model of the agent (the filter's prediction f(xhat, u, 0) and h(xhat, 0)) uses zero noise: the agent
predicts the expected next state; the noise enters only through the true state and the observation.

create_joint_dynamics is the paper's g: an EKF in one-step predictor form (b_{t+1} uses y_t) with gains precomputed
along the nominal (kf.forward). create_filtered_joint_dynamics is the measurement-update form (b_{t+1} uses
y_{t+1}, kf.forward_filtered): with exact observations it gives b_t = x_t, so the fully observed likelihood
(inv_ilqr) is its sigma_o -> 0 limit (nested models), which the predictor form is not (one-step delay).

Run-time prediction (prophet_ioc.human_prediction) reuses the same pieces: belief_filter runs lines 3-5 over the
observed prefix and returns the belief at its end, p(b | history); joint_predictive_moments propagates the joint
Gaussian of (x, b) through the same g over the future, line 4 only (no conditioning: nothing is recorded there).
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import vmap, jacobian, lax

from prophet_ioc.infer.base import InverseOptimalControl
from prophet_ioc.infer.gaussian import command_noise_second_moment, condition, mvn_logpdf, psd_projection
from prophet_ioc.envs import Env
from prophet_ioc.belief import Belief, kf
from prophet_ioc.control import ilqr, ilqg_fixed, glqg, make_lqg_approx
from prophet_ioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from prophet_ioc.infer.utils import estimate_controls


class SolvedModel(NamedTuple):
    """Output of every apply_solver: the policy pi(t, x or xhat, xi) -> u and, for the partially observed model, the
    joint dynamics g(t, x, xhat, v, w, xi, policy, params) -> [x_{t+1}, xhat_{t+1}] (None when fully observed)."""
    policy: Callable
    joint_dynamics: Optional[Callable] = None


def create_joint_dynamics(p: Env, K: jnp.ndarray) -> Callable:
    r"""Joint dynamics of the true state and the agent's belief, EKF in one-step predictor form (paper code).

        u_t = pi_t(xhat_t, xi_t)
        x_{t+1} = f(x_t, u_t, v_t)
        xhat_{t+1} = f(xhat_t, u_t, 0) + K_t (h(x_t, w_t) - h(xhat_t, 0))

    The gains K (T, d, d_obs) are precomputed along the nominal trajectory (kf.forward), not recomputed along the
    data: the filter is the one the agent designed for its plan. The internal prediction f(xhat, u, 0) has zero noise
    (expected value).
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


def create_filtered_joint_dynamics(p: Env, K: jnp.ndarray) -> Callable:
    r"""Joint dynamics with the EKF in filtered (measurement-update) form: the belief at t+1 includes y_{t+1}.

        u_t = pi_t(xhat_t, xi_t)
        x_{t+1} = f(x_t, u_t, v_t)
        xhat-_{t+1} = f(xhat_t, u_t, 0)                                       (internal model: expected value)
        xhat_{t+1} = xhat-_{t+1} + K[t] (h(x_{t+1}, w_{t+1}) - h(xhat-_{t+1}, 0))

    K (T, d, d_obs) from kf.forward_filtered along the nominal (K[t] is the gain of the update at t + 1). With
    exact observations (K -> I for h = identity) xhat_{t+1} = x_{t+1}: the fully observed model is the limit.
    """
    f = p._dynamics
    h = p._observation

    def joint_dynamics(t, x, xhat, state_noise, obs_noise, policy_noise, policy, params):
        u = policy(t, xhat, policy_noise)
        x_next = f(x, u, state_noise, params)
        xhat_pred = f(xhat, u, jnp.zeros_like(state_noise), params)
        xhat_next = xhat_pred + K[t] @ (h(x_next, obs_noise, params) - h(xhat_pred, jnp.zeros_like(obs_noise), params))
        return jnp.hstack((x_next, xhat_next))

    return joint_dynamics


def belief_moments(env: Env, x: jnp.ndarray, model: SolvedModel, params: Any, b0: Belief, jitter: float = 1e-6,
                   relative: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Algorithm 1, lines 3-5, for one trajectory x (T+1, d): predictive moments of the joint (x_{t+1}, b_{t+1})
    given x_{0:t}, conditioning the belief on each recorded state.

    b0: Belief(mean, covariance) of the agent's belief at t = 0, i.e. the experimenter's uncertainty about it.
    jitter / relative: regularization of the conditioning (prophet_ioc.infer.gaussian).

    lax.scan, not vmap: the belief distribution at t + 1 depends on all past observations through the conditioning
    (the model is Markov only in (x, b)), so the steps are sequential; in the fully observed model (inv_ilqr) each
    transition depends only on the recorded x_t and the steps are independent (vmap).

    Returns mu (T, 2d), Sigma (T, 2d, 2d): the joint predictive moments before conditioning ([:d] is the x block).
    """
    return belief_filter(env, x, model, params, b0, jitter, relative)[1]


def belief_filter(env: Env, x: jnp.ndarray, model: SolvedModel, params: Any, b0: Belief, jitter: float = 1e-6,
                  relative: bool = False) -> Tuple[Belief, Tuple[jnp.ndarray, jnp.ndarray]]:
    """belief_moments, also returning the experimenter's distribution of the agent's belief after the last state:
    (Belief(mean, covariance) of b_T given x_{0:T}, (mu (T, 2d), Sigma (T, 2d, 2d)))."""
    d = x.shape[-1]
    g = model.joint_dynamics
    v0 = jnp.zeros(env.state_noise_shape)
    w0 = jnp.zeros(env.obs_noise_shape)
    xi0 = jnp.zeros(env.action_shape)

    def step(carry, t):
        mu_b, Sigma_b = carry
        # line 4: joint predictive mean (all noises at zero) and first-order covariance, with the signal-dependent
        # motor noise averaged over the agent's command u = pi(b, xi), uncertain through its belief and the max-ent
        # decision noise (cov_u = L Sigma_b L^T + Gamma Gamma^T, as joint_predictive_moments)
        mu_z = g(t, x[t], mu_b, v0, w0, xi0, model.policy, params)
        J_b, J_w, J_xi = jacobian(g, argnums=(2, 4, 5))(t, x[t], mu_b, v0, w0, xi0, model.policy, params)
        Lb, Gb = jacobian(model.policy, argnums=(1, 2))(t, mu_b, xi0)
        J_v_of_u = lambda u: jacobian(lambda v: g(t, x[t], mu_b, v, w0, xi0, lambda *_: u, params))(v0)
        VV = command_noise_second_moment(J_v_of_u, model.policy(t, mu_b, xi0), Lb @ Sigma_b @ Lb.T + Gb @ Gb.T)
        Sigma_z = J_b @ Sigma_b @ J_b.T + VV + J_w @ J_w.T + J_xi @ J_xi.T
        # line 5: condition the belief block on the recorded x_{t+1}
        mu_b, cov = condition(mu_z[d:], mu_z[:d], Sigma_z[d:, d:], Sigma_z[d:, :d], Sigma_z[:d, :d], x[t + 1],
                              jitter=jitter, relative=relative)
        return (mu_b, psd_projection(cov)), (mu_z, Sigma_z)

    (mu_b, Sigma_b), (mu, Sigma) = lax.scan(step, (b0.xhat, b0.Sigma), jnp.arange(x.shape[0] - 1))
    return Belief(mu_b, Sigma_b), (mu, Sigma)


def joint_predictive_moments(env: Env, model: SolvedModel, params: Any, mu0: jnp.ndarray, Sigma0: jnp.ndarray,
                             T: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Predictive moments (T+1, 2d), (T+1, 2d, 2d) of the joint z = (x, b) over T future steps from
    z_0 ~ N(mu0, Sigma0), through the joint dynamics g of `model` (Algorithm 1 line 4, first order):

        mu_{k+1} = g(k, mu_k, 0, 0, 0),   Sigma_{k+1} = J_z Sigma_k J_z^T + E[J_v J_v^T] + J_w J_w^T + J_xi J_xi^T

    without line 5: in the future nothing is recorded, so the belief is never conditioned and the uncertainty about
    both the state and the agent's belief accumulates (the agent still observes its own state through h and corrects
    its belief with K, which g contains). The x block [:d] is the predictive distribution of the state; with exact
    observations (sigma_o -> 0, K -> I, b = x) it is the fully observed inv_ilqr.closed_loop_moments. The motor
    noise covariance is averaged over the commands, E[J_v J_v^T] (gaussian.command_noise_second_moment), as there:
    the agent's command u = pi(b, xi) is uncertain through its belief, cov_u = L Sigma_bb L^T + Gamma Gamma^T."""
    d = mu0.shape[0] // 2
    g = model.joint_dynamics
    v0 = jnp.zeros(env.state_noise_shape)
    w0 = jnp.zeros(env.obs_noise_shape)
    xi0 = jnp.zeros(env.action_shape)

    def step(carry, t):
        mu, S = carry
        gz = lambda z, v, w, xi: g(t, z[:d], z[d:], v, w, xi, model.policy, params)
        J_z, J_w, J_xi = jacobian(gz, argnums=(0, 2, 3))(mu, v0, w0, xi0)
        Lb, Gb = jacobian(model.policy, argnums=(1, 2))(t, mu[d:], xi0)
        cov_u = Lb @ S[d:, d:] @ Lb.T + Gb @ Gb.T            # the command acts on the belief
        J_v_of_u = lambda u: jacobian(lambda v: g(t, mu[:d], mu[d:], v, w0, xi0, lambda *_: u, params))(v0)
        VV = command_noise_second_moment(J_v_of_u, model.policy(t, mu[d:], xi0), cov_u)
        mu = gz(mu, v0, w0, xi0)
        S = J_z @ S @ J_z.T + VV + J_w @ J_w.T + J_xi @ J_xi.T
        S = 0.5 * (S + S.T)
        return (mu, S), (mu, S)

    _, (mu, Sigma) = lax.scan(step, (mu0, Sigma0), jnp.arange(T))
    return jnp.concatenate([mu0[None], mu]), jnp.concatenate([Sigma0[None], Sigma])


class InverseILQG(InverseOptimalControl):
    r"""Probabilistic inverse optimal control with the partially observed iLQG model (paper Algorithm 1).

    The agent solves the iLQG problem from the belief b0 = (x0, Sigma0) of the constructor (one policy for all
    trials, the paper setting); the likelihood marginalizes its belief by belief_moments. For per-trial / per-window
    policies solved from each trajectory's own start, see prophet_ioc.infer.multi_env.trial_loglikelihood.
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

    def initial_belief(self, x: jnp.ndarray) -> Belief:
        """Initial belief of each trial of x (N, T+1, d) when none is given: the constructor's mean and the identity
        covariance (the original code)."""
        n = x.shape[0]
        return Belief(jnp.broadcast_to(self.b0, (n, self.bdim)), jnp.broadcast_to(jnp.eye(self.bdim),
                                                                                    (n, self.bdim, self.bdim)))

    def moments(self, x: jnp.ndarray, model: SolvedModel, params: Any,
                b0: Optional[Belief] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""Algorithm 1, lines 3-5, for one trajectory x (T+1, d) (see belief_moments; b0 default: the constructor's
        mean and identity covariance). Returns the joint predictive mu (T, 2d), Sigma (T, 2d, 2d)."""
        if b0 is None:
            b0 = Belief(self.b0, jnp.eye(self.bdim))
        return belief_moments(self.env, x, model, params, b0)

    def loglikelihood(self, x: jnp.ndarray, params: Any, b0: Optional[Belief] = None) -> jnp.ndarray:
        r"""Algorithm 1: log p(X | theta) = sum_i sum_t log N(x_{t+1}^i; mu_x^i(t), Sigma_xx^i(t)).

        Args:
            x: trajectories (N, T+1, d).
            params: theta (cost weights, sigma_m, sigma_o).
            b0: initial belief per trial, Belief((N, d), (N, d, d)); default initial_belief(x).
        """
        model = self.apply_solver(x, params)                                     # lines 1-2
        b0 = self.initial_belief(x) if b0 is None else b0
        mu, Sigma = vmap(lambda xi, b: self.moments(xi, model, params, Belief(*b)))(x, tuple(b0))   # lines 3-5
        d = x.shape[-1]
        return jnp.sum(vmap(vmap(lambda o, m, S: mvn_logpdf(o, m, S, 1e-6, relative=False)))(
            x[:, 1:], mu[:, :, :d], Sigma[:, :, :d, :d]))                       # line 6

    def apply_solver(self, x: jnp.ndarray, params: Any) -> SolvedModel:
        r"""Algorithm 1, lines 1-2: iLQG solution for theta (nominal xbar, ubar, Gains(L feedback, l feedforward
        correction ~ 0 at convergence, H = Q_uu)) and the filter gains K along the nominal."""
        T = x.shape[1] - 1

        gains, xbar, ubar = self.solve(p=self.env, x0=self.x0, Sigma0=self.Sigma0,
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params, max_iter=self.max_iter)
        policy = self.create_policy(gains, xbar, ubar)

        lqgspec = make_lqg_approx(p=self.env, params=params)(xbar, ubar)
        K = self.kf.forward(spec=lqgspec, gains=gains, xhat0=self.x0, Sigma0=self.Sigma0)

        return SolvedModel(policy, create_joint_dynamics(self.env, K))


class FixedLinearizationInverseILQG(InverseILQG):
    r"""Inverse iLQG with the data-based linearization (paper section 3.3).

    Instead of solving the control problem for every theta, the dynamics are linearized and the cost quadratized
    along the demonstration itself, with the unobserved controls estimated by Gauss-Newton,
        u_t = argmin_u |x_{t+1} - f(x_t, u, 0)|^2,
    and one backward pass gives the gains (one Riccati sweep per likelihood evaluation, per trajectory).
    """

    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0., max_iter: int = 0):
        super().__init__(env, b0, solve=ilqg_fixed.solve, maxent_temp=0., max_iter=max_iter)

        if maxent_temp > 0:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar, maxent_temp)
        else:
            self.create_policy = create_lqr_policy

    def apply_solver(self, x: jnp.ndarray, params: Any) -> SolvedModel:
        """Lines 1-2 for one trajectory x (T+1, d), linearized along x."""
        K, gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params),
                                          Sigma0=self.Sigma0, params=params)
        return SolvedModel(self.create_policy(gains, xbar, ubar), create_joint_dynamics(self.env, K))

    def loglikelihood(self, x: jnp.ndarray, params: Any, b0: Optional[Belief] = None):
        b0 = self.initial_belief(x) if b0 is None else b0
        mu, Sigma = vmap(lambda xi, b: self.moments(xi, self.apply_solver(xi, params), params, Belief(*b)))(
            x, tuple(b0))
        d = self.xdim
        return jnp.sum(vmap(vmap(lambda o, m, S: mvn_logpdf(o, m, S, 1e-6, relative=False)))(
            x[:, 1:], mu[:, :, :d], Sigma[:, :, :d, :d]))


class FixedLinearizationInverseGILQG(FixedLinearizationInverseILQG):
    r"""Data-linearized inverse generalized iLQG (signal-dependent motor noise).

    Biological motor noise grows with the control (Harris & Wolpert 1998): w_t ~ N(0, Sigma_w + sum_i u_{i,t}^2 C_i).
    The generalized LQG (glqg) accounts for it in the backward pass,
        Q_uu = R_t + B_t^T S_{t+1} B_t + sum_i C_i^T S_{t+1} C_i,
    so that larger controls are penalized by the variance they inject, and sigma_m can be recovered with the costs.
    """

    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0., max_iter: int = 0):
        super().__init__(env, b0, maxent_temp=maxent_temp, max_iter=max_iter)

    def apply_solver(self, x: jnp.ndarray, params: Any) -> SolvedModel:
        K, gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params,
                                          Sigma0=self.Sigma0,
                                          lqg_module=glqg)
        return SolvedModel(self.create_policy(gains, xbar, ubar), create_joint_dynamics(self.env, K))
