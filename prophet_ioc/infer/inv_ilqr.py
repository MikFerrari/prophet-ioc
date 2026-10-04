"""Inverse iLQR: the likelihood of the fully observed model (paper Appendix E).

The agent observes its state exactly, so x_{t+1} given x_{0:t} depends only on x_t (Markov in x): every transition is
scored independently with the policy evaluated at the recorded x_t (vmap over t). This is the sigma_o -> 0 limit of
the partially observed model (inv_ilqg, filtered form).
"""
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import lax, vmap, jacobian

from prophet_ioc.envs import Env
from prophet_ioc.control import glqr, gilqr, ilqr, ilqr_fixed
from prophet_ioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from prophet_ioc.infer.base import InverseOptimalControl
from prophet_ioc.infer.gaussian import command_noise_second_moment, mvn_logpdf
from prophet_ioc.infer.inv_ilqg import SolvedModel
from prophet_ioc.infer.utils import estimate_controls


def fully_observed_moments(env: Env, x: jnp.ndarray, policy: Callable, params: Any
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Predictive mean and covariance (T, d), (T, d, d) of x_{t+1} given the recorded x_t, for one trajectory x.

    mean   f(x_t, pi_t(x_t, 0), 0)
    cov    E[V V^T] + W W^T:
           V = df/dv, the motor noise Jacobian (execution noise inside f; for HumanKinematicReaching signal-dependent,
             std proportional to |u|, scaled by sqrt(dt) like white noise so sigma_m does not depend on the time grid,
             plus an additive part for u ~ 0), averaged over the commands of the max-ent policy
             (gaussian.command_noise_second_moment with cov_u = Gamma Gamma^T): the signal-dependent noise scales with
             the command actually issued, u + Gamma xi, not with the nominal u. Exact for noise linear in the command;
             a first-order V V^T at xi = 0 dropped sigma_m^2 Gamma Gamma^T, which dominates the directions the
             decision noise does not reach (B is rank 19 of 38) when Gamma xi >> u (data simulated from the model
             scored ~170x their expected squared Mahalanobis residual with temperature 1e-6);
           W = df/dxi = B Gamma_t, the policy noise of the max-ent policy (decision noise, u ~ N(., temperature H^-1));
             zero with the deterministic policy, which ignores its noise argument (then E[V V^T] = V V^T).
    vmap over t: the steps are independent given the recorded states (Markov in x).
    """
    v0 = jnp.zeros(env.state_noise_shape)
    xi0 = jnp.zeros(env.action_shape)

    def step(xt, t):
        ut = policy(t, xt, xi0)
        mu = env._dynamics(xt, ut, v0, params)
        Gt = jacobian(policy, argnums=2)(t, xt, xi0)
        VV = command_noise_second_moment(lambda u: jacobian(env._dynamics, argnums=2)(xt, u, v0, params), ut,
                                         Gt @ Gt.T)
        W = jacobian(lambda xi: env._dynamics(xt, policy(t, xt, xi), v0, params))(xi0)
        return mu, VV + W @ W.T

    return vmap(step)(x[:-1], jnp.arange(x.shape[0] - 1))


def closed_loop_step(env: Env, policy: Callable, params: Any, xt: jnp.ndarray, t: Any):
    """One step of the closed loop x' = f(x, pi_t(x, xi), v) at xt, all noises at zero: (mean, F, V, W) with
    F = df/dx through the policy (A + B L_t for the affine policy), V = df/dv (motor noise), W = df/dxi = B Gamma_t
    (max-ent policy noise; zero for a deterministic policy)."""
    v0 = jnp.zeros(env.state_noise_shape)
    xi0 = jnp.zeros(env.action_shape)
    closed = lambda x, v, xi: env._dynamics(x, policy(t, x, xi), v, params)
    F, V, W = jacobian(closed, argnums=(0, 1, 2))(xt, v0, xi0)
    return closed(xt, v0, xi0), F, V, W


def closed_loop_moments(env: Env, policy: Callable, params: Any, X: jnp.ndarray, Sigma0: jnp.ndarray
                        ) -> jnp.ndarray:
    """Predictive covariance (T+1, d, d) of the fully observed model over the nominal X (T+1, d) of `policy`, from an
    uncertain initial state x_0 ~ N(X[0], Sigma0) (run-time prediction, human_prediction):

        Sigma_{k+1} = F_k Sigma_k F_k^T + E[V_k V_k^T] + W_k W_k^T,    Sigma_0 = Sigma0

    first order around the nominal (closed_loop_step at X[k]): the agent observes its state, so a deviation from the
    nominal is fed back by its policy (F_k = A_k + B_k L_k, the closed-loop linearization) while the motor noise V_k
    and the max-ent decision noise W_k = B_k Gamma_k enter at every step. The motor noise is signal dependent:
    its covariance is averaged over the commands the closed loop issues, E[V V^T] (command_noise_second_moment:
    sigma_m^2 E[u u^T], the nominal command ubar plus the spread L Sigma_k L^T + Gamma Gamma^T of the feedback and
    decision noise; V_k V_k^T at the nominal command plus that second-order term, exact for the linear ZOH dynamics).
    Unlike the likelihood (fully_observed_moments) nothing is recorded in the future, so the steps are chained
    (lax.scan) instead of restarting from the data at every step. The mean is the nominal itself
    (policy(t, X[t], 0) = U[t] when the policy is built around (X, U) with l = 0)."""
    v0 = jnp.zeros(env.state_noise_shape)
    xi0 = jnp.zeros(env.action_shape)

    def step(S, t):
        _, F, _, W = closed_loop_step(env, policy, params, X[t], t)
        # command distribution of the closed loop: u = U + L dx + Gamma xi -> cov_u = L S L^T + Gamma Gamma^T
        Lt, Gt = jacobian(policy, argnums=(1, 2))(t, X[t], xi0)
        V_of_u = lambda u: jacobian(env._dynamics, argnums=2)(X[t], u, v0, params)
        VV = command_noise_second_moment(V_of_u, policy(t, X[t], xi0), Lt @ S @ Lt.T + Gt @ Gt.T)
        S = F @ S @ F.T + VV + W @ W.T
        S = 0.5 * (S + S.T)
        return S, S

    _, Sigma = lax.scan(step, Sigma0, jnp.arange(X.shape[0] - 1))
    return jnp.concatenate([Sigma0[None], Sigma])


class InverseILQR(InverseOptimalControl):
    def __init__(self, env: Env, solve: Callable = ilqr.solve, maxent_temp: float = 0., max_iter: int = 10):
        self.env = env
        self.solve = solve
        if maxent_temp > 0.:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar,
                                                                                    temperature=maxent_temp)
        else:
            self.create_policy = create_lqr_policy

        self.max_iter = max_iter

    def moments(self, x: jnp.ndarray, model: SolvedModel, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predictive moments of one trajectory x (T+1, d), see fully_observed_moments."""
        return fully_observed_moments(self.env, x, model.policy, params)

    def apply_solver(self, x: jnp.ndarray, params: Any) -> SolvedModel:
        """One policy for all trials, solved from the mean initial state."""
        T = x.shape[1] - 1

        gains, xbar, ubar = self.solve(self.env, x0=x[:, 0].mean(axis=0),
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params, max_iter=self.max_iter)
        return SolvedModel(self.create_policy(gains, xbar, ubar))

    def loglikelihood(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        model = self.apply_solver(x, params)
        mu, Sigma = vmap(lambda xi: self.moments(xi, model, params))(x)
        return _sum_logpdf(x, mu, Sigma)


def _sum_logpdf(x, mu, Sigma):
    return jnp.sum(vmap(vmap(lambda o, m, S: mvn_logpdf(o, m, S, 1e-6, relative=False)))(x[:, 1:], mu, Sigma))


class InverseGILQR(InverseILQR):
    def __init__(self, env: Env, maxent_temp: float = 0., max_iter: int = 10):
        super().__init__(env, solve=gilqr.solve, maxent_temp=maxent_temp, max_iter=max_iter)


class FixedLinearizationInverseILQR(InverseILQR):
    """Data-based linearization (paper section 3.3): policy linearized along each trajectory with Gauss-Newton
    controls."""

    def __init__(self, env: Env, maxent_temp: float = 0., max_iter: int = 0):
        super().__init__(env, solve=ilqr_fixed.solve, maxent_temp=maxent_temp, max_iter=max_iter)

    def apply_solver(self, x: jnp.ndarray, params: Any) -> SolvedModel:
        """Policy of one trajectory x (T+1, d), linearized along x."""
        gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params)
        return SolvedModel(self.create_policy(gains, xbar, ubar))

    def loglikelihood(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        mu, Sigma = vmap(lambda xi: self.moments(xi, self.apply_solver(xi, params), params))(x)
        return _sum_logpdf(x, mu, Sigma)


class FixedLinearizationInverseGILQR(FixedLinearizationInverseILQR):
    def __init__(self, env: Env, maxent_temp: float = 0., max_iter: int = 0):
        super().__init__(env, maxent_temp=maxent_temp, max_iter=max_iter)

    def apply_solver(self, x: jnp.ndarray, params: Any) -> SolvedModel:
        gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params,
                                       lqr=glqr)
        return SolvedModel(self.create_policy(gains, xbar, ubar))
