"""gILQR IOC log-likelihood summed over trials that each have their own environment.

`compute_mle` (prophet_ioc.infer.utils) expects one `ioc` object with an `env` and a `loglikelihood(xs, params)`. Real
reaching trials differ in target, body parameters, initial posture and time step, so each needs its own environment.
`MultiTrialInverseGILQR` takes groups of trials whose environments share the same static configuration (e.g. the
reaching hand), stacked into one batched pytree environment (`stack_envs`), and vmaps the per-trial likelihood of
`trial_loglikelihood` over each group.
"""
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy as js
from jax import jacobian, lax, vmap

from prophet_ioc.control import glqr, ilqr, make_lqg_approx
from prophet_ioc.control.lqr import Gains
from prophet_ioc.control.policy import create_lqr_policy
from prophet_ioc.envs import Env
from prophet_ioc.infer.inv_ilqr import FixedLinearizationInverseGILQR


def gauss_newton_controls(x: jnp.ndarray, env: Env, params: Any, iters: int = 2) -> jnp.ndarray:
    """u_t = argmin_u |x_{t+1} - f(x_t, u, 0)|^2 by unrolled Gauss-Newton steps (exact in one step for dynamics
    linear in u). Same estimate as prophet_ioc.infer.utils.estimate_controls, but without jaxopt's implicit
    differentiation, which fails when the environment is a traced (batched) pytree closed over by the solver."""
    zero_noise = jnp.zeros(env.state_noise_shape)

    def estimate(x0, x1):
        residual = lambda u: env._dynamics(x0, u, zero_noise, params) - x1
        u = jnp.zeros(env.action_shape)
        for _ in range(iters):
            u = u - jnp.linalg.lstsq(jacobian(residual)(u), residual(u))[0]
        return u

    return vmap(estimate)(x[:-1], x[1:])


LINE_SEARCH_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.01, 0.0)


def ilqr_unrolled(env: Env, x0: jnp.ndarray, U: jnp.ndarray, params: Any, iters: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """gILQR with a fixed number of iterations and a parallel line search over LINE_SEARCH_STEPS (the step with the
    lowest cost is taken; step 0 keeps the current trajectory, so the cost never increases). Same iteration as
    prophet_ioc.control.ilqr.solve, but free of jaxopt and lax.while_loop, so it runs under vmap with traced
    environments. Returns the nominal (X, U). Each iteration is rematerialized in the backward pass
    (jax.checkpoint), so reverse-mode memory does not grow with iters x line-search candidates."""
    T = U.shape[0]
    X = jnp.vstack([x0, jnp.repeat(x0[None], T, axis=0)])
    zero_gains = Gains(L=jnp.zeros((T,) + env.action_shape + env.state_shape), l=jnp.zeros((T,) + env.action_shape),
                       H=jnp.zeros((T,) + env.action_shape * 2))
    X, U = ilqr.rollout(env, X, U, zero_gains, params)

    @jax.checkpoint
    def body(_, XU):
        X, U = XU
        gains = glqr.backward(make_lqg_approx(env, params)(X, U), psd_projection=False)

        def candidate(eps):
            Xn, Un = ilqr.rollout(env, X, U, Gains(gains.L, eps * gains.l, gains.H), params)
            return Xn, Un, env.trajectory_cost(Xn, Un, params)

        Xs, Us, costs = vmap(candidate)(jnp.asarray(LINE_SEARCH_STEPS))
        best = jnp.argmin(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
        return Xs[best], Us[best]

    return lax.fori_loop(0, iters, body, (X, U))


def solved_policy(env: Env, x0: jnp.ndarray, T: int, params: Any, iters: int = 8):
    """
    Closed-loop policy of the model's own optimal trajectory from x0, differentiable w.r.t. params.
    """
    X, U = ilqr_unrolled(env, x0, jnp.zeros((T,) + env.action_shape), params, iters)
    gains = glqr.backward(make_lqg_approx(env, params)(X, U), psd_projection=False)
    X1, U1 = ilqr.rollout(env, X, U, gains, params)
    return create_lqr_policy(gains, X1, U1)


def trial_loglikelihood(env: Env, x: jnp.ndarray, params: Any, velocity_block: Optional[slice] = None,
                        jitter: float = 1e-6, linearization: str = "solve", solve_iters: int = 8,
                        mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Log-likelihood of the transitions of one trajectory x (T+1, state) under a linearized optimal policy.

    mask (T+1,): weights of the states; transition t -> t+1 counts with weight mask[t+1] (e.g. 1 on an observed
    prefix and 0 afterwards, where x is only a placeholder: the policy is still solved over the whole horizon T).

    linearization="solve": policy of the model's optimal trajectory from x[0] (solved_policy), evaluated at the
        observed states (InverseGILQR-like). The predicted means must follow the data through the optimal nominal,
        which identifies the cost weights.
    linearization="data": policy linearized around x itself, with the controls estimated by gauss_newton_controls
        (FixedLinearizationInverseGILQR). The mean prediction is then x_{t+1} + dt * l_t, and every trajectory that
        ends at the target is a stationary point (l_t -> 0) as the running costs vanish: on reaching data the fit
        drifts to the lower bounds of the cost weights and noises (see the synthetic recovery check).

    A `residual_noise` field of params adds a likelihood-only variance: model mismatch is then explained by it
    rather than by the motor noise, which also shapes the controller (signal-dependent noise makes the optimal plan
    conservative: on CARI, fitting the motor noise to the one-step residuals pushed it to its upper bound and made the
    nominal stop 2-3 cm short of the target).

    If velocity_block is given, only those state components enter the likelihood. Use it when the remaining
    components follow deterministically from the others (q_{t+1} = q_t + dt qd_t) but the data do not satisfy that
    relation exactly, e.g. with Savitzky-Golay velocities: their near-zero predicted variance would otherwise turn
    the integration mismatch into a huge, parameter-independent term that swamps the float32 objective.
    """
    ioc = FixedLinearizationInverseGILQR(env)
    if linearization == "solve":
        policy = solved_policy(env, x[0], x.shape[0] - 1, params, solve_iters)
    elif linearization == "data":
        U = lax.stop_gradient(gauss_newton_controls(x, env, params))
        # as ilqr_fixed.solve, without glqr's eigh PSD projection (NaN gradients with repeated eigenvalues)
        gains = glqr.backward(make_lqg_approx(env, params)(x, U), psd_projection=False)
        policy = create_lqr_policy(gains, x, U)
    else:
        raise ValueError(f"linearization must be 'solve' or 'data', got {linearization}")
    mu, Sigma = ioc.moments(x, policy, params)
    obs = x[1:]
    if velocity_block is not None:
        mu, Sigma, obs = mu[:, velocity_block], Sigma[:, velocity_block, velocity_block], obs[:, velocity_block]
    # params.residual_noise (if present): likelihood-only additive noise, invisible to the controller
    Sigma = Sigma + (jitter + getattr(params, "residual_noise", 0.0) ** 2) * jnp.eye(Sigma.shape[-1])
    logp = js.stats.multivariate_normal.logpdf(obs, mu, Sigma)
    return jnp.sum(logp if mask is None else logp * mask[1:])


class MultiTrialInverseGILQR:
    """Sum of per-trial gILQR log-likelihoods (trial_loglikelihood) over groups of stacked environments.

    Args:
        groups: sequence of (stacked_env, xs) or (stacked_env, xs, masks), with xs of shape (n_trials, T+1, state) and
            masks (n_trials, T+1) matching the env batch axis (see the mask of trial_loglikelihood).
        fixed_params: values of the parameters that are not inferred (compute_mle builds the params NamedTuple from
            the inferred fields only, the others would otherwise take the class defaults).
        infer: names of the inferred parameters.
        velocity_block, linearization, solve_iters: see trial_loglikelihood.
        batch_size: if set, the trials of a group are evaluated with lax.map in batches of this size instead of
            a single vmap (lower peak memory, e.g. on the GPU).
    """

    def __init__(self, groups: Sequence[Tuple[Env, jnp.ndarray]], fixed_params: NamedTuple, infer: Sequence[str],
                 velocity_block: Optional[slice] = None, linearization: str = "solve", solve_iters: int = 8,
                 batch_size: Optional[int] = None):
        self.groups = [(g[0], jnp.asarray(g[1]), jnp.ones(np.shape(g[1])[:2]) if len(g) < 3 else jnp.asarray(g[2]))
                       for g in groups]
        self.env = self.groups[0][0]  # compute_mle only uses it for get_params_type()
        self.fixed_params = fixed_params
        self.infer = tuple(infer)
        self.velocity_block = velocity_block
        self.linearization = linearization
        self.solve_iters = solve_iters
        self.batch_size = batch_size

    def full_params(self, params: NamedTuple) -> NamedTuple:
        return self.fixed_params._replace(**{name: getattr(params, name) for name in self.infer})

    def loglikelihood(self, xs: Any, params: NamedTuple) -> jnp.ndarray:
        """xs is ignored (the trajectories are stored per group); kept for the compute_mle interface."""
        params = self.full_params(params)
        return self._sum(lambda env, x, m: trial_loglikelihood(env, x, params, self.velocity_block,
                                                               linearization=self.linearization,
                                                               solve_iters=self.solve_iters, mask=m))

    def _sum(self, per_trial: Callable) -> jnp.ndarray:
        """Sum of per_trial(env, x, mask) over all trials (vmap, or lax.map in batches of batch_size)."""
        if not self.groups:
            return jnp.array(0.0)

        f = lambda exm: per_trial(*exm)
        if self.batch_size is None:
            return jnp.sum(jnp.stack([vmap(f)(g).sum() for g in self.groups]))
        return jnp.sum(jnp.stack([lax.map(f, g, batch_size=self.batch_size).sum() for g in self.groups]))


def trial_open_loop_error(env: Env, x: jnp.ndarray, params: Any, output_fn: Callable, solve_iters: int = 8,
                          mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Mean over time of the squared distance, in the space of output_fn(env, state), between the model's open-loop
    optimal trajectory from x[0] over the whole horizon T and the observed trajectory x (T+1, state); with mask
    (T+1,), a weighted mean (e.g. only the observed prefix)."""
    X, _ = ilqr_unrolled(env, x[0], jnp.zeros((x.shape[0] - 1,) + env.action_shape), params, solve_iters)
    out = vmap(lambda s: output_fn(env, s))
    err = jnp.sum((out(X) - out(x)) ** 2, axis=-1)
    return jnp.mean(err) if mask is None else jnp.sum(err * mask) / jnp.maximum(jnp.sum(mask), 1.0)


class MultiTrialTrajectoryMatching(MultiTrialInverseGILQR):
    """Cost-weight fit on the open-loop prediction error instead of the one-step likelihood.

    loglikelihood(xs, params) = -scale * sum over trials of trial_open_loop_error, i.e. the log-likelihood (up to a
    constant) of the observed outputs under the model's open-loop prediction with a fixed isotropic Gaussian error.
    This is the quantity a predictor that rolls the model out from the last observed state is evaluated on; the
    one-step likelihood (MultiTrialInverseGILQR) conditions every transition on the observed state instead, and on
    CARI selected weights that predicted worse open loop. Same interface, so compute_mle can be used unchanged.
    """

    def __init__(self, groups: Sequence[Tuple[Env, jnp.ndarray]], fixed_params: NamedTuple, infer: Sequence[str],
                 output_fn: Callable, scale: float = 1e4, solve_iters: int = 8, batch_size: Optional[int] = None):
        super().__init__(groups, fixed_params, infer, solve_iters=solve_iters, batch_size=batch_size)
        self.output_fn = output_fn
        self.scale = scale

    def loglikelihood(self, xs: Any, params: NamedTuple) -> jnp.ndarray:
        params = self.full_params(params)
        return -self.scale * self._sum(lambda env, x, m: trial_open_loop_error(env, x, params, self.output_fn,
                                                                               self.solve_iters, mask=m))
