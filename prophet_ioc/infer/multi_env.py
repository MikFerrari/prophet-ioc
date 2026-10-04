"""IOC log-likelihood summed over trials (or training windows) that each have their own environment.

Real reaching trials differ in target, body parameters, initial posture and time step, so each needs its own
environment, and the agent's policy is solved per trial / window from that window's own start (not one policy for all
trials as in InverseILQG / InverseILQR). `MultiTrialLikelihood` takes groups of trials whose environments share the
same static configuration (e.g. the reaching hand), stacked into one batched pytree environment (`stack_envs`), and
vmaps (or lax.maps) the per-trial likelihood `trial_loglikelihood` over each group.

trial_loglikelihood covers both observation models of the paper with the same policy:
    observability="full"     paper Appendix E (inv_ilqr.fully_observed_moments)
    observability="partial"  paper Algorithm 1 (inv_ilqg.belief_moments): joint (x, b) dynamics with an EKF belief,
                             filtered form, so that "full" is exactly its sigma_o -> 0 limit
and two linearizations:
    linearization="solve"    the policy of the model's optimal trajectory from the window start, solved by the
                             differentiable unrolled gILQR (prophet_ioc.control.ilqr_unrolled)
    linearization="data"     paper section 3.3 (ablation): linearized along the data with Gauss-Newton controls
The gains are always those of the generalized LQR (signal-dependent motor noise in the backward pass), and the max-ent
policy noise (temperature) enters the covariance through W = df/dxi in both observation models.
"""
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacobian, lax, vmap

from prophet_ioc.belief import Belief, kf
from prophet_ioc.control.ilqr_unrolled import LINE_SEARCH_STEPS, backward, ilqr_unrolled  # noqa: F401 (re-exported)
from prophet_ioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from prophet_ioc.envs import Env
from prophet_ioc.infer.gaussian import mvn_logpdf, regression
from prophet_ioc.infer.inv_ilqg import SolvedModel, belief_filter, belief_moments, create_filtered_joint_dynamics
from prophet_ioc.infer.inv_ilqr import fully_observed_moments


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


class TrialModel(NamedTuple):
    """The agent's solution for one trial / window: nominal (X, U), gains around it and the policy."""
    X: jnp.ndarray
    U: jnp.ndarray
    gains: Any
    model: SolvedModel


def solve_trial(env: Env, x: jnp.ndarray, params: Any, linearization: str = "solve", temperature: float = 1e-6,
                solve_iters: int = 8, checkpoint: bool = False, x0: Optional[jnp.ndarray] = None) -> TrialModel:
    """Nominal, gains and policy of one trial x (T+1, d).

    "solve": nominal (X, U) = ilqr_unrolled from x0 (default x[0], the window start; with a belief, its mean), then
    the generalized-LQR gains around it; "data": X = x, U = Gauss-Newton controls (no gradient through them). Policy:
    u_t = U_t + l_t + L_t (x - X_t) + Gamma_t xi (max-ent, Gamma_t = sqrt(temperature) chol(H_t^-1); deterministic if
    temperature = 0); l ~ 0 at a converged nominal."""
    T = x.shape[0] - 1
    if linearization == "solve":
        X, U = ilqr_unrolled(env, x[0] if x0 is None else x0, jnp.zeros((T,) + env.action_shape), params,
                             solve_iters, checkpoint=checkpoint)
    elif linearization == "data":
        X, U = x, lax.stop_gradient(gauss_newton_controls(x, env, params))
    else:
        raise ValueError(f"linearization must be 'solve' or 'data', got {linearization}")
    gains = backward(env, X, U, params)
    policy = create_maxent_lqr_policy(gains, X, U, temperature) if temperature > 0 else create_lqr_policy(gains, X, U)
    return TrialModel(X, U, gains, SolvedModel(policy))


def observation_covariance(env: Env, x: jnp.ndarray, params: Any) -> jnp.ndarray:
    """W W^T of the observation model at state x (W = dh/dw)."""
    W = jacobian(env._observation, argnums=1)(x, jnp.zeros(env.obs_noise_shape), params)
    return W @ W.T


def initial_belief(env: Env, x0: jnp.ndarray, params: Any) -> Belief:
    """Default belief of the agent at the start of a window: mean x0 and covariance W_0 W_0^T, the posterior after
    the first observation with a flat prior (-> 0 with sigma_o: nested models)."""
    return Belief(x0, observation_covariance(env, x0, params))


def filter_gains(env: Env, X: jnp.ndarray, U: jnp.ndarray, params: Any, Sigma0: jnp.ndarray,
                 return_cov: bool = False):
    """Kalman gains (filtered form, kf.forward_filtered) along the nominal (X, U): dynamics and motor noise at
    t = 0..T-1, observation model at t = 1..T. return_cov: (K, P_T), also the filter covariance at the end.
    Only the Jacobians the filter needs (A = df/dx, V = df/dv, the values of make_lqg_approx): the cost
    quadratization of make_lqg_approx (FK Hessians) is not needed here and dominated the run-time prediction."""
    v0 = jnp.zeros(env.state_noise_shape)
    w0 = jnp.zeros(env.obs_noise_shape)
    A, V = vmap(lambda x, u: jacobian(env._dynamics, argnums=(0, 2))(x, u, v0, params))(X[:-1], U)
    F = vmap(lambda s: jacobian(env._observation, argnums=0)(s, w0, params))(X[1:])
    W = vmap(lambda s: jacobian(env._observation, argnums=1)(s, w0, params))(X[1:])
    return kf.forward_filtered(A, V, F, W, Sigma0, return_cov=return_cov)


def prefix_belief(env: Env, x: jnp.ndarray, X: jnp.ndarray, U: jnp.ndarray, policy: Callable, params: Any,
                  P_x: jnp.ndarray, jitter: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Joint Gaussian of the state and the agent's belief at the end of an observed prefix x (n+1, d), n >= 1, for
    the partially observed run-time prediction (human_prediction).

    The agent planned from the start of the prefix: nominal (X, U) (at least n steps) and `policy` (on its belief),
    initial belief initial_belief(x[0]) as in trial_loglikelihood. Algorithm 1 is run over the prefix exactly as in
    the likelihood (filter gains along the nominal, belief_filter: the belief is conditioned on every recorded state),
    which gives b_n | x_{0:n} ~ N(mu_b, S_b) and, from the last predictive step, the regression G = S_bx S_xx^-1 of
    b_n on x_n. The prefix states are estimates, and the last one carries the observer's uncertainty P_x
    (x_n ~ N(x[n], P_x), the handover Kalman covariance): with b_n = mu_b + G (x_n - x[n]) + e, e ~ N(0, S_b),

        mean (x[n], mu_b),   covariance [[P_x, P_x G^T], [G P_x, G P_x G^T + S_b]].

    With exact observations (sigma_o -> 0) b = x, G = I and S_b = 0: both blocks are the same random vector, which
    makes the partially observed prediction tend to the fully observed one (the policy acts on the true x_n).
    Returns (mean (2d,), covariance (2d, 2d), P_agent (d, d): the agent's own filter covariance at x_n, from which
    its filter continues on the next plan)."""
    d, n = x.shape[-1], x.shape[0] - 1
    b0 = initial_belief(env, x[0], params)
    K, P_agent = filter_gains(env, X[:n + 1], U[:n], params, b0.Sigma, return_cov=True)
    model = SolvedModel(policy, create_filtered_joint_dynamics(env, K))
    b_n, (_, Sigma) = belief_filter(env, x, model, params, b0, jitter=jitter, relative=True)
    G = regression(Sigma[-1][d:, :d], Sigma[-1][:d, :d], jitter=jitter, relative=True)
    S_bx = G @ P_x
    S_bb = S_bx @ G.T + b_n.Sigma
    Sigma0 = jnp.block([[P_x, S_bx.T], [S_bx, 0.5 * (S_bb + S_bb.T)]])
    return jnp.concatenate([x[-1], b_n.xhat]), Sigma0, P_agent


def trial_loglikelihood(env: Env, x: jnp.ndarray, params: Any, velocity_block: Optional[slice] = None,
                        jitter: float = 1e-6, linearization: str = "solve", solve_iters: int = 8,
                        mask: Optional[jnp.ndarray] = None, observability: str = "full", temperature: float = 1e-6,
                        b0: Optional[Belief] = None, checkpoint: bool = False) -> jnp.ndarray:
    """Log-likelihood of the transitions of one trajectory / training window x (T+1, state).

    The agent's policy is solved for this window (solve_trial); then
        observability="full":    p(x_{t+1} | x_t) = N(f(x_t, pi_t(x_t)), V V^T + W W^T) for every t (vmap)
        observability="partial": Algorithm 1 with the filtered-form EKF belief (inv_ilqg.belief_moments), gains
                                 K along the nominal, initial belief b0; the x block of each predictive Gaussian
    plus, in both, the likelihood-only residual covariance env.residual_covariance(params) (params.residual_noise:
    model mismatch explained without changing the controller; on CARI, fitting the motor noise to the residuals
    pushed it to its upper bound and made the nominal stop 2-3 cm short of the target).

    b0 (partial only): Belief(mean, covariance) of the agent's belief at t = 0; default mean x[0] and covariance
        W_0 W_0^T, the posterior after the first observation with a flat prior (-> 0 with sigma_o: nested models).
        The policy is solved from the belief mean.
    mask (T+1,): weights of the states; transition t -> t+1 counts with weight mask[t+1].
    velocity_block: if given, only those state components are scored (fallback). It was needed with the former Euler
        integrator, whose position update q_{t+1} = q_t + dt qd_t had no noise: the position block of the predictive
        covariance was exactly zero and theta-independent, so it contributed a huge constant (data positions never
        satisfy the relation exactly) that broke the relative stopping tolerance and model comparison, with zero
        gradient. With the exact ZOH noise the position block is non-singular and the full state is scored (default).
    jitter: relative regularization of the predictive covariances (prophet_ioc.infer.gaussian).
    checkpoint: jax.checkpoint of the solver iterations (reverse-mode memory, offline fit only).
    """
    d = x.shape[-1]
    if observability == "full":
        trial = solve_trial(env, x, params, linearization, temperature, solve_iters, checkpoint)
        mu, Sigma = fully_observed_moments(env, x, trial.model.policy, params)
    elif observability == "partial":
        if b0 is None:
            b0 = initial_belief(env, x[0], params)
        trial = solve_trial(env, x, params, linearization, temperature, solve_iters, checkpoint, x0=b0.xhat)
        K = filter_gains(env, trial.X, trial.U, params, b0.Sigma)
        model = SolvedModel(trial.model.policy, create_filtered_joint_dynamics(env, K))
        mu, Sigma = belief_moments(env, x, model, params, b0, jitter=jitter, relative=True)
        mu, Sigma = mu[:, :d], Sigma[:, :d, :d]       # Algorithm 1 line 6: marginal x block
    else:
        raise ValueError(f"observability must be 'full' or 'partial', got {observability}")
    Sigma = Sigma + env.residual_covariance(params)
    obs = x[1:]
    if velocity_block is not None:
        mu, Sigma, obs = mu[:, velocity_block], Sigma[:, velocity_block, velocity_block], obs[:, velocity_block]
    logp = vmap(lambda o, m, S: mvn_logpdf(o, m, S, jitter))(obs, mu, Sigma)
    return jnp.sum(logp if mask is None else logp * mask[1:])


class MultiTrialLikelihood:
    """Sum of per-trial log-likelihoods (trial_loglikelihood) over groups of stacked environments.

    Args:
        groups: sequence of (stacked_env, xs) or (stacked_env, xs, masks), with xs of shape (n_trials, T+1, state) and
            masks (n_trials, T+1) matching the env batch axis (see the mask of trial_loglikelihood).
        fixed_params: values of the parameters that are not inferred (compute_mle builds the params NamedTuple from
            the inferred fields only, the others would otherwise take the class defaults).
        infer: names of the inferred parameters.
        velocity_block, linearization, solve_iters, observability, temperature, checkpoint, jitter: see
            trial_loglikelihood.
        batch_size: if set, the trials of a group are evaluated with lax.map in batches of this size instead of
            a single vmap (lower peak memory, e.g. on the GPU).

    Under jax.jit, pass the groups as an argument (loglikelihood(None, params, groups)) rather than closing over
    self.groups: with the batched environments and trajectories embedded as constants of the compiled program, the
    gILQR iterations returned NaN for every trial but the first of a group on the CPU (XLA 0.11), while the same
    program with the data as arguments is finite.
    """

    def __init__(self, groups: Sequence[Tuple[Env, jnp.ndarray]], fixed_params: NamedTuple, infer: Sequence[str],
                 velocity_block: Optional[slice] = None, linearization: str = "solve", solve_iters: int = 8,
                 batch_size: Optional[int] = None, observability: str = "full", temperature: float = 1e-6,
                 checkpoint: bool = False, jitter: float = 1e-6):
        self.groups = [(g[0], jnp.asarray(g[1]), jnp.ones(np.shape(g[1])[:2]) if len(g) < 3 else jnp.asarray(g[2]))
                       for g in groups]
        self.env = self.groups[0][0]  # compute_mle only uses it for get_params_type()
        self.fixed_params = fixed_params
        self.infer = tuple(infer)
        self.velocity_block = velocity_block
        self.linearization = linearization
        self.solve_iters = solve_iters
        self.batch_size = batch_size
        self.observability = observability
        self.temperature = temperature
        self.checkpoint = checkpoint
        self.jitter = jitter

    def full_params(self, params: NamedTuple) -> NamedTuple:
        return self.fixed_params._replace(**{name: getattr(params, name) for name in self.infer})

    def loglikelihood(self, xs: Any, params: NamedTuple, groups: Optional[Sequence] = None) -> jnp.ndarray:
        """xs is ignored (the trajectories are stored per group); kept for the compute_mle interface. groups: the
        groups to evaluate (default self.groups; pass them explicitly under jit, see the class docstring)."""
        params = self.full_params(params)
        return self._sum(lambda env, x, m: trial_loglikelihood(
            env, x, params, self.velocity_block, jitter=self.jitter, linearization=self.linearization,
            solve_iters=self.solve_iters, mask=m, observability=self.observability, temperature=self.temperature,
            checkpoint=self.checkpoint), groups)

    def _sum(self, per_trial: Callable, groups: Optional[Sequence] = None) -> jnp.ndarray:
        """Sum of per_trial(env, x, mask) over all trials (vmap, or lax.map in batches of batch_size)."""
        groups = self.groups if groups is None else groups
        if not groups:
            return jnp.array(0.0)

        f = lambda exm: per_trial(*exm)
        if self.checkpoint:
            # Without this, the backward pass of lax.map stores the residuals of every trial of the group (e.g. the
            # noise-derivative tensors of the gILQR linearization: 162 windows x 30 steps x 38 x 76 x 38 floats
            # = 2.1 GB), whatever batch_size is; checkpointed, it stores only each trial's inputs and recomputes.
            f = jax.checkpoint(f)
        if self.batch_size is None:
            return jnp.sum(jnp.stack([vmap(f)(tuple(g)).sum() for g in groups]))
        return jnp.sum(jnp.stack([lax.map(f, tuple(g), batch_size=self.batch_size).sum() for g in groups]))


MultiTrialInverseGILQR = MultiTrialLikelihood   # former name (fully observed gILQR likelihood)


def trial_open_loop_error(env: Env, x: jnp.ndarray, params: Any, output_fn: Callable, solve_iters: int = 8,
                          mask: Optional[jnp.ndarray] = None, checkpoint: bool = False) -> jnp.ndarray:
    """Mean over time of the squared distance, in the space of output_fn(env, state), between the model's open-loop
    optimal trajectory from x[0] over the whole horizon T and the observed trajectory x (T+1, state); with mask
    (T+1,), a weighted mean (e.g. only the observed prefix)."""
    X, _ = ilqr_unrolled(env, x[0], jnp.zeros((x.shape[0] - 1,) + env.action_shape), params, solve_iters,
                         checkpoint=checkpoint)
    out = vmap(lambda s: output_fn(env, s))
    err = jnp.sum((out(X) - out(x)) ** 2, axis=-1)
    return jnp.mean(err) if mask is None else jnp.sum(err * mask) / jnp.maximum(jnp.sum(mask), 1.0)


class MultiTrialTrajectoryMatching(MultiTrialLikelihood):
    """Cost-weight fit on the open-loop prediction error instead of the one-step likelihood (ablation,
    ioc.objective = open_loop).

    loglikelihood(xs, params) = -scale * sum over trials of trial_open_loop_error, i.e. the log-likelihood (up to a
    constant) of the observed outputs under the model's open-loop prediction with a fixed isotropic Gaussian error.
    This is the quantity a predictor that rolls the model out from the last observed state is evaluated on; the
    one-step likelihood conditions every transition on the observed state instead. Same interface, so compute_mle
    can be used unchanged.
    """

    def __init__(self, groups: Sequence[Tuple[Env, jnp.ndarray]], fixed_params: NamedTuple, infer: Sequence[str],
                 output_fn: Callable, scale: float = 1e4, solve_iters: int = 8, batch_size: Optional[int] = None,
                 checkpoint: bool = False):
        super().__init__(groups, fixed_params, infer, solve_iters=solve_iters, batch_size=batch_size,
                         checkpoint=checkpoint)
        self.output_fn = output_fn
        self.scale = scale

    def loglikelihood(self, xs: Any, params: NamedTuple, groups: Optional[Sequence] = None) -> jnp.ndarray:
        params = self.full_params(params)
        return -self.scale * self._sum(lambda env, x, m: trial_open_loop_error(env, x, params, self.output_fn,
                                                                               self.solve_iters, mask=m,
                                                                               checkpoint=self.checkpoint), groups)


def simulate_trial(env: Env, params: Any, key: jax.Array, T: int, temperature: float = 1e-6, solve_iters: int = 8,
                   x0: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """A trajectory (T+1, d) of the fully observed model: the agent solves its policy from x0 (default env.x0) as
    solve_trial does, then acts with it (max-ent policy noise xi) through the noisy dynamics (motor noise v). Synthetic
    data for parameter-recovery checks."""
    x0 = env.x0 if x0 is None else x0
    trial = solve_trial(env, jnp.repeat(x0[None], T + 1, axis=0), params, "solve", temperature, solve_iters)
    k_xi, k_v = jax.random.split(key)
    xis = jax.random.normal(k_xi, (T,) + env.action_shape)
    vs = jax.random.normal(k_v, (T,) + env.state_noise_shape)

    def step(x, inp):
        t, xi, v = inp
        x_next = env._dynamics(x, trial.model.policy(t, x, xi), v, params)
        return x_next, x_next

    _, X = lax.scan(step, x0, (jnp.arange(T), xis, vs))
    return jnp.vstack([x0, X])
