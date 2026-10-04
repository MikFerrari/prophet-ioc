"""gILQR without jaxopt: a fixed number of iterations (differentiable, offline IOC fit) or early stopping (online).

Both variants run the same iteration as prophet_ioc.control.ilqr.solve with the generalized (signal-dependent noise)
backward pass glqr.backward:
    1. quadratize the cost and linearize the dynamics and the noise around the nominal (X, U) (make_lqg_approx);
    2. backward pass: Gains(L feedback, l feedforward correction, H = Q_uu) (glqr.backward, without the eigh PSD
       projection, whose derivative is NaN with repeated eigenvalues);
    3. parallel line search: the rollouts u = U_t + eps l_t + L_t (x - X_t) for every eps in LINE_SEARCH_STEPS are
       evaluated at once (vmap) and the one with the lowest cost is taken; eps = 0 keeps the current trajectory, so
       the cost never increases and no data-dependent loop is needed.
At convergence l ~ 0: the nominal (X, U) is the optimal open-loop trajectory and L the time-varying feedback around it.

Why not lax.while_loop / jaxopt.FixedPointIteration (ilqr.solve) in the IOC fit: reverse-mode differentiation needs a
fixed number of iterations (lax.while_loop has no reverse-mode derivative), and jaxopt's implicit differentiation of
the fixed point failed when the environment is a traced, batched pytree closed over by the solver. Unrolling a fixed
number of iterations (lax.fori_loop) differentiates through the solver exactly as it is run. The online prediction
only needs the forward solve and can stop early (`solve` with tol).

Regularization: Ht = H + max(0, eps - lambda_min(H)) I (Li's thesis, 5.4.1) with eps = env.reg_eps (the scale of
the environment's cost), DEFAULT_EPS = 1e-4 (glqr.backward's default) if the environment does not set it.
HumanKinematicReaching sets 1e-6: the former 1e-4 divided by 100 like every cost weight (terminal wrist weight 1
instead of 100), so the regularization acts as before.
"""
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, vmap

from prophet_ioc.control import glqr, ilqr, make_lqg_approx
from prophet_ioc.control.lqr import Gains
from prophet_ioc.envs import Env

LINE_SEARCH_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.01, 0.0)
DEFAULT_EPS = 1e-4


def zero_gains(env: Env, T: int) -> Gains:
    return Gains(L=jnp.zeros((T,) + env.action_shape + env.state_shape), l=jnp.zeros((T,) + env.action_shape),
                 H=jnp.zeros((T,) + env.action_shape * 2))


def backward(env: Env, X: jnp.ndarray, U: jnp.ndarray, params: Any, eps: Optional[float] = None) -> Gains:
    """Gains of the generalized LQR (signal-dependent noise) around the nominal (X, U); eps default: env.reg_eps or
    DEFAULT_EPS."""
    if eps is None:
        eps = DEFAULT_EPS if env.reg_eps is None else env.reg_eps
    return glqr.backward(make_lqg_approx(env, params)(X, U), eps=eps, psd_projection=False)


def _iteration(env: Env, X: jnp.ndarray, U: jnp.ndarray, params: Any, eps: Optional[float]):
    """One iteration: backward pass around (X, U) and parallel line search. Returns (gains with the chosen step
    applied to l, new X, new U, new cost)."""
    gains = backward(env, X, U, params, eps)

    def candidate(step):
        Xn, Un = ilqr.rollout(env, X, U, Gains(gains.L, step * gains.l, gains.H), params)
        return Xn, Un, env.trajectory_cost(Xn, Un, params)

    steps = jnp.asarray(LINE_SEARCH_STEPS)
    Xs, Us, costs = vmap(candidate)(steps)
    best = jnp.argmin(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
    return Gains(gains.L, steps[best] * gains.l, gains.H), Xs[best], Us[best], costs[best]


def _initial(env: Env, x0: jnp.ndarray, U: jnp.ndarray, params: Any):
    T = U.shape[0]
    X = jnp.vstack([x0, jnp.repeat(x0[None], T, axis=0)])
    X, U = ilqr.rollout(env, X, U, zero_gains(env, T), params)   # open-loop rollout of the initial controls
    return X, U


def ilqr_unrolled(env: Env, x0: jnp.ndarray, U: jnp.ndarray, params: Any, iters: int, checkpoint: bool = False,
                  eps: Optional[float] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """gILQR with exactly `iters` iterations (lax.fori_loop), differentiable in reverse mode w.r.t. params and the
    environment's leaves; runs under vmap with traced (batched) environments. Returns the nominal (X, U).

    checkpoint: wrap each iteration in jax.checkpoint (rematerialization): the backward pass of the gradient
    recomputes the iteration's intermediates (Jacobians, Hessians, line-search rollouts) instead of storing them, so
    reverse-mode memory does not grow with iters x line-search candidates, at the price of about one more forward
    evaluation. It only matters for the gradient of the offline fit (ioc.checkpoint), not for a forward solve."""
    X, U = _initial(env, x0, U, params)

    def body(_, XU):
        _, Xn, Un, _ = _iteration(env, XU[0], XU[1], params, eps)
        return Xn, Un

    return lax.fori_loop(0, iters, jax.checkpoint(body) if checkpoint else body, (X, U))


def solve(env: Env, x0: jnp.ndarray, U_init: jnp.ndarray, params: Any, max_iter: int, tol: Optional[float] = None,
          eps: Optional[float] = None) -> Tuple[Gains, jnp.ndarray, jnp.ndarray]:
    """Forward gILQR solve for prediction: at most max_iter iterations; with tol (early stopping), stops as soon as an
    iteration lowers the cost by less than tol relative to it (lax.while_loop: forward mode only). tol=None runs
    exactly max_iter iterations. Returns (gains, X, U) like gilqr.solve: the gains of the last iteration (computed
    around the previous nominal, l scaled by the accepted step) and the nominal (X, U)."""
    T = U_init.shape[0]
    X, U = _initial(env, x0, U_init, params)
    gains = zero_gains(env, T)
    cost = env.trajectory_cost(X, U, params)

    if tol is None:
        def body(_, carry):
            gains, X, U, cost = carry
            return _iteration(env, X, U, params, eps)
        gains, X, U, _ = lax.fori_loop(0, max_iter, body, (gains, X, U, cost))
        return gains, X, U

    def cond(carry):
        it, done = carry[0], carry[1]
        return (it < max_iter) & ~done

    def step(carry):
        it, _, gains, X, U, cost = carry
        gains, Xn, Un, cost_new = _iteration(env, X, U, params, eps)
        done = (cost - cost_new) <= tol * jnp.abs(cost)
        return it + 1, done, gains, Xn, Un, cost_new

    _, _, gains, X, U, _ = lax.while_loop(cond, step, (0, jnp.array(False), gains, X, U, cost))
    return gains, X, U
