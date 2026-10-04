"""Policies from (i)LQR / (i)LQG gains.

A policy is a function pi(t, x, noise) -> u. The constructors return Python closures (`lambda t, x, noise=...: ...`):
the lambda captures gains, xbar, ubar (and temperature) from the enclosing call, and the default argument `noise=...`
lets the policy be called without noise (deterministic use) or with a standard-normal sample (simulation) or with a
zero vector to differentiate w.r.t. it (the max-ent covariance term W = df/dxi of the likelihood). `@` is the matrix
product (jnp.matmul).

All policies are affine around the nominal trajectory (xbar_t, ubar_t) of the iLQR / iLQG solution:

    u_t = ubar_t + l_t + L_t (x - xbar_t)

with Gains(L feedback, l feedforward correction, H = Q_uu) of the backward pass; l_t ~ 0 at convergence, since the
nominal is then already optimal. The gains depend on t because the horizon is finite (time-varying Riccati solution).
"""
from typing import Callable

import jax.numpy as jnp
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular

from prophet_ioc.control.lqr import Gains


def create_lqr_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray) -> Callable:
    """Deterministic policy u = ubar_t + l_t + L_t (x - xbar_t); the noise argument is ignored, so the max-ent
    covariance term W = df/dxi of a likelihood is zero with this policy."""
    return lambda t, x, noise=None: gains.L[t] @ (x - xbar[t]) + gains.l[t] + ubar[t]


def create_lqg_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray) -> Callable:
    """As create_lqr_policy, on the mean b[0] of a belief b = (mean, covariance)."""
    return lambda t, b, noise=None: gains.L[t] @ (b[0] - xbar[t]) + gains.l[t] + ubar[t]


def maxent_noise_factor(H: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Gamma with Gamma Gamma^T = temperature H^-1: the factor of the max-ent policy noise.

    The maximum-entropy (soft-optimal) linear-Gaussian policy of a quadratic Q-function with Hessian H = Q_uu is
    u ~ N(ubar + l + L (x - xbar), temperature * H^-1): actions are more random along directions that cost little.
    Any A with A A^T = Sigma turns a standard-normal xi into u = mean + A xi ~ N(mean, Sigma); a Cholesky factor is
    the cheapest such A (and differentiable). It is computed from the Cholesky factor of H itself, H = C C^T,
    A = C^-T (A A^T = (C C^T)^-1 = H^-1; A is upper triangular, i.e. the Cholesky factor of H^-1 up to an orthogonal
    transformation, which leaves the distribution unchanged): unlike chol(inv(H)), H^-1 is never formed, which in
    float32 lost positive definiteness (NaN) for ill-conditioned H. `temperature` multiplies the covariance (a
    temperature, not an inverse temperature); temperature -> 0 recovers the deterministic policy.
    """
    C = cholesky(0.5 * (H + H.T))
    return jnp.sqrt(temperature) * solve_triangular(C.T, jnp.eye(H.shape[0], dtype=H.dtype), lower=False)


def create_maxent_lqr_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray, temperature=1e-6) -> Callable:
    """Max-ent policy u = ubar_t + l_t + L_t (x - xbar_t) + Gamma_t xi, Gamma_t = sqrt(temperature) chol(H_t^-1)
    (maxent_noise_factor), xi ~ N(0, I) the `noise` argument (zero by default)."""
    return lambda t, x, noise=jnp.zeros(gains.H.shape[1]): gains.L[t] @ (x - xbar[t]) + gains.l[t] + ubar[t] \
        + maxent_noise_factor(gains.H[t], temperature) @ noise


def create_maxent_lqg_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray, temperature=1e-6) -> Callable:
    """As create_maxent_lqr_policy, on the mean b[0] of a belief b = (mean, covariance)."""
    return lambda t, b, noise=jnp.zeros(gains.H.shape[1]): gains.L[t] @ (b[0] - xbar[t]) + gains.l[t] + ubar[t] \
        + maxent_noise_factor(gains.H[t], temperature) @ noise


def create_zero_policy():
    return lambda t, b, noise: jnp.zeros_like(noise)


def create_random_policy(noise_mean=0., noise_std=1.):
    return lambda t, b, noise: noise_std * noise + noise_mean
