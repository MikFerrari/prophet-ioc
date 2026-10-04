"""Exact zero-order-hold (ZOH) discretization of the damped double integrator of the joint-space human model.

Continuous model, per joint (q position, qd velocity, u commanded acceleration held constant over a step dt):

    qdd = u - b qd + sigma u w(t),        w(t) white noise of unit intensity

Exact discretization over one step (e = exp(-b dt)):

    qd_{t+1} = e qd_t + a1 u_t,                     a1 = (1 - e) / b            -> dt       (b -> 0)
    q_{t+1}  = q_t + a1 qd_t + a2 u_t,              a2 = (dt - a1) / b          -> dt^2 / 2 (b -> 0)

a1 = dt phi1(z), a2 = dt^2 phi2(z) with z = b dt, phi1(z) = (1 - e^-z) / z, phi2(z) = (z - 1 + e^-z) / z^2; both are
evaluated with their Taylor series for small z, so b = 0 (the default) is exact, b -> 0 is continuous and no division
by zero (nor a NaN gradient) occurs.

Noise. Over a step the white-noise acceleration integrates to (b = 0)

    delta qd = sigma u int_0^dt w(s) ds,            delta q = sigma u int_0^dt (dt - s) w(s) ds

so that (Ito isometry) Var(delta qd) = sigma^2 u^2 dt, Var(delta q) = sigma^2 u^2 int_0^dt (dt - s)^2 ds
= sigma^2 u^2 dt^3 / 3 and Cov(delta q, delta qd) = sigma^2 u^2 int_0^dt (dt - s) ds = sigma^2 u^2 dt^2 / 2:

    Cov[(delta q, delta qd)] = sigma^2 u^2 M(dt),   M(dt) = [[dt^3 / 3, dt^2 / 2], [dt^2 / 2, dt]]

The b = 0 matrix M is used for any damping (small-b approximation: the exact one has O(b dt) corrections, below
1 % for b dt < 0.03). Its Cholesky factor is [[sqrt(dt^3 / 3), 0], [sqrt(3 dt) / 2, sqrt(dt) / 2]]. Because the
velocity variance is sigma^2 u^2 dt, sigma is a noise intensity (units of u per sqrt(s) per unit of u): it does not
depend on the time grid.

All functions take an array module `xp` (jax.numpy, default, or numpy), so that the environment (JAX), the handover
Kalman filter, the prediction covariance and the RTS smoother of the training data use the same discretization.
"""

from typing import Any, Tuple

import jax.numpy as jnp

# |b dt| below which phi1, phi2 are evaluated by their Taylor series (terms up to z^6, truncation error < 1e-11
# relative). Above it the closed forms lose at most ~2 eps / z (cancellation in z - 1 + e^-z): < 3e-6 in float32.
_SERIES_Z = 0.1


def zoh_coefficients(dt: Any, damping: Any, xp=jnp) -> Tuple[Any, Any, Any]:
    """(e, a1, a2) of the exact ZOH discretization of qdd = u - b qd (module docstring), smooth in b at b = 0."""
    z = damping * dt
    small = xp.abs(z) < _SERIES_Z
    zs = xp.where(small, 1.0, z)  # safe branch: no division by zero, finite gradients in both branches
    em1 = -xp.expm1(-zs)  # 1 - e^-z without cancellation
    # phi1(z) = sum_k (-z)^k / (k + 1)!, phi2(z) = sum_k (-z)^k / (k + 2)!
    s1 = 1.0 - z / 2.0 + z ** 2 / 6.0 - z ** 3 / 24.0 + z ** 4 / 120.0 - z ** 5 / 720.0 + z ** 6 / 5040.0
    s2 = 0.5 - z / 6.0 + z ** 2 / 24.0 - z ** 3 / 120.0 + z ** 4 / 720.0 - z ** 5 / 5040.0 + z ** 6 / 40320.0
    phi1 = xp.where(small, s1, em1 / zs)
    phi2 = xp.where(small, s2, (zs - em1) / zs ** 2)
    return xp.exp(-z), dt * phi1, dt ** 2 * phi2


def zoh_noise_cov(dt: Any, xp=jnp) -> Any:
    """M(dt) = [[dt^3 / 3, dt^2 / 2], [dt^2 / 2, dt]]: covariance of (delta q, delta qd) of a unit white-noise
    acceleration over one step (module docstring)."""
    return xp.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])


def zoh_noise_chol(dt: Any, xp=jnp) -> Tuple[Any, Any, Any]:
    """(l11, l21, l22): lower Cholesky factor of M(dt), l11 = sqrt(dt^3 / 3), l21 = sqrt(3 dt) / 2, l22 = sqrt(dt) / 2:
    delta q = s l11 n1, delta qd = s (l21 n1 + l22 n2) with n1, n2 ~ N(0, 1) has covariance s^2 M(dt)."""
    sq = xp.sqrt(dt)
    return sq * dt / xp.sqrt(3.0), 0.5 * xp.sqrt(3.0) * sq, 0.5 * sq


def zoh_state_matrices(dt: Any, damping: Any, n: int, xp=jnp) -> Tuple[Any, Any]:
    """A (2n, 2n), B (2n, n) of the deterministic ZOH step [q; qd]_{t+1} = A [q; qd]_t + B u_t for n joints."""
    e, a1, a2 = zoh_coefficients(dt, damping, xp)
    I = xp.eye(n)
    A = xp.block([[I, a1 * I], [xp.zeros((n, n)), e * I]])
    B = xp.concatenate([a2 * I, a1 * I], axis=0)
    return A, B


def zoh_process_cov(dt: Any, n: int, xp=jnp) -> Any:
    """Covariance (2n, 2n) over [q; qd] of a unit-intensity white-noise acceleration on each of n joints: M(dt)
    per joint (block layout of the state)."""
    M = zoh_noise_cov(dt, xp)
    I = xp.eye(n)
    return xp.block([[M[0, 0] * I, M[0, 1] * I], [M[1, 0] * I, M[1, 1] * I]])
