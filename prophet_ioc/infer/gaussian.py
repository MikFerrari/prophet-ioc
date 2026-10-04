"""Numerically robust Gaussian log-density and conditioning for the IOC likelihoods (float32 friendly).

The transition covariances of the joint-space model span many orders of magnitude (position variance ~ dt^3 / 3 vs
velocity variance ~ dt per unit noise), so both functions work on the correlation matrix R = D^-1 S D^-1,
D = sqrt(diag(S) + floor), regularized by jitter * I: the regularization is relative to each component's variance
(S + jitter * (diag(S) + floor), about), instead of an absolute jitter * I that is either negligible for the
velocities or dominant for the positions. relative=False uses the absolute S + jitter * I (the paper code).
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

FLOOR = 1e-12


def _scaled(S: jnp.ndarray, jitter: float, relative: bool):
    if relative:
        d = jnp.sqrt(jnp.diagonal(S) + FLOOR)
        R = S / (d[:, None] * d[None, :])
    else:
        d = jnp.ones(S.shape[0], dtype=S.dtype)
        R = S
    R = 0.5 * (R + R.T) + jitter * jnp.eye(S.shape[0], dtype=S.dtype)
    return d, jnp.linalg.cholesky(R)


def mvn_logpdf(x: jnp.ndarray, mu: jnp.ndarray, S: jnp.ndarray, jitter: float = 1e-6,
               relative: bool = True) -> jnp.ndarray:
    """log N(x; mu, S) of one vector (vmap for batches), with the regularization of the module docstring."""
    d, L = _scaled(S, jitter, relative)
    z = solve_triangular(L, (x - mu) / d, lower=True)
    n = x.shape[0]
    return -0.5 * jnp.dot(z, z) - jnp.sum(jnp.log(jnp.diagonal(L))) - jnp.sum(jnp.log(d)) \
        - 0.5 * n * jnp.log(2.0 * jnp.pi)


def _gain(S_bx: jnp.ndarray, S_xx: jnp.ndarray, jitter: float, relative: bool):
    """(A, G, d) with G = S_bx D^-1 and A = S_bx D^-1 R^-1 = S_bx S_xx^-1 D (regularized S_xx, _scaled)."""
    d, L = _scaled(S_xx, jitter, relative)
    G = S_bx / d[None, :]
    return cho_solve((L, True), G.T).T, G, d


def regression(S_bx: jnp.ndarray, S_xx: jnp.ndarray, jitter: float = 1e-6, relative: bool = True) -> jnp.ndarray:
    """S_bx S_xx^-1: the coefficient of the linear regression of b on x in a joint Gaussian [b; x]
    (E[b | x] = mu_b + S_bx S_xx^-1 (x - mu_x)), with the regularization of condition."""
    A, _, d = _gain(S_bx, S_xx, jitter, relative)
    return A / d[None, :]


def condition(mu_b: jnp.ndarray, mu_x: jnp.ndarray, S_bb: jnp.ndarray, S_bx: jnp.ndarray, S_xx: jnp.ndarray,
              x: jnp.ndarray, jitter: float = 1e-6, relative: bool = True):
    """Gaussian conditioning of b on an observed x for a joint Gaussian [b; x]:
        mu_{b|x} = mu_b + S_bx S_xx^-1 (x - mu_x),   S_{b|x} = S_bb - S_bx S_xx^-1 S_xb
    (only the observed block S_xx is inverted, through its regularized Cholesky factor)."""
    A, G, d = _gain(S_bx, S_xx, jitter, relative)   # A = S_bx S_xx^-1 D, G = S_bx D^-1
    mean = mu_b + A @ ((x - mu_x) / d)
    cov = S_bb - A @ G.T
    return mean, 0.5 * (cov + cov.T)


@jax.custom_jvp
def psd_projection(S: jnp.ndarray) -> jnp.ndarray:
    """Nearest positive semi-definite matrix (Frobenius norm) to the symmetric part of S: V max(Lambda, 0) V^T.

    Its derivative is the Daleckii-Krein formula dP = V (F o V^T dS V) V^T, F_ij = (f(l_i) - f(l_j)) / (l_i - l_j),
    f = max(., 0), with F_ij = f'(l_i) = [l_i > 0] for equal eigenvalues. The derivative of eigh divides by the
    eigenvalue gaps instead, so differentiating through it gives NaN whenever eigenvalues repeat (the conditioned
    belief covariances of the kinematic model have exactly repeated eigenvalues in float32)."""
    ev, V = jnp.linalg.eigh(0.5 * (S + S.T))
    return V @ (jnp.maximum(ev, 0.0)[:, None] * V.T)


@psd_projection.defjvp
def _psd_projection_jvp(primals, tangents):
    (S,), (dS,) = primals, tangents
    ev, V = jnp.linalg.eigh(0.5 * (S + S.T))
    f = jnp.maximum(ev, 0.0)
    gap = ev[:, None] - ev[None, :]
    tol = 1e-6 * jnp.max(jnp.abs(ev)) + jnp.finfo(S.dtype).tiny
    close = jnp.abs(gap) <= tol
    pos = (ev > 0).astype(S.dtype)                                    # f' (bool + bool would be a logical or)
    F = jnp.where(close, 0.5 * (pos[:, None] + pos[None, :]),
                  (f[:, None] - f[None, :]) / jnp.where(close, 1.0, gap))
    dP = V @ (F * (V.T @ (0.5 * (dS + dS.T)) @ V)) @ V.T
    return V @ (f[:, None] * V.T), dP


def command_noise_second_moment(noise_jacobian, u: jnp.ndarray, cov_u: jnp.ndarray) -> jnp.ndarray:
    """E[J J^T] of a noise Jacobian J = noise_jacobian(u') that is affine in a Gaussian command u' ~ N(u, cov_u) (the
    covariance J J^T of the noise v ~ N(0, I) it maps, averaged over the commands):

        J(u) J(u)^T + sum_ij cov_u[i, j] C_i C_j^T,    C_i = dJ/du_i

    Signal-dependent motor noise (std sigma_m |u|) has a Jacobian linear in the command, so this is its exact
    covariance, sigma_m^2 E[u u^T], instead of sigma_m^2 ubar ubar^T at the nominal command. Under the closed loop
    the command is uncertain, u = ubar + L (x - xbar) + Gamma xi: cov_u = L Sigma L^T + Gamma Gamma^T (a second-order
    term that the nominal-command evaluation drops; it matters when the feedback corrections are not small compared
    with the nominal command, e.g. joints at rest). Dependences of J on the state other than through u are not
    averaged (there are none for HumanKinematicReaching)."""
    J = noise_jacobian(u)
    C = jax.jacfwd(noise_jacobian)(u)                       # (d, m, n_u)
    return J @ J.T + jnp.einsum("aiu,uv,biv->ab", C, cov_u, C)
