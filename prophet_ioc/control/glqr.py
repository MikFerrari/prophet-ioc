from jax import random, lax, numpy as jnp
from typing import Tuple

from prophet_ioc.control import lqr, LQGSpec
from prophet_ioc.utils import quadratic_form, bilinear_form


def backward(spec: LQGSpec, eps: float = 1e-4, psd_projection: bool = True) -> lqr.Gains:
    """psd_projection clips negative eigenvalues of the cost-to-go Hessian at every step. Its derivative is NaN
    when eigenvalues are repeated (eigh), so turn it off when differentiating w.r.t. the cost parameters."""
    def loop(carry, step):
        S, s = carry

        Q, q, P, R, r, A, B, V, Cx, Cu = step

        H = R + B.T @ S @ B + quadratic_form(Cu, S).sum(axis=0)
        H = 0.5 * (H + H.T)
        G = P + B.T @ S @ A + bilinear_form(Cu, S, Cx).sum(axis=0)
        g = r + B.T @ s + bilinear_form(Cu, S, V).sum(axis=0)

        # Deal with negative eigenvals of H, see section 5.4.1 of Li's PhD thesis
        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        Sn = Q + A.T @ S @ A + L.T @ Ht @ L + L.T @ G + G.T @ L + quadratic_form(Cx, S).sum(axis=0)
        Sn = 0.5 * (Sn + Sn.T)
        sn = q + A.T @ s + G.T @ l + L.T @ Ht @ l + L.T @ g + bilinear_form(Cx, S, V).sum(axis=0)

        # Ensure cost-to-go Hessian Sn remains PSD (prevents negative eigenvalue accumulation)
        if psd_projection:
            s_evals, s_evecs = jnp.linalg.eigh(Sn)
            Sn = s_evecs @ jnp.diag(jnp.maximum(0., s_evals)) @ s_evecs.T

        return (Sn, sn), (L, l, Ht)

    _, (L, l, H) = lax.scan(loop, (spec.Qf, spec.qf),
                            (spec.Q, spec.q, spec.P, spec.R, spec.r, spec.A, spec.B, spec.V, spec.Cx, spec.Cu),
                            reverse=True)

    return lqr.Gains(L=L, l=l, H=H)


def backward_joint_signal_noise(spec, U: jnp.ndarray, sigma_m, M: jnp.ndarray, eps: float = 1e-4,
                                psd_projection: bool = False) -> lqr.Gains:
    """backward() for the dynamics noise of the joint-space models: state [q; qd] with n joints and n inputs, and a
    state-independent noise of covariance sigma_m^2 u_j^2 M on the (q_j, qd_j) pair of each input j (M the 2 x 2
    zero-order-hold covariance, plus additive noise that does not depend on u). Then Cx = 0 and every input's noise
    column is proportional to one joint, so the generalized terms of backward() reduce to
        sum_i Cu_i^T S Cu_i = diag(d),   sum_i Cu_i^T S V_i = d * u,   all Cx terms = 0,
        d_j = sigma_m^2 (M00 S[j, j] + 2 M01 S[j, n + j] + M11 S[n + j, n + j])
    (O(n) per step instead of O(noise channels x n^3)); same gains as backward() on the full LQG spec, up to
    floating-point rounding. spec: an LQRSpec (Q, q, P, R, r, A, B, Qf, qf); U (T, n) the nominal inputs."""
    n = U.shape[-1]
    idx = jnp.arange(n)

    def loop(carry, step):
        S, s = carry
        Q, q, P, R, r, A, B, u = step
        d = sigma_m ** 2 * (M[0, 0] * S[idx, idx] + 2.0 * M[0, 1] * S[idx, n + idx] + M[1, 1] * S[n + idx, n + idx])
        H = R + B.T @ S @ B + jnp.diag(d)
        H = 0.5 * (H + H.T)
        G = P + B.T @ S @ A
        g = r + B.T @ s + d * u

        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        Sn = Q + A.T @ S @ A + L.T @ Ht @ L + L.T @ G + G.T @ L
        Sn = 0.5 * (Sn + Sn.T)
        sn = q + A.T @ s + G.T @ l + L.T @ Ht @ l + L.T @ g

        if psd_projection:
            s_evals, s_evecs = jnp.linalg.eigh(Sn)
            Sn = s_evecs @ jnp.diag(jnp.maximum(0., s_evals)) @ s_evecs.T

        return (Sn, sn), (L, l, Ht)

    _, (L, l, H) = lax.scan(loop, (spec.Qf, spec.qf), (spec.Q, spec.q, spec.P, spec.R, spec.r, spec.A, spec.B, U),
                            reverse=True)
    return lqr.Gains(L=L, l=l, H=H)


def simulate(key: random.PRNGKey,
             spec: LQGSpec, x0: jnp.ndarray,
             gains: lqr.Gains = None, eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulates noiseless forward dynamics"""

    T = spec.A.shape[0]

    if gains is None:
        gains = backward(spec, eps=eps)

    key1, key2, key3 = random.split(key, 3)
    noise_x = random.normal(key1, (T, spec.V.shape[-1],))
    noise_Cu = random.normal(key2, (T, spec.Cu.shape[-2],))
    noise_Cx = random.normal(key3, (T, spec.Cx.shape[-2],))

    def dyn(x, inps):
        A, B, V, Cx, Cu, gain, eps_x, eps_Cx, eps_Cu = inps
        u = gain.L @ x + gain.l
        nx = A @ x + B @ u + V @ eps_x + Cx @ x @ eps_Cx + Cu @ u @ eps_Cu
        return nx, (nx, u)

    _, (X, U) = lax.scan(dyn, x0, (spec.A, spec.B, spec.V, spec.Cx, spec.Cu, gains, noise_x, noise_Cx, noise_Cu))
    return jnp.vstack([x0, X]), U
