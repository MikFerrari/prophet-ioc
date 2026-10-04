from jax import lax, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from prophet_ioc.control import LQGSpec


def forward(spec: LQGSpec, Sigma0: jnp.ndarray, xhat0=None, gains=None) -> jnp.ndarray:
    def loop(P, step):
        A, F, V, W = step

        G = F @ P @ F.T + W @ W.T
        K = A @ P @ F.T @ jnp.linalg.inv(G)
        P = V @ V.T + (A - K @ F) @ P @ A.T

        return P, K

    _, K = lax.scan(loop, Sigma0,
                    (spec.A, spec.F, spec.V, spec.W))

    return K


def forward_filtered(A: jnp.ndarray, V: jnp.ndarray, F: jnp.ndarray, W: jnp.ndarray, Sigma0: jnp.ndarray,
                     return_cov: bool = False):
    """Kalman gains of the filter in filtered (measurement-update) form, along a nominal trajectory.

    The belief at t+1 uses the observation y_{t+1} = F_{t+1} x_{t+1} + W_{t+1} w:
        P-_{t+1} = A_t P_t A_t^T + V_t V_t^T,   K_{t+1} = P-_{t+1} F^T (F P-_{t+1} F^T + W W^T)^-1,
        P_{t+1}  = (I - K_{t+1} F) P-_{t+1}
    (forward() is the one-step predictor form, which uses y_t for the belief at t+1). With exact observations
    (W -> 0, F = I) K -> I and the belief equals the state: the fully observed model is the limit of this filter.

    Args: A (T, d, d), V (T, d, m) of the dynamics at t = 0..T-1; F (T, p, d), W (T, p, p) of the observation at
    t = 1..T; Sigma0 (d, d) the covariance of the belief at t = 0 (after y_0).
    Returns: K (T, d, p), K[t] the gain of the update at time t + 1; with return_cov, (K, P_T): also the filter's
    covariance after the last update (the agent's own uncertainty at the end of the nominal, e.g. to continue the
    filter on a new plan).
    """
    def loop(P, step):
        A, V, F, W = step
        P_pred = A @ P @ A.T + V @ V.T
        S = F @ P_pred @ F.T + W @ W.T
        # K = P- F^T S^-1 through the Cholesky factor of S (symmetric positive definite): same values as an LU solve,
        # ~50x faster on the CPU inside the scan (run-time prediction)
        K = cho_solve(cho_factor(0.5 * (S + S.T), lower=True), F @ P_pred.T).T
        P = (jnp.eye(P.shape[0]) - K @ F) @ P_pred
        return 0.5 * (P + P.T), K

    P, K = lax.scan(loop, Sigma0, (A, V, F, W))
    return (K, P) if return_cov else K
