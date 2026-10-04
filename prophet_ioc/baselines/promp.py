r"""Probabilistic Movement Primitives (ProMP) baseline.

Paraschos, Daniel, Peters, Neumann, "Probabilistic Movement Primitives", NeurIPS 2013; used for human motion
prediction in human-robot collaboration by Maeda, Neumann, Ewerton, Lioutikov, Kroemer, Peters, "Probabilistic
movement primitives for coordination of multiple human-robot collaborative tasks", Autonomous Robots 2017.

Model. Every reach is mapped to a phase z in [0, 1] (onset -> end of the reach) and each of the D = 27 canonical
dimensions (`common`: 9 upper-body keypoints, displacements from the onset in a body-centred frame, left-hand reaches
mirrored) is a weighted sum of K normalized Gaussian basis functions:

    y_t = Psi(z_t)^T w + eps,   Psi(z) = I_D (x) phi(z),   eps ~ N(0, sigma_y^2 I),   w ~ N(mu_w, Sigma_w).

The weights of each training reach are fitted by ridge regression; mu_w and Sigma_w (D K x D K: couples all the
joints and all the phases) are their sample mean and covariance, with a ridge `cov_reg` x mean variance on the
diagonal (81 training reaches for 405 weights; cov_reg = 0.03 was selected on held-out TRAINING subjects, lower
values gave over-confident, less accurate predictions), and sigma_y is the RMS residual of the per-reach fits.

Choices (see README):
- Cartesian space, the 9 keypoints of the evaluation: the space where every other baseline predicts and is scored,
  and where the reaching-wrist goal is a linear (Gaussian) observation of w. In joint space the goal would be a
  non-linear (forward-kinematics) constraint.
- One distribution pooled over all the instructions and both hands, with the reaching-wrist goal as a via-point at
  z = 1 (Gaussian conditioning). Per instruction there would be 9 demonstrations (one per training subject), and the
  goal location of an instruction differs between the subjects' cells anyway; the pooled covariance learns how the
  whole upper body co-varies with the wrist goal.
- Phase of the observed prefix: z = t / T with t the time since the onset and T the arrival time given to the
  goal-directed baselines (minjerk, gcv: on CARI v2 the end of the reach). Optionally (`phase="ml"`) T is
  estimated by maximizing the marginal likelihood of the observed prefix over a grid of durations (phase
  estimation as in Maeda et al. 2017); predictions after the estimated arrival hold the final pose.

Prediction: the weight distribution is conditioned (Kalman update) on `n_cond` frames of the observed prefix
(evenly spaced, always including the last one, noise sigma_y; consecutive frames carry correlated residuals and
would make the posterior over-confident) and on the reaching-wrist goal at z = 1 (noise `sigma_goal`, default
sigma_y), then the mean and the covariance Psi^T Sigma_w' Psi + sigma_y^2 I are evaluated on the prediction grid
and mapped back to the world frame (per-joint 3 x 3 position covariances, for the 95 % coverage).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from prophet_ioc.baselines.common import (
    N_DIMS, WRIST_DIMS, BaselinePrediction, Reach, canonical_frame, check_reaches,
)


def gaussian_basis(z: np.ndarray, n_basis: int, width: float = 1.0) -> np.ndarray:
    """Normalized Gaussian basis (len(z), n_basis): centres evenly spaced on [0, 1], standard deviation `width`
    times the spacing of the centres."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    c = np.linspace(0.0, 1.0, n_basis)
    h = width / (n_basis - 1)
    phi = np.exp(-0.5 * ((z[:, None] - c[None, :]) / h) ** 2)
    return phi / phi.sum(axis=1, keepdims=True)


@dataclass
class ProMP:
    """Weight distribution N(mu_w, Sigma_w) of a D-dimensional ProMP with K basis functions per dimension.
    Weights are ordered dimension-major: w[d * K + k]."""
    mu_w: np.ndarray       # (D K,)
    Sigma_w: np.ndarray    # (D K, D K)
    sigma_y: float         # observation noise (std, m)
    n_basis: int
    n_dims: int
    width: float = 1.0

    @classmethod
    def fit(cls, demos: Sequence[np.ndarray], n_basis: int = 12, width: float = 1.0, ridge: float = 1e-6,
            cov_reg: float = 3e-2, min_sigma_y: float = 1e-3) -> "ProMP":
        """Learns the weight distribution from demonstrations (each (n_i, D), sampled uniformly in phase from z = 0
        to z = 1)."""
        W, res = [], []
        for Y in demos:
            Y = np.asarray(Y, dtype=float)
            Phi = gaussian_basis(np.linspace(0.0, 1.0, len(Y)), n_basis, width)
            Wd = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(n_basis), Phi.T @ Y)   # (K, D)
            W.append(Wd.T.ravel())
            res.append(Y - Phi @ Wd)
        W = np.stack(W)
        D = W.shape[1] // n_basis
        mu = W.mean(axis=0)
        S = np.cov(W, rowvar=False) if len(W) > 1 else np.zeros((len(mu), len(mu)))
        S = 0.5 * (S + S.T)
        scale = max(float(np.mean(np.diag(S))), 1e-8)
        S = S + cov_reg * scale * np.eye(len(mu))
        sigma_y = max(float(np.sqrt(np.mean(np.concatenate([r.ravel() for r in res]) ** 2))), min_sigma_y)
        return cls(mu, S, sigma_y, n_basis, D, width)

    # --- observation model -------------------------------------------------------------------------------------
    def _rows(self, z: float, dims: np.ndarray) -> np.ndarray:
        """Observation matrix (len(dims), D K) of dimensions `dims` at phase z."""
        phi = gaussian_basis(z, self.n_basis, self.width)[0]
        return np.kron(np.eye(self.n_dims)[dims], phi)

    def condition(self, zs: Sequence[float], ys: Sequence[np.ndarray], dims: Sequence[np.ndarray],
                  sigmas: Sequence[float]) -> "ProMP":
        """Posterior weight distribution given observations ys[i] (len(dims[i]),) of dimensions dims[i] at phase
        zs[i] with noise std sigmas[i] (Kalman update, all observations jointly)."""
        H = np.concatenate([self._rows(z, np.asarray(d)) for z, d in zip(zs, dims)])
        y = np.concatenate([np.asarray(v, dtype=float) for v in ys])
        r = np.concatenate([np.full(len(d), s ** 2) for d, s in zip(dims, sigmas)])
        SH = self.Sigma_w @ H.T
        S = H @ SH + np.diag(r)
        K = np.linalg.solve(S, SH.T).T          # Sigma H^T S^-1 (S symmetric)
        mu = self.mu_w + K @ (y - H @ self.mu_w)
        Sigma = self.Sigma_w - K @ SH.T
        return ProMP(mu, 0.5 * (Sigma + Sigma.T), self.sigma_y, self.n_basis, self.n_dims, self.width)

    def log_likelihood(self, zs: Sequence[float], ys: Sequence[np.ndarray], dims: Sequence[np.ndarray],
                       sigmas: Sequence[float]) -> float:
        """Marginal log-likelihood of the observations (same arguments as condition)."""
        H = np.concatenate([self._rows(z, np.asarray(d)) for z, d in zip(zs, dims)])
        y = np.concatenate([np.asarray(v, dtype=float) for v in ys])
        r = np.concatenate([np.full(len(d), s ** 2) for d, s in zip(dims, sigmas)])
        S = H @ self.Sigma_w @ H.T + np.diag(r)
        e = y - H @ self.mu_w
        L = np.linalg.cholesky(S)
        a = np.linalg.solve(L, e)
        return float(-0.5 * a @ a - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2 * np.pi))

    def mean(self, z: np.ndarray) -> np.ndarray:
        """Mean trajectory (len(z), D)."""
        Phi = gaussian_basis(z, self.n_basis, self.width)
        return Phi @ self.mu_w.reshape(self.n_dims, self.n_basis).T

    def cov(self, z: np.ndarray, noise: bool = True) -> np.ndarray:
        """Covariance (len(z), D, D) of y at each phase (with the observation noise sigma_y^2 I if `noise`)."""
        Phi = gaussian_basis(z, self.n_basis, self.width)
        S4 = self.Sigma_w.reshape(self.n_dims, self.n_basis, self.n_dims, self.n_basis)
        C = np.einsum("nk,dkel,nl->nde", Phi, S4, Phi, optimize=True)
        if noise:
            C = C + self.sigma_y ** 2 * np.eye(self.n_dims)
        return 0.5 * (C + np.swapaxes(C, 1, 2))


class ProMPBaseline:
    """ProMP predictor of the 9 upper-body joints (see the module docstring).

    Options: n_basis (basis functions per dimension), width (basis std in units of the centre spacing), cov_reg
    (diagonal ridge of Sigma_w, relative to its mean variance), n_cond (observed frames conditioned on), sigma_goal
    (std of the goal via-point, m; None = sigma_y), phase ("given": z = t / T with the given arrival time; "ml":
    maximum-likelihood duration from the observed prefix).
    """

    def __init__(self, promp: ProMP, n_cond: int = 10, sigma_goal: Optional[float] = None, phase: str = "given"):
        if phase not in ("given", "ml"):
            raise ValueError(f"phase must be 'given' or 'ml', got {phase}")
        self.promp, self.n_cond, self.phase = promp, int(n_cond), phase
        self.sigma_goal = promp.sigma_y if sigma_goal is None else float(sigma_goal)

    @classmethod
    def fit(cls, reaches: Sequence[Reach], n_basis: int = 12, width: float = 1.0, cov_reg: float = 3e-2,
            n_cond: int = 10, sigma_goal: Optional[float] = None, phase: str = "given") -> "ProMPBaseline":
        check_reaches(reaches)
        promp = ProMP.fit([r.canonical() for r in reaches], n_basis=n_basis, width=width, cov_reg=cov_reg)
        return cls(promp, n_cond=n_cond, sigma_goal=sigma_goal, phase=phase)

    def _observations(self, Y_obs: np.ndarray, t_obs: np.ndarray, T: float, goal_c: np.ndarray):
        idx = np.unique(np.round(np.linspace(0, len(Y_obs) - 1, min(self.n_cond, len(Y_obs)))).astype(int))
        all_dims = np.arange(N_DIMS)
        zs = [float(t_obs[i] / T) for i in idx] + [1.0]
        ys = [Y_obs[i] for i in idx] + [goal_c]
        dims = [all_dims] * len(idx) + [WRIST_DIMS]
        sigmas = [self.promp.sigma_y] * len(idx) + [self.sigma_goal]
        return zs, ys, dims, sigmas

    def estimate_duration(self, Y_obs: np.ndarray, t_obs: np.ndarray, goal_c: np.ndarray, t_min: float,
                          n_grid: int = 40) -> float:
        """Total duration of the reach maximizing the marginal likelihood of the observed prefix and goal, over
        durations giving the last observation a phase in [0.05, 0.95] (and at least t_min)."""
        t_last = max(float(t_obs[-1]), 1e-3)
        Ts = np.unique(np.maximum(t_last / np.linspace(0.05, 0.95, n_grid), max(t_min, t_last)))
        lls = [self.promp.log_likelihood(*self._observations(Y_obs, t_obs, T, goal_c)) for T in Ts]
        return float(Ts[int(np.argmax(lls))])

    def predict(self, obs: dict, hand: str, target: np.ndarray, fut_times: np.ndarray, dt: float
                ) -> BaselinePrediction:
        """Prediction from the observed prefix.

        obs: {joint: (n_obs, 3)} every frame from the onset of the reach to the last observation; hand: reaching
        hand; target: reaching-wrist goal (3,); fut_times: (H+1,) prediction times since the onset, the first one
        being the last observation and the last one the arrival time; dt: frame period of obs."""
        frame = canonical_frame({j: v[0] for j, v in obs.items()}, hand)
        Y_obs = frame.to_canonical(obs)
        t_obs = np.arange(len(Y_obs)) * dt
        goal_c = frame.point_to_canonical(target, 5)
        T = float(fut_times[-1])
        if self.phase == "ml":
            T = self.estimate_duration(Y_obs, t_obs, goal_c, t_min=float(fut_times[0]) + dt)
        post = self.promp.condition(*self._observations(Y_obs, t_obs, max(T, 1e-6), goal_c))
        z = np.clip(np.asarray(fut_times, dtype=float) / max(T, 1e-6), 0.0, 1.0)
        return BaselinePrediction(frame.from_canonical(post.mean(z)), frame.cov_from_canonical(post.cov(z)))

    def prior(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prior mean (len(z), D) and covariance (len(z), D, D) in the canonical frame."""
        return self.promp.mean(z), self.promp.cov(z)
