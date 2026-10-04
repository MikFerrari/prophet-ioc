r"""Dynamic Movement Primitives (DMP) baseline.

Ijspeert, Nakanishi, Hoffmann, Pastor, Schaal, "Dynamical movement primitives: learning attractor models for motor
behaviors", Neural Computation 25(2), 2013.

Model. One discrete DMP per canonical dimension (`common`: 27 = 9 upper-body keypoints, displacements from the onset
in a body-centred frame, left-hand reaches mirrored), all driven by the same canonical system:

    tau s'  = -alpha_s s                                             (s: 1 at the onset, s_end at the arrival)
    tau v'  = alpha_z (beta_z (g - y) - v) + (g - y0) f(s),   tau y' = v
    f(s)    = s sum_i psi_i(s) w_i / sum_i psi_i(s),   psi_i(s) = exp(-h_i (s - c_i)^2)

with tau the duration of the reach (onset -> arrival), y0 the position at the onset, g the goal, critically damped
(beta_z = alpha_z / 4) and the amplitude scaling (g - y0) of Ijspeert et al. 2013 (sec. 2.1.4).

Learning (training reaches only). For each demonstration and dimension, the target forcing term
f_d = tau^2 y'' - alpha_z (beta_z (g - y) - tau y') is resampled on a common phase grid; the forcing shape of each
dimension is the amplitude-weighted average over the demonstrations, f(z) = sum_d a_d f_d(z) / sum_d a_d^2
(a_d = g_d - y0_d: the least-squares fit of f_d ~ a_d f, so that dimensions that barely move in a reach do not
blow up the 1 / a_d normalization), then fitted with the basis functions by ridge least squares. With a single
demonstration f is the demonstration's own forcing term and the DMP reproduces it.

Goal. The reaching wrist goes to the given target (the only goal the goal-directed baselines get). For the other
dimensions the end displacement is learned from the training reaches: by default (`goal_model="regression"`) an
affine ridge regression of the end displacement of every dimension on the end displacement of the reaching wrist
(in the canonical frame: e.g. the elbow follows the wrist), or (`"mean"`) the mean end displacement of the
training reaches. On CARI v2 these prior goals of the trunk / head / other-arm keypoints are ~10 cm off (subjects
differ), so by default (`infer_goal=True`) they are refined from the observed prefix: the transformation system is
linear in g, and the goal is the Gaussian combination of the prior (variance: regression residuals) with the
least-squares goal of the prefix (noise: the phase-dependent spread of the demonstrations' forcing terms), see
DMPBaseline.goal_from_prefix. Without it the DMP pulls those keypoints towards the prior goals even late in the reach.

Prediction from the handover (`phase="continue"`, default): the DMP of the whole reach (tau = arrival time since
the onset, the arrival time given to the goal-directed baselines; y0 = the observed onset pose) is integrated from
the handover state: phase s(t_obs), position and Savitzky-Golay velocity at the last observation. With
`phase="restart"` the DMP is instead restarted at the handover (y0 = handover position, s = 1, tau = remaining
time, initial velocity kept). Deterministic: no covariance.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter

from prophet_ioc.baselines.common import (
    N_DIMS, WRIST_DIMS, BaselinePrediction, Reach, canonical_frame, check_reaches, sg_velocity,
)


def _derivatives(Y: np.ndarray, dt: float):
    """First and second time derivatives of Y (n, D) at every frame (Savitzky-Golay, window <= 9, order 3; finite
    differences for very short sequences)."""
    n = len(Y)
    w = min(9, n if n % 2 else n - 1)
    if w < 5:
        yd = np.gradient(Y, dt, axis=0)
        return yd, np.gradient(yd, dt, axis=0)
    return (savgol_filter(Y, w, 3, deriv=1, delta=dt, axis=0), savgol_filter(Y, w, 3, deriv=2, delta=dt, axis=0))


@dataclass
class DMP:
    """Discrete DMPs of D dimensions sharing one canonical system."""
    weights: np.ndarray     # (D, K) forcing-term weights
    centers: np.ndarray     # (K,) basis centres in s
    widths: np.ndarray      # (K,) h_i
    alpha_z: float = 25.0
    beta_z: float = 6.25
    alpha_s: float = 4.6    # s(1) = exp(-alpha_s) ~ 0.01
    forcing_var: Optional[np.ndarray] = None   # (n_phase, D) variance of the demonstrations' forcing terms
                                               # around the fit, on a uniform phase grid z in [0, 1]

    @staticmethod
    def make_basis(n_basis: int, alpha_s: float):
        """Centres evenly spaced in phase z (c_i = exp(-alpha_s z_i)), widths from the spacing of the centres."""
        c = np.exp(-alpha_s * np.linspace(0.0, 1.0, n_basis))
        d = np.diff(c)
        h = 1.0 / np.append(d, d[-1]) ** 2
        return c, h

    def features(self, s: np.ndarray) -> np.ndarray:
        """(len(s), K): s psi_i(s) / sum_j psi_j(s)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        psi = np.exp(-self.widths[None] * (s[:, None] - self.centers[None]) ** 2)
        return s[:, None] * psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-300)

    def forcing(self, s: np.ndarray) -> np.ndarray:
        """Forcing shape f(s) per dimension (len(s), D), before the amplitude scaling."""
        return self.features(s) @ self.weights.T

    @staticmethod
    def forcing_target(Y: np.ndarray, dt: float, alpha_z: float, beta_z: float) -> np.ndarray:
        """Target forcing term (n, D) of a demonstration Y (n, D) sampled every dt from the onset to the arrival
        (Savitzky-Golay derivatives, window 9, order 3, as for the observed prefix)."""
        tau = (len(Y) - 1) * dt
        yd, ydd = _derivatives(Y, dt)
        return tau ** 2 * ydd - alpha_z * (beta_z * (Y[-1] - Y) - tau * yd)

    @classmethod
    def fit(cls, demos: Sequence[np.ndarray], dts: Sequence[float], n_basis: int = 30, alpha_z: float = 25.0,
            alpha_s: float = 4.6, ridge: float = 1e-8, n_phase: int = 200) -> "DMP":
        """Forcing terms from demonstrations Y (n_i, D) sampled every dts[i] from the onset to the arrival
        (amplitude-weighted average over the demonstrations, see the module docstring)."""
        beta_z = alpha_z / 4.0
        z = np.linspace(0.0, 1.0, n_phase)
        fs, amps = [], []
        for Y, dt in zip(demos, dts):
            Y = np.asarray(Y, dtype=float)
            f = cls.forcing_target(Y, dt, alpha_z, beta_z)
            zd = np.linspace(0.0, 1.0, len(Y))
            fs.append(np.stack([np.interp(z, zd, f[:, j]) for j in range(f.shape[1])], axis=1))
            amps.append(Y[-1] - Y[0])
        fs, amps = np.stack(fs), np.stack(amps)                                         # (N, n_phase, D), (N, D)
        f_avg = np.einsum("nd,nzd->zd", amps, fs) / np.maximum((amps ** 2).sum(axis=0), 1e-12)[None]
        c, h = cls.make_basis(n_basis, alpha_s)
        dmp = cls(np.zeros((f_avg.shape[1], n_basis)), c, h, alpha_z, beta_z, alpha_s)
        F = dmp.features(np.exp(-alpha_s * z))
        dmp.weights = np.linalg.solve(F.T @ F + ridge * np.eye(n_basis), F.T @ f_avg).T
        # spread of the demonstrations around the scaled average shape (m^2, per phase and dimension, smoothed over
        # +-0.05 in phase): noise of the goal inference
        var = np.mean((fs - amps[:, None, :] * (F @ dmp.weights.T)[None]) ** 2, axis=0)
        win = max(int(0.05 * n_phase), 1)
        kernel = np.ones(2 * win + 1) / (2 * win + 1)
        dmp.forcing_var = np.stack([np.convolve(np.pad(var[:, j], win, mode="edge"), kernel, mode="valid")
                                    for j in range(var.shape[1])], axis=1)
        return dmp

    def rollout(self, y: np.ndarray, v: np.ndarray, goal: np.ndarray, y0: np.ndarray, tau: float, t_start: float,
                times: np.ndarray, dt_int: float = 1e-3) -> np.ndarray:
        """Integrates the DMP from time t_start (since the onset; phase s = exp(-alpha_s t / tau)) with position y and
        velocity v = dy/dt (D,) to the given times (>= t_start); returns (len(times), D). RK4 steps of at most
        dt_int."""
        y, yd = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
        goal, y0 = np.asarray(goal, dtype=float), np.asarray(y0, dtype=float)
        times = np.asarray(times, dtype=float)
        amp = goal - y0
        a, b = self.alpha_z / tau ** 2, self.alpha_z * self.beta_z / tau ** 2

        def acc(y, yd, f):
            return b * (goal - y) - a * tau * yd + f

        out, t = np.empty((len(times), len(y))), float(t_start)
        for k, t_next in enumerate(times):
            n = int(np.ceil(max(t_next - t, 0.0) / dt_int - 1e-9))
            if n:
                h = (t_next - t) / n
                # the forcing term depends on time only: evaluated once at every RK4 node (half steps)
                F = amp * self.forcing(np.exp(-self.alpha_s * (t + 0.5 * h * np.arange(2 * n + 1)) / tau)) / tau ** 2
                for i in range(n):
                    f0, f1, f2 = F[2 * i], F[2 * i + 1], F[2 * i + 2]
                    k1y, k1v = yd, acc(y, yd, f0)
                    k2y, k2v = yd + h / 2 * k1v, acc(y + h / 2 * k1y, yd + h / 2 * k1v, f1)
                    k3y, k3v = yd + h / 2 * k2v, acc(y + h / 2 * k2y, yd + h / 2 * k2v, f1)
                    k4y, k4v = yd + h * k3v, acc(y + h * k3y, yd + h * k3v, f2)
                    y = y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
                    yd = yd + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
                t = float(t_next)
            out[k] = y
        return out




class DMPBaseline:
    """DMP predictor of the 9 upper-body joints (see the module docstring).

    Options: n_basis (basis functions per dimension), alpha_z (spring constant, beta_z = alpha_z / 4), goal_model
    ("regression" | "mean": prior end displacement of the dimensions without a given goal), infer_goal (refine those
    goals from the observed prefix), n_cond (observed frames used by the goal inference), phase ("continue" |
    "restart"), dt_int (integration step, s).
    """

    def __init__(self, dmp: DMP, goal_coef: np.ndarray, goal_var: np.ndarray, infer_goal: bool = True,
                 n_cond: int = 10, phase: str = "continue", dt_int: float = 2e-3):
        if phase not in ("continue", "restart"):
            raise ValueError(f"phase must be 'continue' or 'restart', got {phase}")
        self.dmp, self.goal_coef, self.goal_var = dmp, goal_coef, goal_var
        self.infer_goal, self.n_cond, self.phase, self.dt_int = bool(infer_goal), int(n_cond), phase, float(dt_int)

    @classmethod
    def fit(cls, reaches: Sequence[Reach], n_basis: int = 30, alpha_z: float = 25.0, goal_model: str = "regression",
            infer_goal: bool = True, n_cond: int = 10, phase: str = "continue", dt_int: float = 2e-3,
            goal_ridge: float = 1e-3) -> "DMPBaseline":
        check_reaches(reaches)
        demos = [r.canonical() for r in reaches]
        dmp = DMP.fit(demos, [r.dt for r in reaches], n_basis=n_basis, alpha_z=alpha_z)
        ends = np.stack([Y[-1] for Y in demos])                                # (N, D) end displacements
        if goal_model == "regression":   # affine ridge regression on the reaching-wrist end displacement
            X = np.hstack([ends[:, WRIST_DIMS], np.ones((len(ends), 1))])
            reg = goal_ridge * np.diag([1.0, 1.0, 1.0, 0.0])
            coef = np.linalg.solve(X.T @ X + reg + 1e-12 * np.eye(4), X.T @ ends)      # (4, D)
            resid = ends - X @ coef
        elif goal_model == "mean":
            coef = np.vstack([np.zeros((3, N_DIMS)), ends.mean(axis=0)[None]])
            resid = ends - ends.mean(axis=0)
        else:
            raise ValueError(f"goal_model must be 'regression' or 'mean', got {goal_model}")
        goal_var = np.maximum(np.mean(resid ** 2, axis=0), 1e-6)
        return cls(dmp, coef, goal_var, infer_goal=infer_goal, n_cond=n_cond, phase=phase, dt_int=dt_int)

    def goal_prior(self, wrist_goal_c: np.ndarray) -> np.ndarray:
        """Canonical goal (D,) before seeing the prefix: the reaching wrist at its goal, the other dimensions from
        the goal model."""
        g = np.append(wrist_goal_c, 1.0) @ self.goal_coef
        g[WRIST_DIMS] = wrist_goal_c
        return g

    def goal_from_prefix(self, Y_obs: np.ndarray, dt: float, tau: float, g_prior: np.ndarray) -> np.ndarray:
        """Goal of every dimension but the reaching wrist, refined from the observed prefix (canonical, from the
        onset). The transformation system is linear in g: with the learned forcing shape f(s),
            tau^2 y'' + alpha_z beta_z y + alpha_z tau y' + y0 f(s) = (alpha_z beta_z + f(s)) g + r,
        r ~ N(0, forcing_var(z)) (spread of the training demonstrations around the average shape at phase z),
        evaluated at n_cond frames of the prefix (Savitzky-Golay derivatives); combined with the goal-model prior
        N(g_prior, goal_var) (Gaussian, per dimension)."""
        n = len(Y_obs)
        if n < 5 or self.dmp.forcing_var is None:
            return g_prior
        yd, ydd = _derivatives(Y_obs, dt)
        idx = np.unique(np.round(np.linspace(0, n - 1, min(self.n_cond, n))).astype(int))
        t = idx * dt
        f = self.dmp.forcing(np.exp(-self.dmp.alpha_s * t / tau))                    # (m, D)
        k = self.dmp.alpha_z * self.dmp.beta_z
        c = k + f
        b = tau ** 2 * ydd[idx] + k * Y_obs[idx] + self.dmp.alpha_z * tau * yd[idx] + Y_obs[0] * f
        fv = self.dmp.forcing_var
        zg = np.linspace(0.0, 1.0, len(fv))
        r_var = np.maximum(np.stack([np.interp(np.clip(t / tau, 0, 1), zg, fv[:, j]) for j in range(fv.shape[1])],
                                    axis=1), 1e-12)                                    # (m, D)
        prec = (c ** 2 / r_var).sum(axis=0) + 1.0 / self.goal_var
        g = ((c * b / r_var).sum(axis=0) + g_prior / self.goal_var) / prec
        g[WRIST_DIMS] = g_prior[WRIST_DIMS]
        return g

    def predict(self, obs: dict, hand: str, target: np.ndarray, fut_times: np.ndarray, dt: float,
                goal: Optional[np.ndarray] = None) -> BaselinePrediction:
        """Prediction from the observed prefix (same arguments as ProMPBaseline.predict; `goal` (27,) overrides the
        canonical goal of every dimension)."""
        frame = canonical_frame({j: v[0] for j, v in obs.items()}, hand)
        Y_obs = frame.to_canonical(obs)
        fut_times = np.asarray(fut_times, dtype=float)
        tau = max(float(fut_times[-1]), 1e-6)
        if goal is None:
            goal = self.goal_prior(frame.point_to_canonical(target, 5))
            if self.infer_goal:
                goal = self.goal_from_prefix(Y_obs, dt, tau, goal)
        goal = np.asarray(goal, dtype=float)
        v = sg_velocity(Y_obs, dt)
        if self.phase == "continue":
            Y = self.dmp.rollout(Y_obs[-1], v, goal, Y_obs[0], tau, fut_times[0], fut_times, self.dt_int)
        else:
            t_rel = fut_times - fut_times[0]
            Y = self.dmp.rollout(Y_obs[-1], v, goal, Y_obs[-1], max(t_rel[-1], 1e-6), 0.0, t_rel, self.dt_int)
        return BaselinePrediction(frame.from_canonical(Y))
