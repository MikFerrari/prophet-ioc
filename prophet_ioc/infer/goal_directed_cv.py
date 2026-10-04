import numpy as np


def predict_goal_directed_cv(
    p0: np.ndarray,
    p_target: np.ndarray,
    fut_times: np.ndarray,
) -> np.ndarray:
    r"""Linear constant-speed interpolation straight to target.

    Computes a straight line in Cartesian space between handover state $p_0$
    and target $p_{\text{target}}$ parameterized by normalized time $\tau \in [0, 1]$:
        $$x(\tau) = p_0 + (p_{\text{target}} - p_0) \tau, \quad \tau = \frac{t - t_0}{T}$$
    where $T = \max(t_f - t_0, \epsilon)$ is the remaining time to arrival.

    Args:
        p0: Starting / handover position of shape (D,).
        p_target: Target position of shape (D,).
        fut_times: Target timestamps of shape (H,).

    Returns:
        Interpolated straight-line trajectory of shape (H, D).
    """
    p0 = np.asarray(p0)
    p_target = np.asarray(p_target)
    T = max(fut_times[-1] - fut_times[0], 1e-6)
    tau = np.clip((fut_times - fut_times[0]) / T, 0.0, 1.0)[:, None]
    return p0[None, :] + (p_target - p0)[None, :] * tau


class GoalDirectedCVBaseline:
    r"""Goal-Directed Constant Velocity Baseline (Linear Straight-Line Interpolation).

    Interpolates between the handover position $p_0$ and the known/inferred
    target position $p_{\text{target}}$ at constant Cartesian speed:
        $$x(t) = p_0 + (p_{\text{target}} - p_0) \frac{t - t_0}{T}$$
    where $T = t_f - t_0$ is the remaining duration to target arrival.
    """

    def predict(self, p0: np.ndarray, p_target: np.ndarray, fut_times: np.ndarray) -> np.ndarray:
        return predict_goal_directed_cv(p0, p_target, fut_times)
