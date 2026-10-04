import numpy as np
from scipy.signal import savgol_filter


def predict_minimum_jerk(
    obs_pos: np.ndarray,
    p_target: np.ndarray,
    fut_times: np.ndarray,
    dt_data: float = 0.01,
) -> np.ndarray:
    r"""Flash & Hogan (1985) Minimum Jerk 5th-order polynomial trajectory.

    Biological reaching movements often exhibit maximally smooth trajectories that
    minimize the total mean squared jerk (third time-derivative of position):
        $$J = \frac{1}{2} \int_0^T \|\dddot{x}(t)\|^2 dt$$
    The Euler-Lagrange optimality condition yields a 5th-order polynomial:
        $$x(\tau) = c_0 + c_1 \tau + c_2 \tau^2 + c_3 \tau^3 + c_4 \tau^4 + c_5 \tau^5, \quad \tau = \frac{t - t_0}{T}$$
    Boundary conditions:
        - At handover $\tau = 0$: $x(0) = x_0$, $\dot{x}(0) = v_0$, $\ddot{x}(0) = a_0$
          (estimated via Savitzky-Golay derivative filtering on the past observation window).
        - At arrival $\tau = 1$: $x(1) = x_{\text{target}}$, $\dot{x}(1) = 0$, $\ddot{x}(1) = 0$.

    Analytical polynomial coefficients:
        $$D = x_f - x_0$$
        $$c_0 = x_0$$
        $$c_1 = v_0 T$$
        $$c_2 = \frac{1}{2} a_0 T^2$$
        $$c_3 = 10 D - 6 c_1 - \frac{3}{2} a_0 T^2$$
        $$c_4 = -15 D + 8 c_1 + \frac{3}{2} a_0 T^2$$
        $$c_5 = 6 D - 3 c_1 - \frac{1}{2} a_0 T^2$$

    Args:
        obs_pos: Observed past positions of shape (N, D).
        p_target: Target position of shape (D,).
        fut_times: Target timestamps of shape (H,).
        dt_data: Sampling period of the observation data in seconds.

    Returns:
        Predicted minimum jerk trajectory of shape (H, D).
    """
    obs_pos = np.asarray(obs_pos)
    p_target = np.asarray(p_target)
    n_obs = len(obs_pos)
    w_len = min(7, n_obs if n_obs % 2 == 1 else n_obs - 1)
    if w_len >= 5:
        v0 = savgol_filter(obs_pos, window_length=w_len, polyorder=2, deriv=1, delta=dt_data, axis=0)[-1]
        a0 = savgol_filter(obs_pos, window_length=w_len, polyorder=2, deriv=2, delta=dt_data, axis=0)[-1]
    elif n_obs >= 2:
        v0 = (obs_pos[-1] - obs_pos[-2]) / dt_data
        a0 = np.zeros(obs_pos.shape[-1])
    else:
        v0 = np.zeros(obs_pos.shape[-1])
        a0 = np.zeros(obs_pos.shape[-1])

    x0 = obs_pos[-1]
    xf = p_target
    T = max(fut_times[-1] - fut_times[0], 1e-6)
    tau = np.clip((fut_times - fut_times[0]) / T, 0.0, 1.0)[:, None]

    D = xf - x0
    c0 = x0
    c1 = v0 * T
    c2 = 0.5 * a0 * (T ** 2)
    c3 = 10.0 * D - 6.0 * c1 - 1.5 * a0 * (T ** 2)
    c4 = -15.0 * D + 8.0 * c1 + 1.5 * a0 * (T ** 2)
    c5 = 6.0 * D - 3.0 * c1 - 0.5 * a0 * (T ** 2)

    return c0 + c1 * tau + c2 * (tau ** 2) + c3 * (tau ** 3) + c4 * (tau ** 4) + c5 * (tau ** 5)


class MinimumJerkBaseline:
    r"""Flash & Hogan (1985) Minimum Jerk Trajectory Model Baseline.

    Biological reaching movements often exhibit maximally smooth trajectories that
    minimize the total mean squared jerk (third time-derivative of position):
        $$J = \frac{1}{2} \int_0^T \|\dddot{x}(t)\|^2 dt$$
    The Euler-Lagrange optimality condition yields a 5th-order polynomial:
        $$x(\tau) = c_0 + c_1 \tau + c_2 \tau^2 + c_3 \tau^3 + c_4 \tau^4 + c_5 \tau^5, \quad \tau = \frac{t - t_0}{T}$$
    Boundary conditions:
        - At handover $\tau = 0$: $x(0) = x_0$, $\dot{x}(0) = v_0$, $\ddot{x}(0) = a_0$
          (estimated via Savitzky-Golay derivative filtering on the past observation window).
        - At arrival $\tau = 1$: $x(1) = x_{\text{target}}$, $\dot{x}(1) = 0$, $\ddot{x}(1) = 0$.
    """

    def __init__(self, dt_data: float = 0.01):
        self.dt_data = dt_data

    def predict(
        self,
        obs_pos: np.ndarray,
        p_target: np.ndarray,
        fut_times: np.ndarray,
        dt_data: float = None,
    ) -> np.ndarray:
        dt = dt_data if dt_data is not None else self.dt_data
        return predict_minimum_jerk(obs_pos, p_target, fut_times, dt_data=dt)
