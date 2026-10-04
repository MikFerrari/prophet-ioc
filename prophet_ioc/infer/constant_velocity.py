import numpy as np
from scipy.signal import savgol_filter


def predict_constant_velocity(
    obs_pos: np.ndarray,
    fut_times: np.ndarray,
    dt_data: float = 0.01,
) -> np.ndarray:
    r"""Zero-order dynamic baseline: projects instantaneous velocity estimated via Savitzky-Golay.

    Extrapolates future trajectory assuming the current instantaneous velocity
    remains constant over the prediction horizon:
        $$x(t) = x(t_0) + v(t_0) (t - t_0)$$
    Instantaneous velocity $v(t_0)$ is estimated from the observation window
    using a Savitzky-Golay polynomial filter to suppress high-frequency mocap noise.

    Args:
        obs_pos: Observed past positions of shape (N, D).
        fut_times: Target timestamps of shape (H,).
        dt_data: Sampling period of the observation data in seconds.

    Returns:
        Predicted future trajectory of shape (H, D).
    """
    obs_pos = np.asarray(obs_pos)
    n_obs = len(obs_pos)
    w_len = min(7, n_obs if n_obs % 2 == 1 else n_obs - 1)
    if w_len >= 5:
        vel_est = savgol_filter(obs_pos, window_length=w_len, polyorder=2, deriv=1, delta=dt_data, axis=0)[-1]
    elif n_obs >= 2:
        vel_est = (obs_pos[-1] - obs_pos[-2]) / dt_data
    else:
        vel_est = np.zeros(obs_pos.shape[-1])

    t_rel = (fut_times - fut_times[0])[:, None]
    return obs_pos[-1:] + vel_est[None, :] * t_rel


class ConstantVelocityBaseline:
    r"""Zero-Order Kinematic Extrapolation Baseline (Constant Velocity).

    Extrapolates future trajectory assuming the current instantaneous velocity
    remains constant over the prediction horizon:
        $$x(t) = x(t_0) + v(t_0) (t - t_0)$$
    Instantaneous velocity $v(t_0)$ is estimated from the observation window
    using a Savitzky-Golay polynomial filter to suppress high-frequency mocap noise.
    """

    def __init__(self, dt_data: float = 0.01):
        self.dt_data = dt_data

    def predict(self, obs_pos: np.ndarray, fut_times: np.ndarray, dt_data: float = None) -> np.ndarray:
        dt = dt_data if dt_data is not None else self.dt_data
        return predict_constant_velocity(obs_pos, fut_times, dt_data=dt)
