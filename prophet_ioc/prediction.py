import os
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List, Dict, Any, Generator
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Safe memory preallocation setting for GPU environments
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jax import jit, lax, vmap, random, jacobian, numpy as jnp

from prophet_ioc.envs.base import Env
from prophet_ioc.envs.nonlinear_reaching import NonlinearReaching, NonlinearReachingParams
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.control import gilqr, make_lqg_approx
from prophet_ioc.control.policy import create_lqg_policy
from prophet_ioc.infer.inv_ilqg import FixedLinearizationInverseGILQG
from prophet_ioc.infer.utils import compute_mle


def cartesian_to_joint(
    pos: np.ndarray,
    l1: float = 0.3,
    l2: float = 0.33,
    dt: float = 0.01,
) -> np.ndarray:
    """Converts 2D Cartesian end-effector positions (p_x, p_y) to 4D state (theta1, theta2, dtheta1, dtheta2).

    Uses geometric inverse kinematics for the 2-link planar arm with positive elbow flexion
    matching the default physical configuration of the reaching model.
    """
    pos = np.asarray(pos, dtype=np.float32)
    single_point = False
    if pos.ndim == 1:
        pos = pos[None, :]
        single_point = True

    H = pos.shape[0]
    states = np.zeros((H, 4), dtype=np.float32)

    # Joint angles via law of cosines
    x, y = pos[:, 0], pos[:, 1]
    r2 = x**2 + y**2
    cos_theta2 = np.clip((r2 - l1**2 - l2**2) / (2.0 * l1 * l2), -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)  # elbow flexion > 0
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))

    states[:, 0] = theta1
    states[:, 1] = theta2

    # Joint velocities via finite differences
    if H > 1:
        dtheta = np.gradient(states[:, :2], dt, axis=0)
        states[:, 2:] = dtheta

    if single_point:
        return states[0]
    return states


def cartesian_to_joint_3d(
    pos: np.ndarray,
    l1: float = 0.30,
    l2: float = 0.33,
    dt: float = 0.01,
) -> np.ndarray:
    """Converts 3D Cartesian end-effector positions (x, y, z) to 6D state (q1, q2, q3, dq1, dq2, dq3).

    Uses analytical closed-form inverse kinematics for the 3-DOF anthropomorphic arm:
        q1: Shoulder yaw (azimuth around vertical z-axis)
        q2: Shoulder pitch (elevation)
        q3: Elbow pitch (flexion)
    """
    pos = np.asarray(pos, dtype=np.float32)
    single_point = False
    if pos.ndim == 1:
        pos = pos[None, :]
        single_point = True

    H = pos.shape[0]
    states = np.zeros((H, 6), dtype=np.float32)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    D2 = r**2 + z**2
    cos_theta3 = np.clip((D2 - l1**2 - l2**2) / (2.0 * l1 * l2), -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    alpha = np.arctan2(z, r)
    beta = np.arctan2(l2 * np.sin(theta3), l1 + l2 * np.cos(theta3))
    theta2 = alpha - beta

    states[:, 0] = theta1
    states[:, 1] = theta2
    states[:, 2] = theta3

    if H > 1:
        dq = np.gradient(states[:, :3], dt, axis=0)
        states[:, 3:] = dq

    if single_point:
        return states[0]
    return states


@dataclass
class PredictionResult:
    """Structured container for probabilistic future trajectory predictions.

    Provides mean estimates, uncertainty (std and full covariance), and confidence
    bounds (Upper and Lower Confidence Limits at the requested confidence level,
    e.g. 95% UCL and LCL) in both state space and Cartesian task space.
    """

    # --- State Space (Joint angles theta1, theta2 & velocities dtheta1, dtheta2) ---
    mean: np.ndarray  # (H_future, state_dim): Expected future state trajectory
    std: np.ndarray  # (H_future, state_dim): State standard deviation
    cov: np.ndarray  # (H_future, state_dim, state_dim): State covariance matrices
    lcl: np.ndarray  # (H_future, state_dim): Lower Confidence Limit (e.g. 95%)
    ucl: np.ndarray  # (H_future, state_dim): Upper Confidence Limit (e.g. 95%)
    samples: Optional[np.ndarray] = None  # (num_samples, H_future, state_dim): Optional rollouts

    # --- Task Space / Cartesian Coordinates (End-effector x, y position in meters) ---
    cartesian_mean: np.ndarray = None  # (H_future, 2 or 3): Expected future Cartesian path
    cartesian_std: np.ndarray = None  # (H_future, 2 or 3): Cartesian standard deviation
    cartesian_cov: np.ndarray = None  # (H_future, 2 or 3, 2 or 3): Cartesian covariance along time
    cartesian_lcl: np.ndarray = None  # (H_future, 2 or 3): Cartesian Lower Confidence Limit
    cartesian_ucl: np.ndarray = None  # (H_future, 2 or 3): Cartesian Upper Confidence Limit
    cartesian_samples: Optional[np.ndarray] = None  # (num_samples, H_future, 2 or 3): Optional rollouts

    # --- Task Space / Cartesian Coordinates: Elbow Keypoint ---
    elbow_mean: Optional[np.ndarray] = None  # (H_future, 2 or 3): Expected future Elbow path
    elbow_std: Optional[np.ndarray] = None   # (H_future, 2 or 3): Elbow standard deviation
    elbow_cov: Optional[np.ndarray] = None   # (H_future, 2 or 3, 2 or 3): Elbow covariance along time
    elbow_lcl: Optional[np.ndarray] = None   # (H_future, 2 or 3): Elbow Lower Confidence Limit
    elbow_ucl: Optional[np.ndarray] = None   # (H_future, 2 or 3): Elbow Upper Confidence Limit
    elbow_samples: Optional[np.ndarray] = None  # (num_samples, H_future, 2 or 3): Optional rollouts
    observed_elbow: Optional[np.ndarray] = None  # (H_obs, 2 or 3): Observed Elbow positions

    # --- Observation Context & Metadata ---
    confidence_level: float = 0.95  # e.g., 0.95
    observed_steps: int = 0  # Length of observed trajectory prefix (H_obs)
    future_steps: int = 0  # Length of predicted future horizon (H_future)
    observed_state: Optional[np.ndarray] = None  # (H_obs, state_dim): Observed joint states
    observed_cartesian: Optional[np.ndarray] = None  # (H_obs, 2 or 3): Observed Cartesian positions
    nominal_trajectory: Optional[np.ndarray] = None  # (H_future, state_dim): Optimal noiseless path
    nominal_cartesian: Optional[np.ndarray] = None  # (H_future, 2 or 3): Optimal noiseless Cartesian path
    controls: Optional[np.ndarray] = None  # (H_future, action_dim): Optimal nominal control sequence
    time_indices: Optional[np.ndarray] = None  # Future step indices [t_obs ... t_obs + H_future - 1]
    mode: str = "analytical"  # "analytical", "monte_carlo", or "both"
    latency_ms: float = 0.0  # Real-time inference latency in milliseconds

    @property
    def uncertainty(self) -> np.ndarray:
        """Alias for state standard deviation uncertainty."""
        return self.std

    @property
    def cartesian_uncertainty(self) -> np.ndarray:
        """Alias for Cartesian standard deviation uncertainty."""
        return self.cartesian_std

    @property
    def elbow_uncertainty(self) -> Optional[np.ndarray]:
        """Alias for Elbow Cartesian standard deviation uncertainty."""
        return self.elbow_std

    def ade(self, ground_truth_future: Union[np.ndarray, jnp.ndarray]) -> float:
        """Average Displacement Error (ADE) in Cartesian meters against actual future trajectory."""
        gt = np.asarray(ground_truth_future)
        H = min(len(self.cartesian_mean), len(gt))
        dim = self.cartesian_mean.shape[-1]
        errors = np.linalg.norm(self.cartesian_mean[:H] - gt[:H, :dim], axis=-1)
        return float(np.mean(errors))

    def fde(self, ground_truth_future: Union[np.ndarray, jnp.ndarray]) -> float:
        """Final Displacement Error (FDE) in Cartesian meters at the final prediction step."""
        gt = np.asarray(ground_truth_future)
        dim = self.cartesian_mean.shape[-1]
        return float(np.linalg.norm(self.cartesian_mean[-1] - gt[-1, :dim]))

    def coverage_rate(self, ground_truth_future: Union[np.ndarray, jnp.ndarray]) -> float:
        """Fraction of ground truth Cartesian coordinates falling within [LCL, UCL] bounds."""
        gt = np.asarray(ground_truth_future)
        H = min(len(self.cartesian_mean), len(gt))
        dim = self.cartesian_mean.shape[-1]
        inside = np.ones(H, dtype=bool)
        for d in range(dim):
            inside &= (gt[:H, d] >= self.cartesian_lcl[:H, d]) & (gt[:H, d] <= self.cartesian_ucl[:H, d])
        return float(np.mean(inside))

    def ade_elbow(self, ground_truth_future: Union[np.ndarray, jnp.ndarray]) -> float:
        """Average Displacement Error (ADE) in Cartesian meters for Elbow."""
        if self.elbow_mean is None or ground_truth_future is None:
            return 0.0
        gt = np.asarray(ground_truth_future)
        H = min(len(self.elbow_mean), len(gt))
        dim = self.elbow_mean.shape[-1]
        errors = np.linalg.norm(self.elbow_mean[:H] - gt[:H, :dim], axis=-1)
        return float(np.mean(errors))

    def fde_elbow(self, ground_truth_future: Union[np.ndarray, jnp.ndarray]) -> float:
        """Final Displacement Error (FDE) in Cartesian meters for Elbow."""
        if self.elbow_mean is None or ground_truth_future is None:
            return 0.0
        gt = np.asarray(ground_truth_future)
        dim = self.elbow_mean.shape[-1]
        return float(np.linalg.norm(self.elbow_mean[-1] - gt[-1, :dim]))

    def coverage_rate_elbow(self, ground_truth_future: Union[np.ndarray, jnp.ndarray]) -> float:
        """Fraction of ground truth Elbow Cartesian coordinates falling within [LCL, UCL] bounds."""
        if self.elbow_mean is None or self.elbow_lcl is None or self.elbow_ucl is None or ground_truth_future is None:
            return 1.0
        gt = np.asarray(ground_truth_future)
        H = min(len(self.elbow_mean), len(gt))
        dim = self.elbow_mean.shape[-1]
        inside = np.ones(H, dtype=bool)
        for d in range(dim):
            inside &= (gt[:H, d] >= self.elbow_lcl[:H, d]) & (gt[:H, d] <= self.elbow_ucl[:H, d])
        return float(np.mean(inside))

    def to_dict(self) -> Dict[str, Any]:
        """Converts structured result to a standard dictionary."""
        return {
            "mean": self.mean,
            "uncertainty": self.uncertainty,
            "cov": self.cov,
            "lcl": self.lcl,
            "ucl": self.ucl,
            "samples": self.samples,
            "cartesian_mean": self.cartesian_mean,
            "cartesian_uncertainty": self.cartesian_uncertainty,
            "cartesian_cov": self.cartesian_cov,
            "cartesian_lcl": self.cartesian_lcl,
            "cartesian_ucl": self.cartesian_ucl,
            "cartesian_samples": self.cartesian_samples,
            "confidence_level": self.confidence_level,
            "observed_steps": self.observed_steps,
            "future_steps": self.future_steps,
            "mode": self.mode,
            "latency_ms": self.latency_ms,
        }

    def summary(self) -> str:
        """Human-readable text summary of prediction metrics and confidence limits."""
        hz_str = f"{1000/max(self.latency_ms, 1e-3):.1f} Hz" if self.latency_ms > 0 else "N/A"
        dim = self.cartesian_mean.shape[-1]
        if dim == 3:
            coord_str = lambda arr: f"({arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f}) m"
            std_str = lambda arr: f"({arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f}) m"
            lbl = "Final Std (x, y, z)"
        else:
            coord_str = lambda arr: f"({arr[0]:.4f}, {arr[1]:.4f}) m"
            std_str = lambda arr: f"({arr[0]:.4f}, {arr[1]:.4f}) m"
            lbl = "Final Std (x, y)"

        lines = [
            f"PredictionResult (Mode: {self.mode}, Confidence: {self.confidence_level * 100:.1f}%)",
            f"  Observed horizon : {self.observed_steps} steps",
            f"  Future horizon   : {self.future_steps} steps",
            f"  Online latency   : {self.latency_ms:.2f} ms ({hz_str})",
            f"  Cartesian Start  : ({self.cartesian_mean[0, 0]:.4f}, {self.cartesian_mean[0, 1]:.4f}) m",
            f"  Cartesian Target : ({self.cartesian_mean[-1, 0]:.4f}, {self.cartesian_mean[-1, 1]:.4f}) m",
            f"  Cartesian 95% UCL: ({self.cartesian_ucl[-1, 0]:.4f}, {self.cartesian_ucl[-1, 1]:.4f}) m",
            f"  Cartesian 95% LCL: ({self.cartesian_lcl[-1, 0]:.4f}, {self.cartesian_lcl[-1, 1]:.4f}) m",
            f"  Final Std (x, y) : ({self.cartesian_std[-1, 0]:.4f}, {self.cartesian_std[-1, 1]:.4f}) m",
            f"  Cartesian Start  : {coord_str(self.cartesian_mean[0])}",
            f"  Cartesian Target : {coord_str(self.cartesian_mean[-1])}",
            f"  Cartesian 95% UCL: {coord_str(self.cartesian_ucl[-1])}",
            f"  Cartesian 95% LCL: {coord_str(self.cartesian_lcl[-1])}",
            f"  {lbl:<18}: {std_str(self.cartesian_std[-1])}",
        ]
        return "\n".join(lines)


# ==============================================================================
# JIT-Compiled Real-Time Solvers (Ahead-Of-Time Pre-compiled for > 50-70 Hz)
# ==============================================================================

@partial(jit, static_argnames=("env", "future_steps", "max_iter"))
def _jit_analytical_predict(env, x_handover, u_init, params, future_steps, max_iter):
    """Closed-form analytical LQG covariance and confidence bound propagation.

    Solves for the optimal nominal trajectory and feedback gains in a single JIT pass,
    linearizes dynamics and noise, and analytically propagates Kalman error covariance P_t
    and closed-loop trajectory covariance Sigma_t via coupled Lyapunov/Riccati equations.
    Runs in ~14-18 ms (> 55-70 Hz) without any Monte Carlo sampling.
    """
    # 1. iLQR solve for optimal nominal path and feedback gains
    gains, xbar, ubar = gilqr.solve(
        p=env,
        x0=x_handover,
        U_init=u_init,
        params=params,
        max_iter=max_iter,
    )

    # 2. Linearization of dynamics and noises along nominal trajectory
    spec = make_lqg_approx(env, params)(xbar, ubar)

    # 3. Closed-form covariance propagation loop
    P0 = jnp.eye(env.state_shape[0]) * 1e-4
    Sigma_hat_0 = jnp.zeros((env.state_shape[0], env.state_shape[0]))

    def cov_scan(carry, t):
        P_t, Sigma_hat_t = carry

        A_t = spec.A[t]
        B_t = spec.B[t]
        F_t = spec.F[t]
        V_t = spec.V[t]
        W_t = spec.W[t]
        L_t = gains.L[t]

        # Innovation covariance and Kalman gain
        Innov_t = F_t @ P_t @ F_t.T + W_t @ W_t.T
        K_t = A_t @ P_t @ F_t.T @ jnp.linalg.inv(Innov_t)

        # Kalman filter error covariance propagation
        P_next = V_t @ V_t.T + (A_t - K_t @ F_t) @ P_t @ A_t.T

        # Belief trajectory covariance propagation (closed-loop)
        closed_loop_A = A_t + B_t @ L_t
        Sigma_hat_next = closed_loop_A @ Sigma_hat_t @ closed_loop_A.T + K_t @ Innov_t @ K_t.T

        # Total state covariance along horizon
        Sigma_total = Sigma_hat_t + P_t
        return (P_next, Sigma_hat_next), Sigma_total

    _, Sigma_history = lax.scan(cov_scan, (P0, Sigma_hat_0), jnp.arange(future_steps))

    # 4. State mean and standard deviation
    mean_state = xbar[1:]
    std_state = jnp.sqrt(jnp.clip(vmap(jnp.diag)(Sigma_history), 1e-8, None))

    # 5. Cartesian task space mapping via kinematic Jacobian J_t = de/dx(xbar_t)
    cart_mean = vmap(env.e)(mean_state)
    J = vmap(jacobian(env.e))(mean_state)
    cart_cov = jnp.einsum("tik,tkl,tjl->tij", J, Sigma_history, J)
    cart_std = jnp.sqrt(jnp.clip(vmap(jnp.diag)(cart_cov), 1e-8, None))

    # 6. Elbow Cartesian covariance propagation
    if hasattr(env, "elbow"):
        elbow_mean = vmap(env.elbow)(mean_state)
        J_e = vmap(jacobian(env.elbow))(mean_state)
        elbow_cov = jnp.einsum("tik,tkl,tjl->tij", J_e, Sigma_history, J_e)
        elbow_std = jnp.sqrt(jnp.clip(vmap(jnp.diag)(elbow_cov), 1e-8, None))
    else:
        elbow_mean = jnp.zeros_like(cart_mean)
        elbow_cov = jnp.zeros_like(cart_cov)
        elbow_std = jnp.zeros_like(cart_std)

    return (
        xbar,
        ubar,
        mean_state,
        std_state,
        Sigma_history,
        cart_mean,
        cart_std,
        cart_cov,
        elbow_mean,
        elbow_std,
        elbow_cov,
    )


@partial(jit, static_argnames=("env", "ekf", "future_steps", "num_samples", "max_iter"))
def _jit_predict_rollout(env, ekf, x_handover, u_init, params, key, future_steps, num_samples, max_iter):
    """JIT-compiled optimal trajectory solve and parallel Monte Carlo forward rollouts."""
    gains, xbar, ubar = gilqr.solve(
        p=env,
        x0=x_handover,
        U_init=u_init,
        params=params,
        max_iter=max_iter,
    )
    policy = create_lqg_policy(gains, xbar, ubar)

    def rollout_single(subkey):
        def scan_body(carry, t):
            (state, k), belief = carry
            k_act, k_step, k_next = random.split(k, 3)
            act_noise = random.normal(k_act, shape=env.action_shape)
            action = policy(t, belief, act_noise)
            env_state, new_belief, cost = ekf.step(
                (state, k_step), belief, action, params
            )
            return (env_state, new_belief), env_state[0]

        b_init = (x_handover, jnp.eye(env.state_shape[0]) * 1e-4)
        init_carry = ((x_handover, subkey), b_init)
        _, future_path = lax.scan(scan_body, init_carry, jnp.arange(future_steps))
        return future_path

    keys = random.split(key, num_samples)
    future_samples = vmap(rollout_single)(keys)
    cartesian_samples = vmap(vmap(env.e))(future_samples)
    if hasattr(env, "elbow"):
        elbow_samples = vmap(vmap(env.elbow))(future_samples)
    else:
        elbow_samples = jnp.zeros_like(cartesian_samples)
    return xbar, ubar, future_samples, cartesian_samples, elbow_samples


# ==============================================================================
# Predictor Class
# ==============================================================================

class MovingWindowMotionPredictor:
    """Probabilistic motion predictor for human reaching movements using IOC and optimal feedback control.

    Real-Time Features:
    1. Ahead-of-Time (AOT) JIT Warmup: Pre-compiles all XLA kernels via `warmup()` at startup.
    2. Analytical Closed-Loop LQG Covariance: Calculates exact 95% UCL/LCL confidence bounds in ~14-17 ms (> 55-70 Hz)
       without Monte Carlo sampling.
    3. Warm-Started Receding-Horizon Control: Reuses shifted control signals from previous steps in `rolling_predict`.
    4. Flexible Modes: Supports "analytical", "monte_carlo", and "both" (analytical tube + sample rollouts).
    """

    def __init__(
        self,
        env: Optional[Env] = None,
        params: Optional[Any] = None,
        b0: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        ioc_cls: type = FixedLinearizationInverseGILQG,
        default_horizon: int = 50,
        seed: int = 1,
    ):
        self.env = env if env is not None else NonlinearReaching()
        ParamsType = self.env.get_params_type()

        # If parameters provided, ensure they are JAX float32 arrays
        if params is not None:
            self.params = ParamsType(
                **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in params._asdict().items()}
            )
            self.is_trained = True
        else:
            self.params = ParamsType(
                action_cost=jnp.float32(1e-4),
                velocity_cost=jnp.float32(1e-2),
                motor_noise=jnp.float32(0.1),
                obs_noise=jnp.float32(1.0),
            )
            self.is_trained = False

        self.default_horizon = default_horizon
        self.ioc_cls = ioc_cls
        self.key = random.PRNGKey(seed)

        # Setup initial belief
        x0 = self.env._reset(None, self.params)
        self.b0 = b0 if b0 is not None else (x0, jnp.eye(x0.shape[0]) * 1e-4)
        self.ekf = EKFWrapper(self.env.__class__)(b0=self.b0)
        self._fk = jit(vmap(self.env.e))
        if hasattr(self.env, "elbow"):
            self._fk_elbow = jit(vmap(self.env.elbow))
        else:
            self._fk_elbow = None

    def warmup(
        self,
        future_steps: int = 30,
        num_samples: int = 50,
        mode: str = "both",
        max_iter: int = 3,
    ):
        """Ahead-Of-Time (AOT) pre-compiles JAX XLA kernels for real-time online inference."""
        dummy_x = self.b0[0]
        dummy_u = jnp.zeros((future_steps, self.env.action_shape[0]))
        dummy_key = random.PRNGKey(0)

        # Pre-compile forward kinematics
        self._fk(jnp.tile(dummy_x, (future_steps, 1))).block_until_ready()
        if self._fk_elbow is not None:
            self._fk_elbow(jnp.tile(dummy_x, (future_steps, 1))).block_until_ready()

        if mode in ("analytical", "both"):
            res = _jit_analytical_predict(
                self.env, dummy_x, dummy_u, self.params, future_steps, max_iter
            )
            res[-1].block_until_ready()

        if mode in ("monte_carlo", "both"):
            res = _jit_predict_rollout(
                self.env, self.ekf, dummy_x, dummy_u, self.params, dummy_key,
                future_steps, num_samples, max_iter
            )
            res[-1].block_until_ready()

        # Exercise the full end-to-end predict pipeline to warm up host/device conversions
        dummy_obs = jnp.tile(dummy_x, (10, 1))
        self.predict(
            observed=dummy_obs,
            future_steps=future_steps,
            mode=mode,
            num_samples=num_samples,
            max_iter=max_iter,
        )

    def fit(
        self,
        trajectories: Union[np.ndarray, jnp.ndarray],
        restarts: int = 10,
        bounds: Optional[Tuple[Any, Any]] = None,
        optim: str = "L-BFGS-B",
        seed: Optional[int] = None,
    ) -> Any:
        """Trains IOC cost and noise parameters on full demonstrated trajectories."""
        trajectories = jnp.asarray(trajectories, dtype=jnp.float32)
        if bounds is None:
            bounds = self.env.get_params_bounds()

        if seed is not None:
            key = random.PRNGKey(seed)
        else:
            self.key, key = random.split(self.key)

        mean_x0 = trajectories.mean(axis=0)[0]
        ioc = self.ioc_cls(self.env, b0=(mean_x0, jnp.eye(self.env.state_shape[0])))

        result = compute_mle(
            xs=trajectories,
            ioc=ioc,
            key=key,
            restarts=restarts,
            bounds=bounds,
            optim=optim,
        )
        self.params = result.params
        self.is_trained = True
        return result

    def fit_window(
        self,
        trajectories: Union[np.ndarray, jnp.ndarray],
        window_start: int = 0,
        window_length: int = 20,
        restarts: int = 5,
        bounds: Optional[Tuple[Any, Any]] = None,
        optim: str = "L-BFGS-B",
        seed: Optional[int] = None,
    ) -> Any:
        """Trains IOC parameters on an observed moving window chunk of trajectories."""
        trajectories = jnp.asarray(trajectories, dtype=jnp.float32)
        if trajectories.ndim == 2:
            trajectories = trajectories[None, ...]

        window_end = min(window_start + window_length, trajectories.shape[1])
        chunk = trajectories[:, window_start:window_end]

        if seed is not None:
            key = random.PRNGKey(seed)
        else:
            self.key, key = random.split(self.key)

        if bounds is None:
            bounds = self.env.get_params_bounds()

        chunk_x0 = chunk.mean(axis=0)[0]
        ioc_chunk = self.ioc_cls(self.env, b0=(chunk_x0, jnp.eye(self.env.state_shape[0])))

        result = compute_mle(
            xs=chunk,
            ioc=ioc_chunk,
            key=key,
            restarts=restarts,
            bounds=bounds,
            optim=optim,
        )
        self.params = result.params
        self.is_trained = True
        return result

    def predict(
        self,
        observed: Union[np.ndarray, jnp.ndarray],
        future_steps: Optional[int] = None,
        mode: str = "analytical",
        num_samples: int = 50,
        confidence_level: float = 0.95,
        max_iter: int = 3,
        u_init: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> PredictionResult:
        """Predicts the future chunk of motion conditioned on the observed horizon.

        Args:
            observed: Observed trajectory chunk (joint states or Cartesian positions).
            future_steps: Number of future time steps to predict ahead (H_future).
            mode: "analytical" (fastest, > 50-70 Hz), "monte_carlo" (stochastic rollouts), or "both".
            num_samples: Number of stochastic rollouts when mode is "monte_carlo" or "both".
            confidence_level: Confidence level for UCL and LCL bounds (default: 0.95).
            max_iter: Max iLQR iterations (default 3 for real-time online tracking).
            u_init: Optional warm-start control sequence from previous time step.
            seed: Optional random seed.

        Returns:
            Structured `PredictionResult`.
        """
        t0_start = time.perf_counter()
        arr = np.asarray(observed, dtype=np.float32)

        # Parse observed prefix
        if arr.ndim == 3:
            observed_state = arr.mean(axis=0)
        elif arr.ndim == 2:
            if arr.shape[1] == 2:
                observed_state = cartesian_to_joint(
                    arr, l1=self.env.l1, l2=self.env.l2, dt=self.env.dt
                )
            elif arr.shape[1] == 3 and hasattr(self.env, "ik"):
                observed_state = cartesian_to_joint_3d(
                    arr, l1=self.env.l1, l2=self.env.l2, dt=self.env.dt
                )
            else:
                observed_state = arr
        elif arr.ndim == 1:
            if len(arr) == 2:
                observed_state = cartesian_to_joint(
                    arr[None, :], l1=self.env.l1, l2=self.env.l2, dt=self.env.dt
                )
            elif len(arr) == 3 and hasattr(self.env, "ik"):
                observed_state = cartesian_to_joint_3d(
                    arr[None, :], l1=self.env.l1, l2=self.env.l2, dt=self.env.dt
                )
            else:
                observed_state = arr[None, :]
        else:
            raise ValueError(f"Unsupported observed trajectory shape: {arr.shape}")

        H_obs = observed_state.shape[0]

        if future_steps is None:
            future_steps = max(10, self.default_horizon - H_obs)

        x_handover = jnp.asarray(observed_state[-1], dtype=jnp.float32)

        # Setup initial control guess (warm-started if provided)
        if u_init is not None:
            u_init_arr = jnp.asarray(u_init, dtype=jnp.float32)
            if u_init_arr.shape[0] != future_steps:
                if u_init_arr.shape[0] > future_steps:
                    u_init_arr = u_init_arr[:future_steps]
                else:
                    pad = jnp.zeros((future_steps - u_init_arr.shape[0], self.env.action_shape[0]))
                    u_init_arr = jnp.vstack([u_init_arr, pad])
        else:
            u_init_arr = jnp.zeros((future_steps, self.env.action_shape[0]))

        # Normal critical value z (e.g. z = 1.95996 for 95%)
        z = math.sqrt(2.0) * float(lax.erf_inv(confidence_level))

        # Mode 1: Analytical Covariance Propagation (> 50-70 Hz)
        if mode in ("analytical", "both"):
            (
                xbar,
                ubar,
                mean_s_jax,
                std_s_jax,
                cov_s_jax,
                cart_m_jax,
                cart_s_jax,
                cart_c_jax,
                elbow_m_jax,
                elbow_s_jax,
                elbow_c_jax,
            ) = _jit_analytical_predict(
                self.env, x_handover, u_init_arr, self.params, future_steps, max_iter
            )
            # Synchronize JAX array
            cart_c_jax.block_until_ready()

            mean_state = np.array(mean_s_jax)
            std_state = np.array(std_s_jax)
            cov_state = np.array(cov_s_jax)
            cartesian_mean = np.array(cart_m_jax)
            cartesian_std = np.array(cart_s_jax)
            cartesian_cov = np.array(cart_c_jax)

            has_elbow = hasattr(self.env, "elbow")
            elbow_mean = np.array(elbow_m_jax) if has_elbow else None
            elbow_std = np.array(elbow_s_jax) if has_elbow else None
            elbow_cov = np.array(elbow_c_jax) if has_elbow else None

            lcl_state = mean_state - z * std_state
            ucl_state = mean_state + z * std_state
            cartesian_lcl = cartesian_mean - z * cartesian_std
            cartesian_ucl = cartesian_mean + z * cartesian_std
            elbow_lcl = (elbow_mean - z * elbow_std) if elbow_mean is not None else None
            elbow_ucl = (elbow_mean + z * elbow_std) if elbow_mean is not None else None

            controls = np.array(ubar)
            nominal_trajectory = np.array(xbar[1:])
            nominal_cartesian = cartesian_mean

            future_samples = None
            cartesian_samples = None
            elbow_samples = None

            # If "both", also generate stochastic rollouts for visualization
            if mode == "both":
                if seed is not None:
                    key = random.PRNGKey(seed)
                else:
                    self.key, key = random.split(self.key)
                _, _, future_samples_jax, cartesian_samples_jax, elbow_samples_jax = _jit_predict_rollout(
                    self.env, self.ekf, x_handover, u_init_arr, self.params, key,
                    future_steps, num_samples, max_iter
                )
                future_samples = np.array(future_samples_jax)
                cartesian_samples = np.array(cartesian_samples_jax)
                elbow_samples = np.array(elbow_samples_jax) if has_elbow else None

        # Mode 2: Pure Monte Carlo Rollouts
        else:
            if seed is not None:
                key = random.PRNGKey(seed)
            else:
                self.key, key = random.split(self.key)

            has_elbow = hasattr(self.env, "elbow")
            xbar, ubar, future_samples_jax, cartesian_samples_jax, elbow_samples_jax = _jit_predict_rollout(
                self.env, self.ekf, x_handover, u_init_arr, self.params, key,
                future_steps, num_samples, max_iter
            )
            cartesian_samples_jax.block_until_ready()

            future_samples = np.array(future_samples_jax)
            cartesian_samples = np.array(cartesian_samples_jax)
            elbow_samples = np.array(elbow_samples_jax) if has_elbow else None

            mean_state = np.mean(future_samples, axis=0)
            std_state = np.std(future_samples, axis=0)
            cov_state = np.zeros((future_steps, self.env.state_shape[0], self.env.state_shape[0]))
            for t in range(future_steps):
                cov_state[t] = np.cov(future_samples[:, t, :], rowvar=False)

            cartesian_mean = np.mean(cartesian_samples, axis=0)
            cartesian_std = np.std(cartesian_samples, axis=0)
            cart_dim = cartesian_samples.shape[-1]
            cartesian_cov = np.zeros((future_steps, cart_dim, cart_dim))
            for t in range(future_steps):
                cartesian_cov[t] = np.cov(cartesian_samples[:, t, :], rowvar=False)

            if elbow_samples is not None:
                elbow_mean = np.mean(elbow_samples, axis=0)
                elbow_std = np.std(elbow_samples, axis=0)
                e_dim = elbow_samples.shape[-1]
                elbow_cov = np.zeros((future_steps, e_dim, e_dim))
                for t in range(future_steps):
                    elbow_cov[t] = np.cov(elbow_samples[:, t, :], rowvar=False)
                elbow_lcl = elbow_mean - z * elbow_std
                elbow_ucl = elbow_mean + z * elbow_std
            else:
                elbow_mean = None
                elbow_std = None
                elbow_cov = None
                elbow_lcl = None
                elbow_ucl = None

            lcl_state = mean_state - z * std_state
            ucl_state = mean_state + z * std_state
            cartesian_lcl = cartesian_mean - z * cartesian_std
            cartesian_ucl = cartesian_mean + z * cartesian_std

            controls = np.array(ubar)
            nominal_trajectory = np.array(xbar[1:])
            nominal_cartesian = np.array(self._fk(nominal_trajectory))

        t_elapsed_ms = (time.perf_counter() - t0_start) * 1000

        observed_cartesian = np.array(self._fk(observed_state))
        if self._fk_elbow is not None:
            observed_elbow = np.array(self._fk_elbow(observed_state))
        elif hasattr(self.env, "elbow"):
            observed_elbow = np.array(vmap(self.env.elbow)(observed_state))
        else:
            observed_elbow = None

        time_indices = np.arange(H_obs, H_obs + future_steps)

        return PredictionResult(
            mean=mean_state,
            std=std_state,
            cov=cov_state,
            lcl=lcl_state,
            ucl=ucl_state,
            samples=future_samples,
            cartesian_mean=cartesian_mean,
            cartesian_std=cartesian_std,
            cartesian_cov=cartesian_cov,
            cartesian_lcl=cartesian_lcl,
            cartesian_ucl=cartesian_ucl,
            cartesian_samples=cartesian_samples,
            elbow_mean=elbow_mean,
            elbow_std=elbow_std,
            elbow_cov=elbow_cov,
            elbow_lcl=elbow_lcl,
            elbow_ucl=elbow_ucl,
            elbow_samples=elbow_samples,
            observed_elbow=observed_elbow,
            confidence_level=confidence_level,
            observed_steps=H_obs,
            future_steps=future_steps,
            observed_state=observed_state,
            observed_cartesian=observed_cartesian,
            nominal_trajectory=nominal_trajectory,
            nominal_cartesian=nominal_cartesian,
            controls=controls,
            time_indices=time_indices,
            mode=mode,
            latency_ms=t_elapsed_ms,
        )

    def rolling_predict(
        self,
        trajectory: Union[np.ndarray, jnp.ndarray],
        window_size: int = 15,
        future_steps: int = 20,
        step_size: int = 1,
        mode: str = "analytical",
        num_samples: int = 20,
        confidence_level: float = 0.95,
        max_iter: int = 2,
        warm_start: bool = True,
    ) -> Generator[Tuple[int, np.ndarray, PredictionResult], None, None]:
        """Slides a moving window across an ongoing trajectory using online warm-starting.

        Yields:
            (step_index, observed_chunk, prediction_result) at each sliding window position.
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        T = traj.shape[0]
        last_controls = None

        for t_end in range(window_size, T - 1, step_size):
            t_start = max(0, t_end - window_size)
            chunk = traj[t_start:t_end]
            h_future = min(future_steps, T - t_end)
            if h_future <= 1:
                break

            # Warm-start from shifted previous control sequence
            u_init = None
            if warm_start and last_controls is not None:
                if len(last_controls) == h_future:
                    u_init = jnp.vstack([last_controls[step_size:], jnp.tile(last_controls[-1:], (step_size, 1))])
                elif len(last_controls) > h_future:
                    u_init = last_controls[:h_future]
                else:
                    pad = jnp.zeros((h_future - len(last_controls), self.env.action_shape[0]))
                    u_init = jnp.vstack([last_controls, pad])

            result = self.predict(
                observed=chunk,
                future_steps=h_future,
                mode=mode,
                num_samples=num_samples,
                confidence_level=confidence_level,
                max_iter=max_iter,
                u_init=u_init,
            )
            last_controls = result.controls
            yield t_end, chunk, result

    def plot_prediction(
        self,
        result: PredictionResult,
        ground_truth_future: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualizes the observed prefix, predicted mean path, 95% confidence bounds, and true future."""
        create_fig = ax is None
        if create_fig:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
        else:
            fig = ax.get_figure()

        # 1. Plot observed chunk
        obs_cart = result.observed_cartesian
        ax.plot(
            obs_cart[:, 0],
            obs_cart[:, 1],
            color="#2b5c8f",
            linewidth=2.8,
            label=f"Observed ({result.observed_steps} steps)",
            zorder=5,
        )
        ax.scatter(obs_cart[0, 0], obs_cart[0, 1], color="black", s=60, zorder=7, label="Start")
        ax.scatter(
            obs_cart[-1, 0],
            obs_cart[-1, 1],
            color="#2b5c8f",
            s=70,
            marker="o",
            zorder=7,
            label="Handover",
        )

        # 2. Plot ground truth future if provided
        if ground_truth_future is not None:
            gt = np.asarray(ground_truth_future)
            if gt.shape[-1] > 2:
                gt_cart = np.array(vmap(self.env.e)(gt))
            else:
                gt_cart = gt
            ax.plot(
                gt_cart[:, 0],
                gt_cart[:, 1],
                color="dimgray",
                linestyle="--",
                linewidth=2.0,
                alpha=0.85,
                label=f"True Future ({len(gt_cart)} steps)",
                zorder=4,
            )

        # 3. Plot sample paths if available
        if result.cartesian_samples is not None and len(result.cartesian_samples) > 0:
            samples = result.cartesian_samples
            num_to_plot = min(25, len(samples))
            for s in range(num_to_plot):
                ax.plot(
                    samples[s, :, 0],
                    samples[s, :, 1],
                    color="#e66101",
                    alpha=0.15,
                    linewidth=0.9,
                    zorder=2,
                )

        # 4. Plot 2D spatial confidence tube along trajectory normal
        mean = result.cartesian_mean
        cov = result.cartesian_cov
        z = math.sqrt(2.0) * float(lax.erf_inv(result.confidence_level))

        # Include handover point so the spatial tube connects smoothly to observed path
        handover_p = result.observed_cartesian[-1:]
        handover_cov = cov[:1] * 0.1
        full_mean = np.vstack([handover_p, mean])
        full_cov = np.vstack([handover_cov, cov])

        # Compute trajectory velocity and normal vectors perpendicular to path
        dx = np.gradient(full_mean[:, 0])
        dy = np.gradient(full_mean[:, 1])
        vel = np.stack([dx, dy], axis=-1)
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        speed = np.maximum(speed, 1e-6)
        tangents = vel / speed
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)

        # Cross-track normal standard deviation: sigma_perp = sqrt(n^T * Cov * n)
        sigma_perp = np.zeros(len(full_mean))
        for t in range(len(full_mean)):
            sigma_perp[t] = np.sqrt(np.clip(normals[t] @ full_cov[t] @ normals[t], 1e-8, None))

        left_bound = full_mean + z * sigma_perp[:, None] * normals
        right_bound = full_mean - z * sigma_perp[:, None] * normals

        tube_x = np.concatenate([left_bound[:, 0], right_bound[::-1, 0]])
        tube_y = np.concatenate([left_bound[:, 1], right_bound[::-1, 1]])

        ax.plot(
            left_bound[:, 0],
            left_bound[:, 1],
            color="#fdb863",
            linestyle=":",
            linewidth=1.6,
            label=f"{result.confidence_level * 100:.0f}% Upper Bound",
            zorder=3,
        )
        ax.plot(
            right_bound[:, 0],
            right_bound[:, 1],
            color="#fdb863",
            linestyle=":",
            linewidth=1.6,
            label=f"{result.confidence_level * 100:.0f}% Lower Bound",
            zorder=3,
        )
        ax.fill(
            tube_x,
            tube_y,
            color="#fdb863",
            alpha=0.25,
            label=f"{result.confidence_level * 100:.0f}% Spatial Tube",
            zorder=1,
        )

        # Draw 2D covariance ellipses at key waypoints to visualize full 2D spatial uncertainty
        ellipse_steps = np.linspace(1, len(full_mean) - 1, min(6, len(full_mean) - 1), dtype=int)
        for idx in ellipse_steps:
            val, vec = np.linalg.eigh(full_cov[idx])
            angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
            w, h = 2 * z * np.sqrt(np.clip(val, 1e-8, None))
            ell = Ellipse(
                xy=full_mean[idx],
                width=w,
                height=h,
                angle=angle,
                edgecolor="#d95f02",
                facecolor="none",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                zorder=2,
            )
            ax.add_patch(ell)

        # 5. Plot predicted mean trajectory
        ax.plot(
            result.cartesian_mean[:, 0],
            result.cartesian_mean[:, 1],
            color="#d95f02",
            linewidth=3.0,
            label=f"Predicted Mean ({result.future_steps} steps)",
            zorder=6,
        )

        # 6. Target marker
        target = np.array(self.env.target)
        ax.scatter(
            target[0],
            target[1],
            color="crimson",
            marker="X",
            s=100,
            linewidth=2,
            zorder=7,
            label="Target",
        )

        # Format plot
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        hz_label = f" | {1000/max(result.latency_ms, 1e-3):.1f} Hz" if result.latency_ms > 0 else ""
        plot_title = (
            title
            or f"Real-Time Motion Prediction ({result.mode.upper()})\n"
               f"Observed: {result.observed_steps} steps | Future: {result.future_steps} steps | Latency: {result.latency_ms:.1f} ms{hz_label}"
        )
        ax.set_title(plot_title, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8.5)

        # Display metric text box if ground truth given
        if ground_truth_future is not None:
            ade_val = result.ade(ground_truth_future)
            fde_val = result.fde(ground_truth_future)
            cov_val = result.coverage_rate(ground_truth_future)
            hz_val = 1000.0 / max(result.latency_ms, 1e-3) if result.latency_ms > 0 else 0.0
            text_box = (
                r"$\mathbf{Real-Time\ Metrics:}$" + "\n"
                f"Mode: {result.mode}\n"
                f"Latency: {result.latency_ms:.1f} ms ({hz_val:.1f} Hz)\n"
                f"ADE: {ade_val * 1000:.1f} mm\n"
                f"FDE: {fde_val * 1000:.1f} mm\n"
                f"95% Cov: {cov_val * 100:.1f}%"
            )
            props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc")
            ax.text(
                0.04,
                0.96,
                text_box,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8.5,
                bbox=props,
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        if show and create_fig:
            plt.show(block=True)

        return fig

    def plot_prediction_3d(
        self,
        result: PredictionResult,
        ground_truth_future: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        ground_truth_joint_future: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualizes:
        1. Left subplots: Observed prefix, predicted future, UCL/LCL bounds, and ground truth for all joints.
        2. Right 3D subplot: 3D task-space trajectory with 3D covariance ellipsoids and spatial confidence tube.
        """
        fig = plt.figure(figsize=(16, 8.5))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.3], hspace=0.35, wspace=0.22)

        dt = getattr(self.env, "dt", 0.01)
        H_obs = result.observed_steps
        H_fut = result.future_steps
        T_total = H_obs + H_fut

        time_obs = np.arange(H_obs) * dt
        time_fut = np.arange(H_obs, T_total) * dt

        joint_names = [
            r"Shoulder Yaw $\theta_1$",
            r"Shoulder Pitch $\theta_2$",
            r"Elbow Pitch $\theta_3$",
        ]
        joint_units = ["rad", "rad", "rad"]

        gt_joints = None
        if ground_truth_joint_future is not None:
            gt_joints = np.asarray(ground_truth_joint_future)
        elif ground_truth_future is not None:
            gt_arr = np.asarray(ground_truth_future)
            if gt_arr.shape[-1] >= 6:
                gt_joints = gt_arr[:, :3]

        axes_joints = []
        for i in range(3):
            ax_j = fig.add_subplot(gs[i, 0], sharex=axes_joints[0] if axes_joints else None)
            axes_joints.append(ax_j)

            ax_j.plot(
                time_obs,
                result.observed_state[:, i],
                color="#2b5c8f",
                linewidth=2.4,
                label="Observed" if i == 0 else None,
            )

            if gt_joints is not None:
                ax_j.plot(
                    time_fut,
                    gt_joints[:H_fut, i],
                    color="#4f4f4f",
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.85,
                    label="True Future" if i == 0 else None,
                )

            ax_j.plot(
                time_fut,
                result.mean[:, i],
                color="#d95f02",
                linewidth=2.6,
                label="Predicted Mean" if i == 0 else None,
            )

            ax_j.fill_between(
                time_fut,
                result.lcl[:, i],
                result.ucl[:, i],
                color="#fdb863",
                alpha=0.35,
                label=f"{result.confidence_level*100:.0f}% Confidence Tube" if i == 0 else None,
            )
            ax_j.plot(
                time_fut,
                result.ucl[:, i],
                color="#e66101",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )
            ax_j.plot(
                time_fut,
                result.lcl[:, i],
                color="#e66101",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )

            ax_j.axvline(
                H_obs * dt,
                color="#2b5c8f",
                linestyle=":",
                linewidth=1.5,
                alpha=0.75,
                label="Handover" if i == 0 else None,
            )

            ax_j.set_ylabel(f"{joint_names[i]} [{joint_units[i]}]", fontsize=10, fontweight="bold")
            ax_j.grid(True, linestyle=":", alpha=0.5)
            if i == 0:
                ax_j.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
            if i == 2:
                ax_j.set_xlabel("Time [s]", fontsize=10, fontweight="bold")

        # ---------------------------------------------------------
        # Right Subplot: 3D Task Space Trajectory with Spatial Tube & Covariance Ellipsoids
        # ---------------------------------------------------------
        ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

        obs_cart = result.observed_cartesian
        ax_3d.plot(
            obs_cart[:, 0],
            obs_cart[:, 1],
            obs_cart[:, 2],
            color="#2b5c8f",
            linewidth=3.2,
            label=f"Observed Hand ({H_obs} steps)",
            zorder=6,
        )
        ax_3d.scatter(
            obs_cart[0, 0],
            obs_cart[0, 1],
            obs_cart[0, 2],
            color="black",
            s=65,
            marker="o",
            label="Start",
            zorder=8,
        )
        ax_3d.scatter(
            obs_cart[-1, 0],
            obs_cart[-1, 1],
            obs_cart[-1, 2],
            color="#2b5c8f",
            s=75,
            marker="o",
            label="Handover",
            zorder=8,
        )

        if ground_truth_future is not None:
            gt = np.asarray(ground_truth_future)
            if gt.shape[-1] >= 6:
                gt_cart = np.array(vmap(self.env.e)(gt))
            else:
                gt_cart = gt
            ax_3d.plot(
                gt_cart[:H_fut, 0],
                gt_cart[:H_fut, 1],
                gt_cart[:H_fut, 2],
                color="#4f4f4f",
                linestyle="--",
                linewidth=2.2,
                alpha=0.9,
                label=f"True Future ({H_fut} steps)",
                zorder=5,
            )

        if result.cartesian_samples is not None and len(result.cartesian_samples) > 0:
            samples = result.cartesian_samples
            for s in range(min(20, len(samples))):
                ax_3d.plot(
                    samples[s, :, 0],
                    samples[s, :, 1],
                    samples[s, :, 2],
                    color="#e66101",
                    alpha=0.15,
                    linewidth=0.8,
                    zorder=3,
                )

        mean = result.cartesian_mean
        cov = result.cartesian_cov
        z_crit = math.sqrt(2.0) * float(lax.erf_inv(result.confidence_level))

        handover_p = result.observed_cartesian[-1:]
        handover_cov = cov[:1] * 0.1
        full_mean = np.vstack([handover_p, mean])
        full_cov = np.vstack([handover_cov, cov])

        dt_vec = np.gradient(full_mean, axis=0)
        speed = np.linalg.norm(dt_vec, axis=-1, keepdims=True)
        speed = np.maximum(speed, 1e-6)
        tangents = dt_vec / speed

        ref = np.array([0.0, 0.0, 1.0])
        n1 = np.cross(tangents, ref)
        n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
        n1 = np.where(n1_norm < 1e-6, np.array([0.0, 1.0, 0.0]), n1 / np.maximum(n1_norm, 1e-6))
        n2 = np.cross(tangents, n1)

        n_angles = 24
        angles = np.linspace(0, 2 * np.pi, n_angles)
        tube_mesh = np.zeros((len(full_mean), n_angles, 3))
        for t_idx in range(len(full_mean)):
            for a_idx, ang in enumerate(angles):
                radial_dir = np.cos(ang) * n1[t_idx] + np.sin(ang) * n2[t_idx]
                sigma_r = np.sqrt(np.clip(radial_dir @ full_cov[t_idx] @ radial_dir, 1e-8, None))
                tube_mesh[t_idx, a_idx] = full_mean[t_idx] + z_crit * sigma_r * radial_dir

        ax_3d.plot_surface(
            tube_mesh[..., 0],
            tube_mesh[..., 1],
            tube_mesh[..., 2],
            color="#fdb863",
            alpha=0.25,
            edgecolor="none",
            shade=True,
            zorder=2,
        )

        guide_indices = [0, n_angles // 4, n_angles // 2, 3 * n_angles // 4]
        for g_idx, g_i in enumerate(guide_indices):
            ax_3d.plot(
                tube_mesh[:, g_i, 0],
                tube_mesh[:, g_i, 1],
                tube_mesh[:, g_i, 2],
                color="#fdb863",
                linestyle=":",
                linewidth=1.2,
                alpha=0.75,
                label=f"{result.confidence_level*100:.0f}% Spatial Tube Bounds" if g_idx == 0 else None,
                zorder=3,
            )

        u_ang = np.linspace(0, 2 * np.pi, 16)
        v_ang = np.linspace(0, np.pi, 8)
        unit_sphere = np.stack(
            [
                np.outer(np.cos(u_ang), np.sin(v_ang)),
                np.outer(np.sin(u_ang), np.sin(v_ang)),
                np.outer(np.ones_like(u_ang), np.cos(v_ang)),
            ]
        )

        ell_steps = np.linspace(1, len(full_mean) - 1, min(5, len(full_mean) - 1), dtype=int)
        for e_idx, idx in enumerate(ell_steps):
            val, vec = np.linalg.eigh(full_cov[idx])
            radii = z_crit * np.sqrt(np.clip(val, 1e-8, None))
            ell = (vec @ (unit_sphere * radii[:, None, None]).reshape(3, -1)).reshape(3, 16, 8)
            ax_3d.plot_wireframe(
                full_mean[idx, 0] + ell[0],
                full_mean[idx, 1] + ell[1],
                full_mean[idx, 2] + ell[2],
                color="#d95f02",
                alpha=0.45,
                linewidth=0.9,
                label=f"{result.confidence_level*100:.0f}% Covariance Ellipsoids" if e_idx == 0 else None,
                zorder=4,
            )

        ax_3d.plot(
            mean[:, 0],
            mean[:, 1],
            mean[:, 2],
            color="#d95f02",
            linewidth=3.4,
            label=f"Predicted Mean ({H_fut} steps)",
            zorder=7,
        )

        target = np.array(self.env.target)
        ax_3d.scatter(
            target[0],
            target[1],
            target[2],
            color="crimson",
            marker="X",
            s=120,
            linewidth=2.5,
            label=f"Target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})",
            zorder=9,
        )

        ax_3d.set_xlabel("X [m]", fontsize=10, fontweight="bold", labelpad=8)
        ax_3d.set_ylabel("Y [m]", fontsize=10, fontweight="bold", labelpad=8)
        ax_3d.set_zlabel("Z [m]", fontsize=10, fontweight="bold", labelpad=8)
        ax_3d.view_init(elev=24, azim=48)

        # Set focused axis limits around the reaching trajectory
        all_pts = np.vstack([obs_cart, full_mean])
        if ground_truth_future is not None:
            all_pts = np.vstack([all_pts, gt_cart])
        pad = 0.025
        ax_3d.set_xlim(np.min(all_pts[:, 0]) - pad, np.max(all_pts[:, 0]) + pad)
        ax_3d.set_ylim(np.min(all_pts[:, 1]) - pad, np.max(all_pts[:, 1]) + pad)
        ax_3d.set_zlim(np.min(all_pts[:, 2]) - pad, np.max(all_pts[:, 2]) + pad)

        ax_3d.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

        if ground_truth_future is not None:
            ade_val = result.ade(ground_truth_future)
            fde_val = result.fde(ground_truth_future)
            cov_val = result.coverage_rate(ground_truth_future)
            hz_val = 1000.0 / max(result.latency_ms, 1e-3) if result.latency_ms > 0 else 0.0
            text_box = (
                r"$\mathbf{3D\ Real-Time\ Metrics:}$" + "\n"
                f"Mode: {result.mode.upper()}\n"
                f"Latency: {result.latency_ms:.1f} ms ({hz_val:.1f} Hz)\n"
                f"ADE: {ade_val * 1000:.2f} mm\n"
                f"FDE: {fde_val * 1000:.2f} mm\n"
                f"95% 3D Cov: {cov_val * 100:.1f}%"
            )
            props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="#bbbbbb")
            ax_3d.text2D(
                0.03,
                0.97,
                text_box,
                transform=ax_3d.transAxes,
                verticalalignment="top",
                fontsize=8.5,
                bbox=props,
            )

        hz_label = f" | {1000/max(result.latency_ms, 1e-3):.1f} Hz" if result.latency_ms > 0 else ""
        plot_title = (
            title
            or f"3D Human Arm Motion Prediction (Strategy: {result.mode.upper()})\n"
               f"Observed: {result.observed_steps} steps | Future: {result.future_steps} steps | Online Latency: {result.latency_ms:.1f} ms{hz_label}"
        )
        fig.suptitle(plot_title, fontsize=12, fontweight="bold", y=0.98)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        if show:
            plt.show(block=True)

        return fig

    def plot_arm_kinematics_3d(
        self,
        result: PredictionResult,
        true_states: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualizes full human arm kinematic chain (Shoulder -> Elbow -> Hand) in 3D:
        - Arm postures at key milestones: Start (t=0), Handover (t=t_obs), Mid-prediction, and Final reach.
        - Continuous 3D spatial evolution of each keypoint.
        """
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")

        H_obs = result.observed_steps
        H_fut = result.future_steps
        obs_states = result.observed_state
        pred_states = result.mean

        def get_kp(s):
            if hasattr(self.env, "keypoints"):
                return [np.array(p) for p in self.env.keypoints(s)]
            else:
                ph = np.array(self.env.e(s))
                return [np.array([0.0, 0.0, 0.0]), np.array([ph[0]*0.5, ph[1]*0.5, 0.0]), np.array([ph[0], ph[1], 0.0])]

        obs_elbow = np.array([get_kp(s)[1] for s in obs_states])
        obs_hand = np.array([get_kp(s)[2] for s in obs_states])
        pred_elbow = np.array([get_kp(s)[1] for s in pred_states])
        pred_hand = np.array([get_kp(s)[2] for s in pred_states])

        # 1. Elbow path
        ax.plot(
            obs_elbow[:, 0], obs_elbow[:, 1], obs_elbow[:, 2],
            color="#41b6c4", linewidth=2.4, label="Elbow (Observed Path)", zorder=5
        )
        ax.plot(
            pred_elbow[:, 0], pred_elbow[:, 1], pred_elbow[:, 2],
            color="#253494", linestyle="--", linewidth=2.2, label="Elbow (Predicted Path)", zorder=5
        )

        # 2. Hand path
        ax.plot(
            obs_hand[:, 0], obs_hand[:, 1], obs_hand[:, 2],
            color="#2b5c8f", linewidth=3.2, label="Hand (Observed Path)", zorder=6
        )
        ax.plot(
            pred_hand[:, 0], pred_hand[:, 1], pred_hand[:, 2],
            color="#d95f02", linewidth=3.4, label="Hand (Predicted Mean Path)", zorder=6
        )

        # 3. True hand path if available
        if true_states is not None:
            ts = np.asarray(true_states)
            if ts.shape[-1] >= 6:
                true_hand = np.array([get_kp(s)[2] for s in ts])
            else:
                true_hand = ts
            ax.plot(
                true_hand[:, 0], true_hand[:, 1], true_hand[:, 2],
                color="#4f4f4f", linestyle=":", linewidth=2.0, alpha=0.85, label="Hand (True Full Path)", zorder=4
            )

        # 4. Shoulder fixed base
        ps_0 = get_kp(obs_states[0])[0]
        ax.scatter(ps_0[0], ps_0[1], ps_0[2], color="#333333", s=140, marker="o", label="Shoulder Joint (Fixed Base)", zorder=8)
        ax.plot([ps_0[0], ps_0[0]], [ps_0[1], ps_0[1]], [ps_0[2] - 0.15, ps_0[2]], color="#555555", linewidth=5.0, alpha=0.6)

        # 5. Arm Kinematic Chain Postures at 4 key milestones
        milestones = [
            (0, "Start (t=0)", obs_states[0], "#7f7f7f", 0.65, 3.2),
            (H_obs - 1, f"Handover (t={H_obs})", obs_states[-1], "#2b5c8f", 0.90, 4.2),
            (H_fut // 2, f"Mid-Prediction (t={H_obs + H_fut//2})", pred_states[H_fut // 2], "#fdb863", 0.85, 3.6),
            (H_fut - 1, f"Target Arrival (t={H_obs + H_fut})", pred_states[-1], "#2ca02c", 1.0, 4.5),
        ]

        for _, label, st, col, alp, lw in milestones:
            ps, pe, ph = get_kp(st)
            ax.plot(
                [ps[0], pe[0]], [ps[1], pe[1]], [ps[2], pe[2]],
                color=col, linewidth=lw, alpha=alp, solid_capstyle="round",
                label=f"Arm: {label}", zorder=7
            )
            ax.plot(
                [pe[0], ph[0]], [pe[1], ph[1]], [pe[2], ph[2]],
                color=col, linewidth=lw * 0.85, alpha=alp, solid_capstyle="round", zorder=7
            )
            ax.scatter([pe[0]], [pe[1]], [pe[2]], color=col, s=70, alpha=alp, zorder=8)
            ax.scatter([ph[0]], [ph[1]], [ph[2]], color=col, s=70, alpha=alp, zorder=8)

        # 6. Target marker
        target = np.array(self.env.target)
        ax.scatter(
            target[0], target[1], target[2],
            color="crimson", marker="X", s=130, linewidth=2.5,
            label=f"Target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})", zorder=9
        )

        ax.set_xlabel("X [m] (Sagittal / Forward)", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("Y [m] (Lateral)", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_zlabel("Z [m] (Vertical / Elevation)", fontsize=10, fontweight="bold", labelpad=8)
        ax.view_init(elev=22, azim=45)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

        l1 = getattr(self.env, "l1", 0.30)
        l2 = getattr(self.env, "l2", 0.33)
        final_err = np.linalg.norm(pred_hand[-1] - target) * 1000
        info_text = (
            r"$\mathbf{3D\ Arm\ Kinematic\ Chain:}$" + "\n"
            f"Upper Arm length: {l1*100:.1f} cm\n"
            f"Forearm length:   {l2*100:.1f} cm\n"
            f"Max arm reach:    {(l1+l2)*100:.1f} cm\n"
            f"Observed steps:   {H_obs}\n"
            f"Predicted steps:  {H_fut}\n"
            f"Final Hand Error: {final_err:.2f} mm"
        )
        props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="#bbbbbb")
        ax.text2D(
            0.03,
            0.97,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8.5,
            bbox=props,
        )

        hz_label = f" | {1000/max(result.latency_ms, 1e-3):.1f} Hz" if result.latency_ms > 0 else ""
        plot_title = (
            title
            or f"3D Human Arm Kinematic Chain & Multi-Keypoint Motion Evolution\n"
               f"Observed: {H_obs} steps | Future: {H_fut} steps | Latency: {result.latency_ms:.1f} ms{hz_label}"
        )
        ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=14)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        if show:
            plt.show(block=True)

        return fig

    def plot_prediction_3d_plotly(
        self,
        result: PredictionResult,
        ground_truth_future: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        ground_truth_joint_future: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        save_pdf: Optional[str] = None,
        save_png: Optional[str] = None,
    ) -> Any:
        """Generates a publication-grade interactive Plotly visualization for 3D motion prediction:
        - Left column: 3 subplots for joint angles (θ₁, θ₂, θ₃) with observed prefix, true future,
          predicted mean, shaded 95% confidence tube, and distinct handover markers.
        - Right column: Prominently zoomed-in 3D task-space scene with observed hand trajectory,
          true future, predicted mean, 95% spatial confidence tube surface, covariance ellipsoids,
          and start/handover/target markers.
        - Clean non-overlapping horizontal top toolbar legend.
        """
        if not _HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Install with 'pip install plotly kaleido'.")

        H_obs = result.observed_steps
        H_fut = result.future_steps
        dt = getattr(self.env, "dt", 0.01)
        time_obs = np.arange(H_obs) * dt
        time_fut = np.arange(H_obs, H_obs + H_fut) * dt

        joint_names = [
            "Shoulder Yaw (θ₁)",
            "Shoulder Pitch (θ₂)",
            "Elbow Pitch (θ₃)",
        ]

        gt_joints = None
        if ground_truth_joint_future is not None:
            gt_joints = np.asarray(ground_truth_joint_future)
        elif ground_truth_future is not None:
            gt_arr = np.asarray(ground_truth_future)
            if gt_arr.shape[-1] >= 6:
                gt_joints = gt_arr[:, :3]

        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.36, 0.64],
            specs=[
                [{"type": "xy"}, {"type": "scene", "rowspan": 3}],
                [{"type": "xy"}, None],
                [{"type": "xy"}, None]
            ],
            subplot_titles=[
                "<b>Shoulder Yaw (θ₁)</b>",
                "<b>3D Task-Space Trajectory & 95% Confidence Tube</b>",
                "<b>Shoulder Pitch (θ₂)</b>",
                "",
                "<b>Elbow Pitch (θ₃)</b>",
                ""
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )

        # -------------------------------------------------------------
        # Left Column: Joint Angle Subplots
        # -------------------------------------------------------------
        for i in range(3):
            # 1. Observed prefix
            fig.add_trace(
                go.Scatter(
                    x=time_obs,
                    y=result.observed_state[:, i],
                    mode="lines",
                    name="Observed Prefix (t ≤ 0.20s)" if i == 0 else None,
                    line=dict(color="#1e40af", width=2.8),
                    showlegend=(i == 0)
                ),
                row=i + 1, col=1
            )

            # 2. True future if available
            if gt_joints is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_fut,
                        y=gt_joints[:H_fut, i],
                        mode="lines",
                        name="True Future (Ground Truth)" if i == 0 else None,
                        line=dict(color="#374151", width=2.2, dash="dash"),
                        showlegend=(i == 0)
                    ),
                    row=i + 1, col=1
                )

            # 3. 95% Confidence Tube (Upper and Lower bounds with fill)
            fig.add_trace(
                go.Scatter(
                    x=time_fut,
                    y=result.ucl[:, i],
                    mode="lines",
                    line=dict(color="rgba(234, 88, 12, 0.4)", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=i + 1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_fut,
                    y=result.lcl[:, i],
                    mode="lines",
                    line=dict(color="rgba(234, 88, 12, 0.4)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(251, 146, 60, 0.28)",
                    name=f"95% Confidence Tube" if i == 0 else None,
                    showlegend=(i == 0)
                ),
                row=i + 1, col=1
            )

            # 4. Predicted Mean
            fig.add_trace(
                go.Scatter(
                    x=time_fut,
                    y=result.mean[:, i],
                    mode="lines",
                    name="Predicted Mean" if i == 0 else None,
                    line=dict(color="#ea580c", width=3.2),
                    showlegend=(i == 0)
                ),
                row=i + 1, col=1
            )

            # Handover vertical line
            handover_t = H_obs * dt
            fig.add_vline(
                x=handover_t,
                line_width=1.5,
                line_dash="dash",
                line_color="#2563eb",
                row=i + 1, col=1
            )

            fig.update_yaxes(
                title_text=f"<b>{joint_names[i]} [rad]</b>",
                row=i + 1, col=1,
                gridcolor="#f1f5f9",
                zerolinecolor="#e2e8f0"
            )
            fig.update_xaxes(gridcolor="#f1f5f9", row=i + 1, col=1)

        fig.update_xaxes(title_text="<b>Time [s]</b>", row=3, col=1, gridcolor="#f1f5f9")

        # -------------------------------------------------------------
        # Right Column: 3D Scene (Cartesian Motion, Tube, Ellipsoids)
        # -------------------------------------------------------------
        obs_cart = result.observed_cartesian
        mean = result.cartesian_mean
        cov = result.cartesian_cov
        z_crit = math.sqrt(2.0) * float(lax.erf_inv(result.confidence_level))

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[obs_cart[0, 0]], y=[obs_cart[0, 1]], z=[obs_cart[0, 2]],
                mode="markers",
                marker=dict(size=7, color="#1e293b", line=dict(color="white", width=1.5)),
                name="Start (t=0)"
            ),
            row=1, col=2
        )
        # Handover marker
        fig.add_trace(
            go.Scatter3d(
                x=[obs_cart[-1, 0]], y=[obs_cart[-1, 1]], z=[obs_cart[-1, 2]],
                mode="markers",
                marker=dict(size=8, color="#1d4ed8", line=dict(color="white", width=1.5)),
                name=f"Handover (t={H_obs})"
            ),
            row=1, col=2
        )

        # Observed Hand trajectory
        fig.add_trace(
            go.Scatter3d(
                x=obs_cart[:, 0], y=obs_cart[:, 1], z=obs_cart[:, 2],
                mode="lines",
                line=dict(color="#1e40af", width=6),
                showlegend=False
            ),
            row=1, col=2
        )

        # Ground truth future if available
        gt_cart = None
        if ground_truth_future is not None:
            gt = np.asarray(ground_truth_future)
            if gt.shape[-1] >= 6:
                gt_cart = np.array(vmap(self.env.e)(gt))
            else:
                gt_cart = gt
            fig.add_trace(
                go.Scatter3d(
                    x=gt_cart[:H_fut, 0], y=gt_cart[:H_fut, 1], z=gt_cart[:H_fut, 2],
                    mode="lines",
                    line=dict(color="#374151", width=4.5, dash="dash"),
                    showlegend=False
                ),
                row=1, col=2
            )

        # Predicted Mean Hand trajectory
        fig.add_trace(
            go.Scatter3d(
                x=mean[:, 0], y=mean[:, 1], z=mean[:, 2],
                mode="lines",
                line=dict(color="#ea580c", width=7),
                showlegend=False
            ),
            row=1, col=2
        )

        # Build 3D Confidence Tube mesh
        handover_p = result.observed_cartesian[-1:]
        handover_cov = cov[:1] * 0.1
        full_mean = np.vstack([handover_p, mean])
        full_cov = np.vstack([handover_cov, cov])

        dt_vec = np.gradient(full_mean, axis=0)
        speed = np.linalg.norm(dt_vec, axis=-1, keepdims=True)
        speed = np.maximum(speed, 1e-6)
        tangents = dt_vec / speed

        ref = np.array([0.0, 0.0, 1.0])
        n1 = np.cross(tangents, ref)
        n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
        n1 = np.where(n1_norm < 1e-6, np.array([0.0, 1.0, 0.0]), n1 / np.maximum(n1_norm, 1e-6))
        n2 = np.cross(tangents, n1)

        n_angles = 20
        angles = np.linspace(0, 2 * np.pi, n_angles)
        tube_x = np.zeros((len(full_mean), n_angles))
        tube_y = np.zeros((len(full_mean), n_angles))
        tube_z = np.zeros((len(full_mean), n_angles))

        for t_idx in range(len(full_mean)):
            for a_idx, ang in enumerate(angles):
                radial_dir = np.cos(ang) * n1[t_idx] + np.sin(ang) * n2[t_idx]
                sigma_r = np.sqrt(np.clip(radial_dir @ full_cov[t_idx] @ radial_dir, 1e-8, None))
                pt = full_mean[t_idx] + z_crit * sigma_r * radial_dir
                tube_x[t_idx, a_idx] = pt[0]
                tube_y[t_idx, a_idx] = pt[1]
                tube_z[t_idx, a_idx] = pt[2]

        fig.add_trace(
            go.Surface(
                x=tube_x,
                y=tube_y,
                z=tube_z,
                opacity=0.32,
                colorscale=[[0, "#fb923c"], [1, "#fb923c"]],
                showscale=False,
                hoverinfo="skip"
            ),
            row=1, col=2
        )

        # Add Covariance Ellipsoids (wireframe orthogonal rings)
        ell_steps = np.linspace(1, len(full_mean) - 1, min(4, len(full_mean) - 1), dtype=int)
        ring_theta = np.linspace(0, 2 * np.pi, 24)
        for e_idx, idx in enumerate(ell_steps):
            val, vec = np.linalg.eigh(full_cov[idx])
            radii = z_crit * np.sqrt(np.clip(val, 1e-8, None))
            center = full_mean[idx]

            # Ring in principal plane 0-1
            r_xy = (vec @ np.vstack([radii[0] * np.cos(ring_theta), radii[1] * np.sin(ring_theta), np.zeros_like(ring_theta)])).T + center
            # Ring in principal plane 0-2
            r_xz = (vec @ np.vstack([radii[0] * np.cos(ring_theta), np.zeros_like(ring_theta), radii[2] * np.sin(ring_theta)])).T + center

            fig.add_trace(
                go.Scatter3d(
                    x=r_xy[:, 0], y=r_xy[:, 1], z=r_xy[:, 2],
                    mode="lines",
                    line=dict(color="#c2410c", width=2.2),
                    name="95% Covariance Ellipsoids" if e_idx == 0 else None,
                    showlegend=(e_idx == 0),
                    hoverinfo="skip"
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter3d(
                    x=r_xz[:, 0], y=r_xz[:, 1], z=r_xz[:, 2],
                    mode="lines",
                    line=dict(color="#c2410c", width=2.2),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=1, col=2
            )

        # Target marker
        target = np.array(self.env.target)
        fig.add_trace(
            go.Scatter3d(
                x=[target[0]], y=[target[1]], z=[target[2]],
                mode="markers",
                marker=dict(size=10, color="#dc2626", symbol="diamond", line=dict(color="white", width=1.5)),
                name=f"Target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
            ),
            row=1, col=2
        )

        # Tightly cropped 3D axis limits
        all_pts = np.vstack([
            obs_cart,
            full_mean,
            np.column_stack([tube_x.ravel(), tube_y.ravel(), tube_z.ravel()]),
            target[None, :]
        ])
        if gt_cart is not None:
            all_pts = np.vstack([all_pts, gt_cart])
        pad = 0.008
        xmin, xmax = float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad)
        ymin, ymax = float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad)
        zmin, zmax = float(np.min(all_pts[:, 2]) - pad), float(np.max(all_pts[:, 2]) + pad)

        hz_val = 1000.0 / max(result.latency_ms, 1e-3) if result.latency_ms > 0 else 0.0
        ade_val = result.ade(ground_truth_future) * 1000.0 if ground_truth_future is not None else 0.0
        fde_val = result.fde(ground_truth_future) * 1000.0 if ground_truth_future is not None else 0.0
        cov_val = result.coverage_rate(ground_truth_future) * 100.0 if ground_truth_future is not None else 100.0

        metrics_sub = (
            f"Observed: {H_obs} steps (0.20s) | Forecast: {H_fut} steps (0.30s) | "
            f"ADE: {ade_val:.2f} mm | FDE: {fde_val:.2f} mm | 95% Coverage: {cov_val:.1f}% | "
            f"Online Latency: {result.latency_ms:.1f} ms ({hz_val:.1f} Hz)"
        )
        main_title = title or f"<b>3D Motion Prediction: Joint Evolutions & Task-Space Confidence Tube ({result.mode.upper()})</b><br><sup>{metrics_sub}</sup>"

        fig.update_layout(
            title=dict(
                text=main_title,
                font=dict(size=14, family="sans-serif"),
                x=0.03, y=0.975,
                xanchor="left", yanchor="top"
            ),
            legend=dict(
                orientation="h",
                yref="container",
                y=0.905,
                x=0.5,
                xanchor="center",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                font=dict(size=10, family="sans-serif")
            ),
            scene=dict(
                xaxis=dict(range=[xmin, xmax], title=dict(text="<b>X [m] (Sagittal)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                yaxis=dict(range=[ymin, ymax], title=dict(text="<b>Y [m] (Lateral)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                zaxis=dict(range=[zmin, zmax], title=dict(text="<b>Z [m] (Elevation)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.15, y=1.05, z=0.75),
                    center=dict(x=0, y=0, z=0),
                ),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=130, b=40),
            width=1450,
            height=820,
        )

        if save_html:
            os.makedirs(os.path.dirname(save_html) or ".", exist_ok=True)
            fig.write_html(save_html)
            print(f"Saved interactive HTML to {save_html}")

        if save_pdf:
            os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
            fig.write_image(save_pdf, width=1450, height=820)
            print(f"Saved vector PDF to {save_pdf}")

        if save_png:
            os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
            fig.write_image(save_png, width=1450, height=820, scale=2)
            print(f"Saved high-res PNG to {save_png}")

        return fig

    def plot_arm_kinematics_3d_plotly(
        self,
        result: PredictionResult,
        true_states: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        save_pdf: Optional[str] = None,
        save_png: Optional[str] = None,
    ) -> Any:
        """Generates a publication-grade interactive Plotly visualization for the full human arm kinematic chain:
        - Postures at 3 key milestones: Start (t=0), Handover (t=t_obs), and Target Arrival (t=T).
        - Multi-keypoint spatial trajectories:
            - Elbow: Observed path, True continuous motion, Predicted future path.
            - Hand/Wrist: Observed path, True continuous motion, Predicted mean path.
        - Fixed shoulder base and vertical torso pillar.
        - Tightly cropped, prominently zoomed 3D camera.
        """
        if not _HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Install with 'pip install plotly kaleido'.")

        H_obs = result.observed_steps
        H_fut = result.future_steps
        obs_states = result.observed_state
        pred_states = result.mean

        def get_kp(s):
            if hasattr(self.env, "keypoints"):
                return [np.array(p) for p in self.env.keypoints(s)]
            else:
                ph = np.array(self.env.e(s))
                return [np.array([0.0, 0.0, 0.0]), np.array([ph[0]*0.5, ph[1]*0.5, 0.0]), np.array([ph[0], ph[1], 0.0])]

        obs_elbow = np.array([get_kp(s)[1] for s in obs_states])
        obs_hand = np.array([get_kp(s)[2] for s in obs_states])
        pred_elbow = np.array([get_kp(s)[1] for s in pred_states])
        pred_hand = np.array([get_kp(s)[2] for s in pred_states])

        true_elbow = None
        true_hand = None
        if true_states is not None:
            ts = np.asarray(true_states)
            if ts.shape[-1] >= 6:
                true_elbow = np.array([get_kp(s)[1] for s in ts])
                true_hand = np.array([get_kp(s)[2] for s in ts])
            else:
                true_hand = ts

        fig = go.Figure()

        # 1. Shoulder fixed base & torso pillar
        ps_0 = get_kp(obs_states[0])[0]
        fig.add_trace(go.Scatter3d(
            x=[ps_0[0]], y=[ps_0[1]], z=[ps_0[2]],
            mode="markers",
            marker=dict(size=8, color="#0f172a", line=dict(color="white", width=1.5)),
            name="Shoulder Joint (Fixed Base)"
        ))
        fig.add_trace(go.Scatter3d(
            x=[ps_0[0], ps_0[0]], y=[ps_0[1], ps_0[1]], z=[ps_0[2] - 0.15, ps_0[2]],
            mode="lines",
            line=dict(color="#64748b", width=8),
            name="Torso Pillar",
            showlegend=False,
            hoverinfo="skip"
        ))

        # 2. Keypoint trajectories: Elbow
        fig.add_trace(go.Scatter3d(
            x=obs_elbow[:, 0], y=obs_elbow[:, 1], z=obs_elbow[:, 2],
            mode="lines",
            line=dict(color="#06b6d4", width=4.5),
            name="Elbow: Observed Path"
        ))
        if true_elbow is not None:
            fig.add_trace(go.Scatter3d(
                x=true_elbow[:, 0], y=true_elbow[:, 1], z=true_elbow[:, 2],
                mode="lines",
                line=dict(color="#0f766e", width=3.5, dash="dash"),
                name="Elbow: True Motion"
            ))
        fig.add_trace(go.Scatter3d(
            x=pred_elbow[:, 0], y=pred_elbow[:, 1], z=pred_elbow[:, 2],
            mode="lines",
            line=dict(color="#1e3a8a", width=3.5, dash="dot"),
            name="Elbow: Predicted Future"
        ))

        # 3. Keypoint trajectories: Hand
        fig.add_trace(go.Scatter3d(
            x=obs_hand[:, 0], y=obs_hand[:, 1], z=obs_hand[:, 2],
            mode="lines",
            line=dict(color="#1d4ed8", width=5.5),
            name="Hand: Observed Path"
        ))
        if true_hand is not None:
            fig.add_trace(go.Scatter3d(
                x=true_hand[:, 0], y=true_hand[:, 1], z=true_hand[:, 2],
                mode="lines",
                line=dict(color="#374151", width=4, dash="dash"),
                name="Hand: True Motion"
            ))
        fig.add_trace(go.Scatter3d(
            x=pred_hand[:, 0], y=pred_hand[:, 1], z=pred_hand[:, 2],
            mode="lines",
            line=dict(color="#ea580c", width=6),
            name="Hand: Predicted Mean"
        ))

        # 4. Arm Kinematic Chain Postures at 3 key milestones (Mid-prediction removed)
        milestones = [
            (0, "Start (t=0)", obs_states[0], "#94a3b8", 6),
            (H_obs - 1, f"Handover (t={H_obs})", obs_states[-1], "#2563eb", 7),
            (H_fut - 1, f"Target Arrival (t={H_obs + H_fut})", pred_states[-1], "#16a34a", 8),
        ]

        for _, label, st, col, lw in milestones:
            ps, pe, ph = get_kp(st)
            fig.add_trace(go.Scatter3d(
                x=[ps[0], pe[0], ph[0]],
                y=[ps[1], pe[1], ph[1]],
                z=[ps[2], pe[2], ph[2]],
                mode="lines+markers",
                line=dict(color=col, width=lw),
                marker=dict(size=7, color=col, line=dict(color="white", width=1.2)),
                name=f"Arm Posture: {label}"
            ))

        # 5. Target marker
        target = np.array(self.env.target)
        fig.add_trace(go.Scatter3d(
            x=[target[0]], y=[target[1]], z=[target[2]],
            mode="markers",
            marker=dict(size=10, color="#dc2626", symbol="diamond", line=dict(color="white", width=1.5)),
            name=f"Target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
        ))

        # 6. Focused Axis Limits around the Arm Mechanism
        all_arm_pts = np.vstack([
            obs_elbow, pred_elbow, obs_hand, pred_hand,
            [ps_0[0], ps_0[1], ps_0[2]],
            [ps_0[0], ps_0[1], ps_0[2] - 0.15],
            target[None, :]
        ])
        if true_elbow is not None:
            all_arm_pts = np.vstack([all_arm_pts, true_elbow, true_hand])
        pad = 0.035
        xmin, xmax = float(np.min(all_arm_pts[:, 0]) - pad), float(np.max(all_arm_pts[:, 0]) + pad)
        ymin, ymax = float(np.min(all_arm_pts[:, 1]) - pad), float(np.max(all_arm_pts[:, 1]) + pad)
        zmin, zmax = float(np.min(all_arm_pts[:, 2]) - pad), float(np.max(all_arm_pts[:, 2]) + pad)

        l1 = getattr(self.env, "l1", 0.30)
        l2 = getattr(self.env, "l2", 0.33)
        final_err = np.linalg.norm(pred_hand[-1] - target) * 1000
        hz_label = f" | {1000/max(result.latency_ms, 1e-3):.1f} Hz" if result.latency_ms > 0 else ""
        metrics_sub = (
            f"Upper Arm: {l1*100:.1f} cm | Forearm: {l2*100:.1f} cm | Max Reach: {(l1+l2)*100:.1f} cm | "
            f"Observed: {H_obs} steps | Future: {H_fut} steps | Endpoint Error: {final_err:.2f} mm | Latency: {result.latency_ms:.1f} ms{hz_label}"
        )
        main_title = title or f"<b>3D Human Arm Kinematic Chain & Multi-Keypoint Motion Evolution</b><br><sup>{metrics_sub}</sup>"

        fig.update_layout(
            title=dict(
                text=main_title,
                font=dict(size=14, family="sans-serif"),
                x=0.03, y=0.975,
                xanchor="left", yanchor="top"
            ),
            legend=dict(
                orientation="h",
                yref="container",
                y=0.895,
                x=0.5,
                xanchor="center",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                font=dict(size=10, family="sans-serif")
            ),
            scene=dict(
                xaxis=dict(range=[xmin, xmax], title=dict(text="<b>X [m] (Sagittal)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                yaxis=dict(range=[ymin, ymax], title=dict(text="<b>Y [m] (Lateral)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                zaxis=dict(range=[zmin, zmax], title=dict(text="<b>Z [m] (Elevation)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.50, y=1.35, z=0.80),
                    center=dict(x=0, y=0, z=-0.05),
                ),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=140, b=40),
            width=1350,
            height=850,
        )

        if save_html:
            os.makedirs(os.path.dirname(save_html) or ".", exist_ok=True)
            fig.write_html(save_html)
            print(f"Saved interactive HTML to {save_html}")

        if save_pdf:
            os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
            fig.write_image(save_pdf, width=1250, height=820)
            print(f"Saved vector PDF to {save_pdf}")

        if save_png:
            os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
            fig.write_image(save_png, width=1250, height=820, scale=2)
            print(f"Saved high-res PNG to {save_png}")

        return fig

    def plot_keypoint_3d_plotly(
        self,
        result: PredictionResult,
        keypoint: str = "hand",
        ground_truth_future: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        ground_truth_obs: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        save_pdf: Optional[str] = None,
        save_png: Optional[str] = None,
    ) -> Any:
        """Generates a dedicated, publication-grade interactive 3D visualization for a single keypoint (Hand or Elbow).
        Features tightly cropped spatial bounds, 3D confidence tube / cone, and 95% covariance ellipsoids.
        """
        if not _HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Install with 'pip install plotly kaleido'.")

        kp = keypoint.lower().strip()
        H_obs = result.observed_steps
        H_fut = result.future_steps
        z_crit = math.sqrt(2.0) * float(lax.erf_inv(result.confidence_level))

        if kp == "elbow":
            kp_label = "Elbow Keypoint"
            obs_pts = result.observed_elbow
            mean_pts = result.elbow_mean
            cov_pts = result.elbow_cov
            col_obs = "#06b6d4"     # Cyan
            col_pred = "#0284c7"    # Sky Blue
            col_gt = "#0f766e"      # Teal
            col_tube = "#38bdf8"    # Sky blue translucent
            col_ell = "#0369a1"     # Deep ocean blue
            ade_val = result.ade_elbow(ground_truth_future) * 1000.0 if ground_truth_future is not None else None
            fde_val = result.fde_elbow(ground_truth_future) * 1000.0 if ground_truth_future is not None else None
            cov_val = result.coverage_rate_elbow(ground_truth_future) * 100.0 if ground_truth_future is not None else None
        else:
            kp_label = "Hand / End-Effector"
            obs_pts = result.observed_cartesian
            mean_pts = result.cartesian_mean
            cov_pts = result.cartesian_cov
            col_obs = "#1e40af"     # Dark Blue
            col_pred = "#ea580c"    # Orange
            col_gt = "#374151"      # Dark Slate
            col_tube = "#fb923c"    # Light Orange translucent
            col_ell = "#c2410c"     # Deep Orange
            ade_val = result.ade(ground_truth_future) * 1000.0 if ground_truth_future is not None else None
            fde_val = result.fde(ground_truth_future) * 1000.0 if ground_truth_future is not None else None
            cov_val = result.coverage_rate(ground_truth_future) * 100.0 if ground_truth_future is not None else None

        if obs_pts is None and ground_truth_obs is not None:
            obs_pts = np.asarray(ground_truth_obs)
        if mean_pts is None:
            raise ValueError(f"No prediction data found for keypoint '{keypoint}'.")

        fig = go.Figure()

        # 1. Start & Handover markers
        if obs_pts is not None and len(obs_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=[obs_pts[0, 0]], y=[obs_pts[0, 1]], z=[obs_pts[0, 2]],
                mode="markers",
                marker=dict(size=7, color="#0f172a", line=dict(color="white", width=1.5)),
                name="Start (t=0)"
            ))
            fig.add_trace(go.Scatter3d(
                x=[obs_pts[-1, 0]], y=[obs_pts[-1, 1]], z=[obs_pts[-1, 2]],
                mode="markers",
                marker=dict(size=8, color="#2563eb", line=dict(color="white", width=1.5)),
                name=f"Handover (t={H_obs})"
            ))
            fig.add_trace(go.Scatter3d(
                x=obs_pts[:, 0], y=obs_pts[:, 1], z=obs_pts[:, 2],
                mode="lines",
                line=dict(color=col_obs, width=6),
                name=f"Observed {kp_label}"
            ))

        # 2. Ground truth future
        gt_pts = None
        if ground_truth_future is not None:
            gt_pts = np.asarray(ground_truth_future)
            if gt_pts.shape[-1] >= 6 and hasattr(self.env, "keypoints"):
                kp_idx = 1 if kp == "elbow" else 2
                gt_pts = np.array([self.env.keypoints(s)[kp_idx] for s in gt_pts])
            elif gt_pts.shape[-1] >= 3:
                gt_pts = gt_pts[:, :3]
            fig.add_trace(go.Scatter3d(
                x=gt_pts[:H_fut, 0], y=gt_pts[:H_fut, 1], z=gt_pts[:H_fut, 2],
                mode="lines",
                line=dict(color=col_gt, width=4, dash="dash"),
                name=f"True {kp_label} Motion"
            ))

        # 3. 3D Spatial Confidence Tube / Cone
        if cov_pts is not None:
            handover_p = obs_pts[-1:] if obs_pts is not None else mean_pts[:1]
            handover_cov = cov_pts[:1] * 0.1
            full_mean = np.vstack([handover_p, mean_pts])
            full_cov = np.vstack([handover_cov, cov_pts])

            dt_vec = np.gradient(full_mean, axis=0)
            speed = np.linalg.norm(dt_vec, axis=-1, keepdims=True)
            speed = np.maximum(speed, 1e-6)
            tangents = dt_vec / speed

            ref = np.array([0.0, 0.0, 1.0])
            n1 = np.cross(tangents, ref)
            n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
            n1 = np.where(n1_norm < 1e-6, np.array([0.0, 1.0, 0.0]), n1 / np.maximum(n1_norm, 1e-6))
            n2 = np.cross(tangents, n1)

            n_angles = 20
            angles = np.linspace(0, 2 * np.pi, n_angles)
            tube_x = np.zeros((len(full_mean), n_angles))
            tube_y = np.zeros((len(full_mean), n_angles))
            tube_z = np.zeros((len(full_mean), n_angles))

            for t_idx in range(len(full_mean)):
                for a_idx, ang in enumerate(angles):
                    radial_dir = np.cos(ang) * n1[t_idx] + np.sin(ang) * n2[t_idx]
                    sigma_r = np.sqrt(np.clip(radial_dir @ full_cov[t_idx] @ radial_dir, 1e-8, None))
                    pt = full_mean[t_idx] + z_crit * sigma_r * radial_dir
                    tube_x[t_idx, a_idx] = pt[0]
                    tube_y[t_idx, a_idx] = pt[1]
                    tube_z[t_idx, a_idx] = pt[2]

            fig.add_trace(go.Surface(
                x=tube_x, y=tube_y, z=tube_z,
                opacity=0.28,
                colorscale=[[0, col_tube], [1, col_tube]],
                showscale=False,
                hoverinfo="skip"
            ))

            # 95% Covariance Ellipsoids
            ell_steps = np.linspace(1, len(full_mean) - 1, min(4, len(full_mean) - 1), dtype=int)
            ring_theta = np.linspace(0, 2 * np.pi, 24)
            for e_idx, idx in enumerate(ell_steps):
                val, vec = np.linalg.eigh(full_cov[idx])
                radii = z_crit * np.sqrt(np.clip(val, 1e-8, None))
                center = full_mean[idx]
                r_xy = (vec @ np.vstack([radii[0] * np.cos(ring_theta), radii[1] * np.sin(ring_theta), np.zeros_like(ring_theta)])).T + center
                r_xz = (vec @ np.vstack([radii[0] * np.cos(ring_theta), np.zeros_like(ring_theta), radii[2] * np.sin(ring_theta)])).T + center

                fig.add_trace(go.Scatter3d(
                    x=r_xy[:, 0], y=r_xy[:, 1], z=r_xy[:, 2],
                    mode="lines", line=dict(color=col_ell, width=2.2),
                    name="95% Covariance Ellipsoids" if e_idx == 0 else None,
                    showlegend=(e_idx == 0), hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter3d(
                    x=r_xz[:, 0], y=r_xz[:, 1], z=r_xz[:, 2],
                    mode="lines", line=dict(color=col_ell, width=2.2),
                    showlegend=False, hoverinfo="skip"
                ))
        else:
            tube_x = tube_y = tube_z = None

        # 4. Predicted Mean Trajectory
        fig.add_trace(go.Scatter3d(
            x=mean_pts[:, 0], y=mean_pts[:, 1], z=mean_pts[:, 2],
            mode="lines",
            line=dict(color=col_pred, width=6.5),
            name=f"Predicted {kp_label} Mean"
        ))

        # 5. Target marker (for hand or elbow if available)
        if kp == "hand" and hasattr(self.env, "target"):
            tgt = np.array(self.env.target)
            fig.add_trace(go.Scatter3d(
                x=[tgt[0]], y=[tgt[1]], z=[tgt[2]],
                mode="markers",
                marker=dict(size=10, color="#dc2626", symbol="diamond", line=dict(color="white", width=1.5)),
                name=f"Target ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f})"
            ))
        elif kp == "elbow":
            tgt_e = getattr(self.env, "target_elbow", None)
            if tgt_e is not None:
                tgt_e = np.array(tgt_e)
                fig.add_trace(go.Scatter3d(
                    x=[tgt_e[0]], y=[tgt_e[1]], z=[tgt_e[2]],
                    mode="markers",
                    marker=dict(size=9, color="#0f766e", symbol="diamond", line=dict(color="white", width=1.5)),
                    name=f"Target Elbow ({tgt_e[0]:.2f}, {tgt_e[1]:.2f}, {tgt_e[2]:.2f})"
                ))

        # 6. Focused Axis Limits strictly around this keypoint
        pts_list = [mean_pts]
        if obs_pts is not None:
            pts_list.append(obs_pts)
        if gt_pts is not None:
            pts_list.append(gt_pts)
        if tube_x is not None:
            pts_list.append(np.column_stack([tube_x.ravel(), tube_y.ravel(), tube_z.ravel()]))
        all_pts = np.vstack(pts_list)
        pad = 0.012
        xmin, xmax = float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad)
        ymin, ymax = float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad)
        zmin, zmax = float(np.min(all_pts[:, 2]) - pad), float(np.max(all_pts[:, 2]) + pad)

        hz_str = f" | {1000/max(result.latency_ms, 1e-3):.1f} Hz" if result.latency_ms > 0 else ""
        metrics_parts = [f"Observed: {H_obs} steps", f"Future: {H_fut} steps"]
        if ade_val is not None:
            metrics_parts.append(f"ADE: {ade_val:.2f} mm")
        if fde_val is not None:
            metrics_parts.append(f"FDE: {fde_val:.2f} mm")
        if cov_val is not None:
            metrics_parts.append(f"95% Coverage: {cov_val:.1f}%")
        metrics_parts.append(f"Latency: {result.latency_ms:.1f} ms{hz_str}")
        sub_str = " | ".join(metrics_parts)

        main_title = title or f"<b>3D Motion Prediction: {kp_label} Trajectory & 95% Confidence Tube</b><br><sup>{sub_str}</sup>"

        fig.update_layout(
            title=dict(
                text=main_title,
                font=dict(size=14, family="sans-serif"),
                x=0.03, y=0.975,
                xanchor="left", yanchor="top"
            ),
            legend=dict(
                orientation="h",
                yref="container",
                y=0.91,
                x=0.5,
                xanchor="center",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                font=dict(size=10, family="sans-serif")
            ),
            scene=dict(
                xaxis=dict(range=[xmin, xmax], title=dict(text="<b>X [m] (Sagittal)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                yaxis=dict(range=[ymin, ymax], title=dict(text="<b>Y [m] (Lateral)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                zaxis=dict(range=[zmin, zmax], title=dict(text="<b>Z [m] (Elevation)</b>", font=dict(size=11)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.35, y=1.25, z=0.75),
                    center=dict(x=0, y=0, z=0),
                ),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=140, b=40),
            width=1250,
            height=820,
        )

        if save_html:
            os.makedirs(os.path.dirname(save_html) or ".", exist_ok=True)
            fig.write_html(save_html)
            print(f"Saved interactive HTML to {save_html}")

        if save_pdf:
            os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
            fig.write_image(save_pdf, width=1250, height=820)
            print(f"Saved vector PDF to {save_pdf}")

        if save_png:
            os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
            fig.write_image(save_png, width=1250, height=820, scale=2)
            print(f"Saved high-res PNG to {save_png}")

        return fig

    def plot_hand_3d_plotly(self, result: PredictionResult, **kwargs) -> Any:
        """Dedicated plot for Hand keypoint trajectory and 95% confidence tube."""
        return self.plot_keypoint_3d_plotly(result, keypoint="hand", **kwargs)

    def plot_elbow_3d_plotly(self, result: PredictionResult, **kwargs) -> Any:
        """Dedicated plot for Elbow keypoint trajectory and 95% confidence tube."""
        return self.plot_keypoint_3d_plotly(result, keypoint="elbow", **kwargs)

    def plot_separated_keypoints_3d_plotly(
        self,
        result: PredictionResult,
        ground_truth_future_hand: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        ground_truth_future_elbow: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        save_pdf: Optional[str] = None,
        save_png: Optional[str] = None,
    ) -> Any:
        """Generates a side-by-side 2-panel Plotly visualization separating Hand and Elbow keypoints into
        independent 3D subplots with dedicated close-up zoom, 3D confidence tubes, and covariance ellipsoids.
        """
        if not _HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Install with 'pip install plotly kaleido'.")

        ade_h = result.ade(ground_truth_future_hand) * 1000.0 if ground_truth_future_hand is not None else 0.0
        ade_e = result.ade_elbow(ground_truth_future_elbow) * 1000.0 if ground_truth_future_elbow is not None else 0.0

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[
                f"<b>Hand / End-Effector Trajectory & 95% Confidence Tube</b><br>ADE: {ade_h:.2f} mm",
                f"<b>Elbow Keypoint Trajectory & 95% Confidence Tube</b><br>ADE: {ade_e:.2f} mm"
            ],
            horizontal_spacing=0.04
        )

        z_crit = math.sqrt(2.0) * float(lax.erf_inv(result.confidence_level))

        # Helper to construct tube and ellipsoids
        def add_kp_to_subplot(obs, mean, cov, col_obs, col_pred, col_tube, col_ell, row, col, name_prefix):
            if obs is not None and len(obs) > 0:
                fig.add_trace(go.Scatter3d(
                    x=[obs[0, 0]], y=[obs[0, 1]], z=[obs[0, 2]],
                    mode="markers", marker=dict(size=6, color="#0f172a", line=dict(color="white", width=1.2)),
                    name=f"{name_prefix} Start (t=0)"
                ), row=row, col=col)
                fig.add_trace(go.Scatter3d(
                    x=[obs[-1, 0]], y=[obs[-1, 1]], z=[obs[-1, 2]],
                    mode="markers", marker=dict(size=7, color="#2563eb", line=dict(color="white", width=1.2)),
                    name=f"{name_prefix} Handover"
                ), row=row, col=col)
                fig.add_trace(go.Scatter3d(
                    x=obs[:, 0], y=obs[:, 1], z=obs[:, 2],
                    mode="lines", line=dict(color=col_obs, width=5.5),
                    name=f"{name_prefix} Observed Path"
                ), row=row, col=col)

            # Tube
            handover_p = obs[-1:] if obs is not None else mean[:1]
            handover_cov = cov[:1] * 0.1
            full_mean = np.vstack([handover_p, mean])
            full_cov = np.vstack([handover_cov, cov])

            dt_vec = np.gradient(full_mean, axis=0)
            speed = np.maximum(np.linalg.norm(dt_vec, axis=-1, keepdims=True), 1e-6)
            tangents = dt_vec / speed
            ref = np.array([0.0, 0.0, 1.0])
            n1 = np.cross(tangents, ref)
            n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
            n1 = np.where(n1_norm < 1e-6, np.array([0.0, 1.0, 0.0]), n1 / np.maximum(n1_norm, 1e-6))
            n2 = np.cross(tangents, n1)

            n_angles = 20
            angles = np.linspace(0, 2 * np.pi, n_angles)
            tx = np.zeros((len(full_mean), n_angles))
            ty = np.zeros((len(full_mean), n_angles))
            tz = np.zeros((len(full_mean), n_angles))

            for t_idx in range(len(full_mean)):
                for a_idx, ang in enumerate(angles):
                    radial_dir = np.cos(ang) * n1[t_idx] + np.sin(ang) * n2[t_idx]
                    sigma_r = np.sqrt(np.clip(radial_dir @ full_cov[t_idx] @ radial_dir, 1e-8, None))
                    pt = full_mean[t_idx] + z_crit * sigma_r * radial_dir
                    tx[t_idx, a_idx] = pt[0]
                    ty[t_idx, a_idx] = pt[1]
                    tz[t_idx, a_idx] = pt[2]

            fig.add_trace(go.Surface(
                x=tx, y=ty, z=tz, opacity=0.28,
                colorscale=[[0, col_tube], [1, col_tube]],
                showscale=False, hoverinfo="skip"
            ), row=row, col=col)

            # Ellipsoids
            ell_steps = np.linspace(1, len(full_mean) - 1, min(4, len(full_mean) - 1), dtype=int)
            ring_theta = np.linspace(0, 2 * np.pi, 24)
            for e_idx, idx in enumerate(ell_steps):
                val, vec = np.linalg.eigh(full_cov[idx])
                radii = z_crit * np.sqrt(np.clip(val, 1e-8, None))
                center = full_mean[idx]
                r_xy = (vec @ np.vstack([radii[0] * np.cos(ring_theta), radii[1] * np.sin(ring_theta), np.zeros_like(ring_theta)])).T + center
                r_xz = (vec @ np.vstack([radii[0] * np.cos(ring_theta), np.zeros_like(ring_theta), radii[2] * np.sin(ring_theta)])).T + center
                fig.add_trace(go.Scatter3d(
                    x=r_xy[:, 0], y=r_xy[:, 1], z=r_xy[:, 2],
                    mode="lines", line=dict(color=col_ell, width=2.0),
                    name=f"{name_prefix} 95% Covariance Ellipsoids" if e_idx == 0 else None,
                    showlegend=(e_idx == 0), hoverinfo="skip"
                ), row=row, col=col)
                fig.add_trace(go.Scatter3d(
                    x=r_xz[:, 0], y=r_xz[:, 1], z=r_xz[:, 2],
                    mode="lines", line=dict(color=col_ell, width=2.0),
                    showlegend=False, hoverinfo="skip"
                ), row=row, col=col)

            # Mean
            fig.add_trace(go.Scatter3d(
                x=mean[:, 0], y=mean[:, 1], z=mean[:, 2],
                mode="lines", line=dict(color=col_pred, width=6.5),
                name=f"{name_prefix} Predicted Mean"
            ), row=row, col=col)

            # Bounds
            pts = [mean]
            if obs is not None:
                pts.append(obs)
            pts.append(np.column_stack([tx.ravel(), ty.ravel(), tz.ravel()]))
            all_p = np.vstack(pts)
            pad = 0.012
            return (
                float(np.min(all_p[:, 0]) - pad), float(np.max(all_p[:, 0]) + pad),
                float(np.min(all_p[:, 1]) - pad), float(np.max(all_p[:, 1]) + pad),
                float(np.min(all_p[:, 2]) - pad), float(np.max(all_p[:, 2]) + pad)
            )

        # 1. Subplot 1: Hand
        h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax = add_kp_to_subplot(
            result.observed_cartesian, result.cartesian_mean, result.cartesian_cov,
            col_obs="#1e40af", col_pred="#ea580c", col_tube="#fb923c", col_ell="#c2410c",
            row=1, col=1, name_prefix="Hand"
        )
        if ground_truth_future_hand is not None:
            gt_h = np.asarray(ground_truth_future_hand)
            fig.add_trace(go.Scatter3d(
                x=gt_h[:, 0], y=gt_h[:, 1], z=gt_h[:, 2],
                mode="lines", line=dict(color="#374151", width=4, dash="dash"),
                name="Hand True Motion"
            ), row=1, col=1)
        if hasattr(self.env, "target"):
            tgt = np.array(self.env.target)
            fig.add_trace(go.Scatter3d(
                x=[tgt[0]], y=[tgt[1]], z=[tgt[2]],
                mode="markers", marker=dict(size=9, color="#dc2626", symbol="diamond", line=dict(color="white", width=1.5)),
                name=f"Target Hand ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f})"
            ), row=1, col=1)

        # 2. Subplot 2: Elbow
        e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax = add_kp_to_subplot(
            result.observed_elbow, result.elbow_mean, result.elbow_cov,
            col_obs="#06b6d4", col_pred="#0284c7", col_tube="#38bdf8", col_ell="#0369a1",
            row=1, col=2, name_prefix="Elbow"
        )
        if ground_truth_future_elbow is not None:
            gt_e = np.asarray(ground_truth_future_elbow)
            fig.add_trace(go.Scatter3d(
                x=gt_e[:, 0], y=gt_e[:, 1], z=gt_e[:, 2],
                mode="lines", line=dict(color="#0f766e", width=3.5, dash="dash"),
                name="Elbow True Motion"
            ), row=1, col=2)

        def make_scene_cfg(xmin, xmax, ymin, ymax, zmin, zmax):
            return dict(
                xaxis=dict(range=[xmin, xmax], title=dict(text="<b>X [m]</b>", font=dict(size=10)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                yaxis=dict(range=[ymin, ymax], title=dict(text="<b>Y [m]</b>", font=dict(size=10)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                zaxis=dict(range=[zmin, zmax], title=dict(text="<b>Z [m]</b>", font=dict(size=10)),
                           backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
                aspectmode="data",
                camera=dict(eye=dict(x=1.35, y=1.25, z=0.75), center=dict(x=0, y=0, z=0)),
            )

        main_title = title or (
            "<b>Separated Keypoint 3D Motion Predictions & Spatial Uncertainty Tubes</b><br>"
            f"<sup>Hand ADE: {ade_h:.2f} mm | Elbow ADE: {ade_e:.2f} mm | Latency: {result.latency_ms:.1f} ms</sup>"
        )
        fig.update_layout(
            title=dict(text=main_title, font=dict(size=14, family="sans-serif"), x=0.03, y=0.98, xanchor="left", yanchor="top"),
            legend=dict(
                orientation="h", yref="container", y=0.915, x=0.5, xanchor="center", yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=10, family="sans-serif")
            ),
            scene=make_scene_cfg(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
            scene2=make_scene_cfg(e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=35, r=35, t=150, b=35),
            width=1650,
            height=820,
        )

        if save_html:
            os.makedirs(os.path.dirname(save_html) or ".", exist_ok=True)
            fig.write_html(save_html)
            print(f"Saved interactive HTML to {save_html}")

        if save_pdf:
            os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
            fig.write_image(save_pdf, width=1650, height=820)
            print(f"Saved vector PDF to {save_pdf}")

        if save_png:
            os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
            fig.write_image(save_png, width=1650, height=820, scale=2)
            print(f"Saved high-res PNG to {save_png}")

        return fig
