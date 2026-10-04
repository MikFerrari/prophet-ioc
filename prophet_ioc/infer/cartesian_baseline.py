from typing import Tuple
from functools import partial
import numpy as np
import jax.numpy as jnp

from prophet_ioc.envs.cartesian_reaching import (
    CartesianMultiPointReaching3D,
    CartesianMultiPointReaching3DParams,
)
from prophet_ioc.control import gilqr


class CartesianMultiPointBaseline:
    r"""Cartesian Multi-Point Reaching Baseline (Independent Point Masses).

    Models human reaching movements in operational Cartesian space by treating
    key anatomical joints (such as the wrist and elbow) as independent point masses
    without kinematic skeletal linkage or joint angle constraints.

    Trajectories are solved via finite-horizon Generalized Iterative Linear Quadratic
    Regulator (generalized iLQR / LQG), balancing reaching target accuracy against
    velocity and control acceleration effort.
    """

    def __init__(
        self,
        dt: float = 0.01,
        w_target_hand: float = 100.0,
        w_target_elbow: float = 30.0,
        action_cost: float = 1e-4,
        velocity_cost: float = 1e-2,
        motor_noise: float = 0.1,
        obs_noise: float = 0.5,
    ):
        self.dt = dt
        self.w_target_hand = w_target_hand
        self.w_target_elbow = w_target_elbow
        self.params = CartesianMultiPointReaching3DParams(
            action_cost=jnp.float32(action_cost),
            velocity_cost=jnp.float32(velocity_cost),
            motor_noise=jnp.float32(motor_noise),
            obs_noise=jnp.float32(obs_noise),
        )

    def predict(
        self,
        p_elbow_obs: np.ndarray,
        p_wrist_obs: np.ndarray,
        v_elbow_obs: np.ndarray,
        v_wrist_obs: np.ndarray,
        target_elbow: np.ndarray,
        target_wrist: np.ndarray,
        horizon: int,
        max_iter: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Solves Cartesian reaching trajectory for elbow and wrist.

        Args:
            p_elbow_obs: Handover 3D position of elbow (3,).
            p_wrist_obs: Handover 3D position of wrist (3,).
            v_elbow_obs: Handover 3D velocity of elbow (3,).
            v_wrist_obs: Handover 3D velocity of wrist (3,).
            target_elbow: Final 3D target for elbow (3,).
            target_wrist: Final 3D target for wrist (3,).
            horizon: Number of future control steps H.
            max_iter: Max optimization iterations for iLQR.

        Returns:
            pred_elbow: Predicted trajectory of shape (H+1, 3).
            pred_wrist: Predicted trajectory of shape (H+1, 3).
        """
        x_obs = jnp.concatenate([
            jnp.asarray(p_elbow_obs, dtype=jnp.float32),
            jnp.asarray(p_wrist_obs, dtype=jnp.float32),
            jnp.asarray(v_elbow_obs, dtype=jnp.float32),
            jnp.asarray(v_wrist_obs, dtype=jnp.float32),
        ])
        env = CartesianMultiPointReaching3D(
            dt=self.dt,
            target_hand=jnp.asarray(target_wrist, dtype=jnp.float32),
            target_elbow=jnp.asarray(target_elbow, dtype=jnp.float32),
            x0=x_obs,
            w_target_hand=self.w_target_hand,
            w_target_elbow=self.w_target_elbow,
        )
        U_init = jnp.zeros((horizon, 6), dtype=jnp.float32)
        gains, X, U = gilqr.solve(p=env, x0=env.x0, U_init=U_init, params=self.params, max_iter=max_iter)
        X_np = np.array(X)
        pred_elbow = X_np[:, 0:3]
        pred_wrist = X_np[:, 3:6]
        return pred_elbow, pred_wrist
