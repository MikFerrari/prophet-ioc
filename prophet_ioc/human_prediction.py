"""Online prediction of human upper-body motion with the 19-DOF HumanKinematicReaching model.

`predict_motion` turns a 28-DOF joint history (IK of observed keypoints, sampled every dt) and a reaching target into
a prediction of the 9 upper-body joints over H steps up to the expected arrival, with the wrist / elbow position
covariance. It is the prediction used by the CARI v2 evaluation (evaluation/cari_kinematic.py) and by the ROS 2 node
(ros2/human_motion_predictor).

Conventions: Kalman-filtered handover state with the chest rotation vector relative to the chest orientation at the
last frame; running cost scaled by dt (weights independent of the time grid); wrist / elbow covariance =
cov_init + pred_noise^2 * cov_unit (KinematicPrediction), pred_noise being calibrated on recorded data.
"""

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import gammaln

import human_kinematic_model_jax as hkm
from prophet_ioc.control import gilqr
from prophet_ioc.data.cari import upper_body_dofs
from prophet_ioc.envs.base import Env
from prophet_ioc.envs.human_kinematic_reaching import _ENV_LEAVES, HumanKinematicParams, HumanKinematicReaching
from prophet_ioc.envs.wrappers import EKFWrapper

# The 9 upper-body joints of the prediction (and of the MPJPE)
JOINTS = ["head", "chest", "pelvis", "left_shoulder", "left_elbow", "left_wrist",
          "right_shoulder", "right_elbow", "right_wrist"]

_fk_batch = jax.jit(jax.vmap(hkm.fk, in_axes=(0, None)))


# =============================================================================
# Handover state estimation (Kalman filter over the observed joint history)
# =============================================================================
class JointKinematicsEnv(Env):
    """Kinematic joint-space integrator environment for EKF state estimation."""
    def __init__(self, n_dof: int = 19, dt: float = 0.01, damping: float = 0.20):
        self.n_dof = n_dof
        self.dt = dt
        self.damping = damping
        super().__init__(
            state_shape=(2 * n_dof,),
            action_shape=(n_dof,),
            observation_shape=(n_dof,),
            state_noise_shape=(2 * n_dof,),
            obs_noise_shape=(n_dof,),
        )

    def _dynamics(self, state, action, noise, params):
        q = state[: self.n_dof]
        qd = state[self.n_dof :]
        q_next = q + self.dt * qd
        qd_next = (1.0 - self.damping * self.dt) * qd + self.dt * action
        next_state = jnp.concatenate([q_next, qd_next])
        pro_noise = jnp.sqrt(self.dt) * params.motor_noise * noise[self.n_dof :]
        return next_state.at[self.n_dof :].add(pro_noise)

    def _observation(self, state, noise, params):
        return state[: self.n_dof] + params.obs_noise * noise

    def _cost(self, state, action, params):
        return 0.0

    def _final_cost(self, state, params):
        return 0.0

    def _reset(self, noise, params):
        return jnp.zeros(2 * self.n_dof, dtype=jnp.float32)


def kalman_filter_joint_history(
    q_obs_history: np.ndarray,
    dt: float,
    sigma_q0: float = 0.03,
    sigma_qd0: float = 0.30,
    sigma_a: float = 2.0,
    sigma_obs: float = 0.008,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman-filtered joint state [q, qd] at the last frame of the observed joint history (n, n_dof).

    Dynamics q_{k+1} = q_k + dt qd_k, qd_{k+1} = (1 - 0.2 dt) qd_k + w_k, observation y_k = q_k. EKFWrapper.filter_step
    is the one-step predictor form b_{k+1|k} = f(b_k) + K (y_k - h(b_k)): it is run over all but the last
    observation, which then enters a measurement update (filtered estimate at t_obs, not the prediction after it).
    Jitted (compiled once per history length).

    Returns:
        x_hat: (2*n_dof,) state estimate, P: (2*n_dof, 2*n_dof) its covariance
    """
    b, P = _kalman_filter(jnp.asarray(q_obs_history, dtype=jnp.float32), jnp.float32(dt), sigma_q0, sigma_qd0,
                          sigma_a, sigma_obs)
    return np.array(b), np.array(P)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _kalman_filter(ys, dt, sigma_q0, sigma_qd0, sigma_a, sigma_obs):
    n_dof = ys.shape[1]
    dim = 2 * n_dof
    b0 = jnp.zeros(dim, dtype=jnp.float32).at[:n_dof].set(ys[0])
    P0 = jnp.zeros((dim, dim), dtype=jnp.float32)
    P0 = P0.at[:n_dof, :n_dof].set((sigma_q0**2) * jnp.eye(n_dof))
    P0 = P0.at[n_dof:, n_dof:].set((sigma_qd0**2) * jnp.eye(n_dof))

    ekf = EKFWrapper(JointKinematicsEnv)(b0=(b0, P0), n_dof=n_dof, dt=dt)
    params = HumanKinematicParams(motor_noise=sigma_a, obs_noise=sigma_obs)
    action_zero = jnp.zeros(n_dof, dtype=jnp.float32)

    def scan_fn(carry, y_k):
        b_k, P_k = carry
        return ekf.filter_step(b_k, P_k, action_zero, y_k, params), None

    (b_pred, P_pred), _ = jax.lax.scan(scan_fn, (b0, P0), ys[:-1])
    H = jnp.hstack([jnp.eye(n_dof), jnp.zeros((n_dof, n_dof))])
    S = H @ P_pred @ H.T + sigma_obs**2 * jnp.eye(n_dof)
    K = P_pred @ H.T @ jnp.linalg.inv(S)
    return b_pred + K @ (ys[-1] - H @ b_pred), (jnp.eye(dim) - K @ H) @ P_pred


def handover_state(q28_history: np.ndarray, body_params: np.ndarray, dt: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimated 19-DOF state at the last frame of a 28-DOF joint history (n, 28) sampled every dt.

    Returns x0 (38,), P (38, 38) and q_chest_ref (4,), the chest orientation at the last frame (rotation-vector
    origin of the chest DOFs, so the trunk angular velocity of the history is kept).
    """
    q_chest_ref = np.asarray(q28_history[-1][3:7])
    q_hist = upper_body_dofs(q28_history, body_params, q_chest_ref)
    x0, P = kalman_filter_joint_history(q_hist, dt=dt)
    return x0, P, q_chest_ref


# =============================================================================
# Environment and prediction
# =============================================================================
def make_reaching_env(body_params: np.ndarray, legs_nominal: np.ndarray, q0: np.ndarray, q_chest_ref: np.ndarray,
                      target: np.ndarray, dt: float, hand: str, params: HumanKinematicParams) -> HumanKinematicReaching:
    """Upper-body reaching environment (pelvis root, running cost scaled by dt)."""
    return HumanKinematicReaching(
        mode="upper_body",
        dt=dt,
        target=target,
        q0=q0,
        body_params=body_params,
        q_chest_ref=q_chest_ref,
        legs_nominal=legs_nominal,
        reaching_hand=hand,
        w_target=params.w_target,
        posture_cost=params.posture_cost,
        base_disp_cost=params.base_disp_cost,
        root_joint="pelvis",
        dt_scaled_cost=True,
    )


@partial(jax.jit, static_argnames=("max_iter",))
def solve_kinematic(env: HumanKinematicReaching, x0, U, params, max_iter: int):
    """gILQR solve with the environment as a traced pytree argument: compiled once for all trials of a hand."""
    return gilqr.solve(p=env, x0=x0, U_init=U, params=params, max_iter=max_iter)


def sg_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Velocity at the last observed frame: Savitzky-Golay (window <= 7, order 2) over the observed positions."""
    w = min(7, len(positions))
    w -= 1 - w % 2
    if w >= 5:
        return savgol_filter(positions, w, 2, deriv=1, delta=dt, axis=0)[-1]
    return (positions[-1] - positions[-2]) / dt


def arrival_time(d_remaining: float, v_obs: float, t_max: float) -> float:
    """Expected arrival time from the last observation: decelerating at constant rate from the current speed,
    2 d / v, clamped to [0.15 s, t_max] (on CARI t_max is the remaining duration of the reach, online a parameter)."""
    return float(np.clip(2.0 * d_remaining / max(v_obs, 0.15), 0.15, t_max))


def to_timeline(a: np.ndarray, t_pred: float, t_gt: float) -> np.ndarray:
    """Resamples (H+1, ...) values on [0, t_pred] to the H+1 ground-truth times on [0, t_gt], holding the last value
    after t_pred."""
    H = len(a) - 1
    if t_pred >= t_gt - 1e-4:
        return a
    t_src, t_dst = np.linspace(0.0, t_pred, H + 1), np.linspace(0.0, t_gt, H + 1)
    flat = a.reshape(H + 1, -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape(a.shape)


CHI2_3_95 = 7.815  # 95 % quantile of the chi-square distribution with 3 degrees of freedom


@jax.jit
def _kinematic_outputs(env, X):
    """Keypoints, chest and the FK Jacobians of the reaching wrist, its elbow and the other wrist along a predicted
    state trajectory X (H+1, 38)."""
    return (jax.vmap(env.all_keypoints)(X), jax.vmap(env.chest)(X), jax.vmap(jax.jacobian(env.e))(X),
            jax.vmap(jax.jacobian(env.elbow))(X), jax.vmap(jax.jacobian(env.passive_wrist))(X))


@dataclass
class KinematicPrediction:
    """Kinematic prediction on the model time grid [0, t_pred] (H+1 steps).

    The position covariances of the reaching wrist ("wrist"), its elbow ("elbow") and the other wrist
    ("passive_wrist") are cov_init + pred_noise^2 * cov_unit: cov_init propagates the Kalman covariance at the
    handover, cov_unit a unit-intensity random walk of the joint velocities (variance dt per step and per joint,
    positions integrated), both through the closed-loop linearized dynamics and the FK Jacobians. pred_noise
    (joint-velocity noise, units/s/sqrt(s)) is calibrated by train.py (calibrate_pred_noise).
    """
    joints: Dict[str, np.ndarray]
    cov_init: Dict[str, np.ndarray]
    cov_unit: Dict[str, np.ndarray]
    t_pred: float
    dt_sim: float

    def cov(self, pred_noise: float) -> Dict[str, np.ndarray]:
        return {k: self.cov_init[k] + pred_noise ** 2 * self.cov_unit[k] for k in self.cov_init}


def _prediction(env: HumanKinematicReaching, L: np.ndarray, X, P0: np.ndarray, t_pred: float, dt_sim: float
                ) -> KinematicPrediction:
    """Joints and covariances of a solved prediction (feedback gains L (H, 19, 38), states X (H+1, 38))."""
    kp, chest, Jw, Je, Jp = (np.array(a) for a in _kinematic_outputs(env, X))
    pelvis = 0.5 * (kp[:, hkm.KP_INDEX["left_hip"]] + kp[:, hkm.KP_INDEX["right_hip"]])
    joints = {j: chest if j == "chest" else pelvis if j == "pelvis" else kp[:, hkm.KP_INDEX[j]] for j in JOINTS}

    n, H = 19, len(L)
    A = np.eye(2 * n)
    A[:n, n:] = dt_sim * np.eye(n)
    A[n:, n:] = (1.0 - 0.20 * dt_sim) * np.eye(n)
    B = np.zeros((2 * n, n))
    B[n:] = dt_sim * np.eye(n)
    W1 = np.zeros((2 * n, 2 * n))
    W1[n:, n:] = dt_sim * np.eye(n)
    W1[:n, :n] = dt_sim ** 3 / 3.0 * np.eye(n)
    W1[:n, n:] = W1[n:, :n] = dt_sim ** 2 / 2.0 * np.eye(n)
    S_init, S_unit = [np.array(P0, dtype=np.float64)], [np.zeros((2 * n, 2 * n))]
    for k in range(H):
        F = A + B @ L[k]
        S_init.append(F @ S_init[-1] @ F.T)
        S_unit.append(F @ S_unit[-1] @ F.T + W1)
    S_init, S_unit = np.array(S_init), np.array(S_unit)
    to_cart = lambda J, S: np.einsum("kij,kjl,kml->kim", J, S, J)
    jac = {"wrist": Jw, "elbow": Je, "passive_wrist": Jp}
    return KinematicPrediction(joints, {k: to_cart(J, S_init) for k, J in jac.items()},
                               {k: to_cart(J, S_unit) for k, J in jac.items()}, t_pred, dt_sim)


def predict_motion(q28_history: np.ndarray, dt: float, body_params: np.ndarray, target: np.ndarray,
                   params: HumanKinematicParams, H: int, max_iter: int, t_max: float, hand: Optional[str] = None,
                   legs_nominal: Optional[np.ndarray] = None) -> Tuple[KinematicPrediction, str]:
    """One complete prediction from a 28-DOF joint history (n, 28) sampled every dt (the last frame is the
    prediction start) to a known target, solved up to the expected arrival (offline evaluation; online with goal
    inference: predict_hypotheses): arrival-time estimate, Kalman-filtered handover state, environment, gILQR solve,
    keypoints and covariances. hand: reaching hand ("right" / "left"; None = the wrist that moved most over the
    history); legs_nominal: leg joints (8,) of the model (None = those of the first frame). Returns the prediction and
    the reaching hand."""
    q28_history = np.asarray(q28_history, dtype=np.float32)
    body = jnp.asarray(body_params, dtype=jnp.float32)
    kp_hist = np.array(_fk_batch(jnp.asarray(q28_history), body))
    if hand is None:
        disp = {s: np.linalg.norm(kp_hist[-1, hkm.KP_INDEX[f"{s}_wrist"]] - kp_hist[0, hkm.KP_INDEX[f"{s}_wrist"]])
                for s in ("right", "left")}
        hand = "right" if disp["right"] >= disp["left"] else "left"
    legs = q28_history[0][18:26] if legs_nominal is None else np.asarray(legs_nominal)
    wrist_obs = kp_hist[:, hkm.KP_INDEX[f"{hand}_wrist"]]
    v_obs = float(np.linalg.norm(sg_velocity(wrist_obs, dt))) if len(wrist_obs) > 1 else 0.0
    t_pred = arrival_time(float(np.linalg.norm(np.asarray(target) - wrist_obs[-1])), v_obs, t_max)
    dt_sim = t_pred / H

    x0, P0, q_chest_ref = handover_state(q28_history, body_params, dt)
    env = make_reaching_env(body_params, legs, x0[:19], q_chest_ref, target, dt_sim, hand, params)
    env.x0 = jnp.asarray(x0, dtype=jnp.float32)
    gains, X, _ = solve_kinematic(env, env.x0, jnp.zeros((H, 19), dtype=jnp.float32), params, max_iter=max_iter)
    return _prediction(env, np.array(gains.L), X, P0, t_pred, dt_sim), hand


# =============================================================================
# Receding-horizon prediction with goal hypotheses (online)
# =============================================================================
def minimum_jerk_state(p0: np.ndarray, v0: np.ndarray, a0: np.ndarray, pf: np.ndarray, D: float, t: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Position and velocity at time t of the minimum-jerk motion from (p0, v0, a0) to rest at pf (zero velocity and
    acceleration) in D seconds (quintic per axis, Flash & Hogan 1985); held at pf after D."""
    tau = float(np.clip(t / D, 0.0, 1.0))
    v0T, a0T2 = np.asarray(v0) * D, np.asarray(a0) * D ** 2
    d = np.asarray(pf) - np.asarray(p0)
    c3 = 10.0 * d - 6.0 * v0T - 1.5 * a0T2
    c4 = -15.0 * d + 8.0 * v0T + 1.5 * a0T2
    c5 = 6.0 * d - 3.0 * v0T - 0.5 * a0T2
    p = p0 + v0T * tau + 0.5 * a0T2 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
    v = (v0T + a0T2 * tau + 3 * c3 * tau ** 2 + 4 * c4 * tau ** 3 + 5 * c5 * tau ** 4) / D
    return p, v


_TAU = np.linspace(0.0, 1.0, 2001)[:-1]
_S = 10 * _TAU ** 3 - 15 * _TAU ** 4 + 6 * _TAU ** 5
_G = 30 * _TAU ** 2 * (1 - _TAU) ** 2 / (1 - _S)   # v D / d along a minimum-jerk motion (increasing in tau)


def minimum_jerk_remaining_time(d_remaining: float, v_towards: float, nominal_duration: float,
                                t_min: float = 0.15) -> float:
    """Expected time to the goal assuming a minimum-jerk motion of total duration nominal_duration: the phase tau
    solves v D / d = s'(tau) / (1 - s(tau)) (s = 10 tau^3 - 15 tau^4 + 6 tau^5) from the remaining distance and the
    speed towards the goal, and the remaining time is D (1 - tau). At rest: the motion starts now (D); at least
    t_min."""
    g = v_towards * nominal_duration / max(d_remaining, 1e-4)
    tau = float(np.interp(g, _G, _TAU))
    return max(nominal_duration * (1.0 - tau), t_min)


@dataclass(frozen=True)
class Hypothesis:
    """A hypothesis of the goal inference: reach `goal` (3,) with `hand`, or (goal None) come to rest."""
    name: str
    hand: str
    goal: Optional[Tuple[float, float, float]] = None

    @property
    def idle(self) -> bool:
        return self.goal is None


@dataclass
class HypothesisPrediction:
    """Prediction of one hypothesis. The optimal-control problem is solved over [0, prediction.t_pred] to `target`
    with the wrist velocity `target_vel` at its end: the goal itself if it is expected to be reached within the
    prediction horizon (temporary = False, target_vel = 0), otherwise the point of the minimum-jerk path to the goal
    reached at the horizon (temporary = True). arrival: expected arrival at the goal (s, may exceed the horizon)."""
    hypothesis: Hypothesis
    prediction: KinematicPrediction
    target: np.ndarray
    target_vel: np.ndarray
    arrival: float
    temporary: bool


@dataclass
class HandState:
    """Position, velocity and acceleration of the two wrists ("right" / "left") at the last frame of the history
    (Savitzky-Golay over the FK of the history)."""
    wrist: Dict[str, np.ndarray]
    wrist_vel: Dict[str, np.ndarray]
    wrist_acc: Dict[str, np.ndarray]


def _wrist_derivatives(positions: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    w = min(9, len(positions))
    w -= 1 - w % 2
    if w >= 5:
        return (savgol_filter(positions, w, 2, deriv=1, delta=dt, axis=0)[-1],
                savgol_filter(positions, w, 2, deriv=2, delta=dt, axis=0)[-1])
    return (positions[-1] - positions[-2]) / dt, np.zeros(3)


@lru_cache(maxsize=None)
def _env_template() -> HumanKinematicReaching:
    return make_reaching_env(np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2]), np.zeros(8), np.zeros(19),
                             np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), 0.05, "any", HumanKinematicParams())


def reaching_env_batch(body_params: np.ndarray, legs_nominal: np.ndarray, x0: np.ndarray, q_chest_ref: np.ndarray,
                       targets: np.ndarray, target_vels: np.ndarray, dts: np.ndarray, hands: List[str],
                       params: HumanKinematicParams) -> HumanKinematicReaching:
    """Batch of B reaching environments (as make_reaching_env, leading axis = hypothesis) from the same state x0
    (38,) to targets (B, 3) with terminal wrist velocities (B, 3), time steps dts (B,) and reaching hands (B,)
    (reaching_hand "any": the hand is a leaf, so both hands share the batch). Built from the pytree leaves
    directly: the constructor evaluates the FK eagerly (~35 ms per environment)."""
    template = _env_template()
    leaves, aux = template.tree_flatten()
    B = len(targets)
    shared = {"w_target": params.w_target, "posture_cost": params.posture_cost,
              "base_disp_cost": params.base_disp_cost, "q_chest_ref": q_chest_ref / np.linalg.norm(q_chest_ref),
              "legs_nominal": legs_nominal, "body_params": body_params, "q0": x0[:19], "q_posture_ref": x0[:19],
              "x0": x0}
    per = {"dt": dts, "target": targets, "target_vel": target_vels,
           "right_hand": np.array([1.0 if h == "right" else 0.0 for h in hands])}
    batch = []
    for name, leaf in zip(_ENV_LEAVES, leaves):
        if name in per:
            value = np.asarray(per[name], dtype=np.float32)
        else:
            value = np.broadcast_to(np.asarray(shared.get(name, leaf), dtype=np.float32),
                                    (B,) + np.shape(leaf)).copy()
        batch.append(jnp.asarray(value))
    return HumanKinematicReaching.tree_unflatten(aux, batch)


@partial(jax.jit, static_argnames=("max_iter",))
def _solve_batch(envs: HumanKinematicReaching, x0, U, params, max_iter: int):
    """gILQR solves of a batch of environments, followed by their keypoints and FK Jacobians."""
    def one(env):
        gains, X, _ = gilqr.solve(p=env, x0=x0, U_init=U, params=params, max_iter=max_iter)
        return gains.L, X, _kinematic_outputs(env, X)
    return jax.vmap(one)(envs)


def _predictions_batch(L: np.ndarray, X: np.ndarray, outputs, P0: np.ndarray, t_preds: np.ndarray
                       ) -> List[KinematicPrediction]:
    """KinematicPrediction of each element of a batch (gains L (B, H, 19, 38), keypoints and Jacobians of
    _kinematic_outputs with a leading batch axis): the covariance propagation of _prediction, vectorized."""
    kp, chest, Jw, Je, Jp = (np.asarray(a, dtype=np.float64) for a in outputs)
    B, H = L.shape[:2]
    n = 19
    dts = np.asarray(t_preds, dtype=np.float64) / H
    I = np.eye(2 * n)
    A = np.broadcast_to(I, (B, 2 * n, 2 * n)).copy()
    A[:, :n, n:] = dts[:, None, None] * np.eye(n)
    A[:, n:, n:] = (1.0 - 0.20 * dts)[:, None, None] * np.eye(n)
    W1 = np.zeros((B, 2 * n, 2 * n))
    W1[:, n:, n:] = dts[:, None, None] * np.eye(n)
    W1[:, :n, :n] = (dts ** 3 / 3.0)[:, None, None] * np.eye(n)
    W1[:, :n, n:] = W1[:, n:, :n] = (dts ** 2 / 2.0)[:, None, None] * np.eye(n)
    S_init = [np.broadcast_to(np.asarray(P0, dtype=np.float64), (B, 2 * n, 2 * n))]
    S_unit = [np.zeros((B, 2 * n, 2 * n))]
    for k in range(H):
        F = A.copy()
        F[:, n:, :] += dts[:, None, None] * np.asarray(L[:, k], dtype=np.float64)   # A + B L, B = [0; dt I]
        S_init.append(F @ S_init[-1] @ F.transpose(0, 2, 1))
        S_unit.append(F @ S_unit[-1] @ F.transpose(0, 2, 1) + W1)
    S_init, S_unit = np.stack(S_init, axis=1), np.stack(S_unit, axis=1)   # (B, H+1, 38, 38)
    to_cart = lambda J, S: np.einsum("bkij,bkjl,bkml->bkim", J, S, J)
    jac = {"wrist": Jw, "elbow": Je, "passive_wrist": Jp}
    c_init = {k: to_cart(J, S_init) for k, J in jac.items()}
    c_unit = {k: to_cart(J, S_unit) for k, J in jac.items()}
    pelvis = 0.5 * (kp[:, :, hkm.KP_INDEX["left_hip"]] + kp[:, :, hkm.KP_INDEX["right_hip"]])
    out = []
    for b in range(B):
        joints = {j: chest[b] if j == "chest" else pelvis[b] if j == "pelvis" else kp[b, :, hkm.KP_INDEX[j]]
                  for j in JOINTS}
        out.append(KinematicPrediction(joints, {k: v[b] for k, v in c_init.items()},
                                       {k: v[b] for k, v in c_unit.items()}, float(t_preds[b]), float(dts[b])))
    return out


def predict_hypotheses(q28_history: np.ndarray, dt: float, body_params: np.ndarray, hypotheses: List[Hypothesis],
                       params: HumanKinematicParams, H: int, max_iter: int, horizon: float, nominal_duration: float,
                       legs_nominal: Optional[np.ndarray] = None, stop_time: float = 0.3
                       ) -> Tuple[List[HypothesisPrediction], HandState]:
    """Receding-horizon prediction of every hypothesis from a 28-DOF joint history (n, 28) sampled every dt.

    Shared Kalman handover state; per hypothesis, the expected arrival at the goal (minimum_jerk_remaining_time with
    the wrist speed towards the goal: a hypothesis assumes its reach is under way, or starts now) and the target of the optimal-control problem over the prediction horizon
    (HypothesisPrediction). An idle hypothesis comes to rest in stop_time s (free end position of the minimum-jerk
    stop: p0 + v0 T / 2 + a0 T^2 / 12). All hypotheses are solved together (vmap over both hands, compiled once per
    number of hypotheses)."""
    q28_history = np.asarray(q28_history, dtype=np.float32)
    body_params = np.asarray(body_params, dtype=np.float32)
    kp_hist = np.array(_fk_batch(jnp.asarray(q28_history), jnp.asarray(body_params)))
    hs = HandState({}, {}, {})
    for side in ("right", "left"):
        w = kp_hist[:, hkm.KP_INDEX[f"{side}_wrist"]].astype(np.float64)
        hs.wrist[side] = w[-1]
        hs.wrist_vel[side], hs.wrist_acc[side] = _wrist_derivatives(w, dt)
    legs = q28_history[0][18:26] if legs_nominal is None else np.asarray(legs_nominal)
    x0, P0, q_chest_ref = handover_state(q28_history, body_params, dt)

    specs = []
    for hyp in hypotheses:
        p0, v0, a0 = hs.wrist[hyp.hand], hs.wrist_vel[hyp.hand], hs.wrist_acc[hyp.hand]
        if hyp.idle:
            T = stop_time
            specs.append((p0 + 0.5 * v0 * T + a0 * T ** 2 / 12.0, np.zeros(3), T, T, False))
            continue
        goal = np.asarray(hyp.goal, dtype=np.float64)
        d = goal - p0
        dist = float(np.linalg.norm(d))
        v_towards = max(float(v0 @ d) / max(dist, 1e-6), 0.0)
        arrival = minimum_jerk_remaining_time(dist, v_towards, nominal_duration)
        if arrival <= horizon:
            specs.append((goal, np.zeros(3), arrival, arrival, False))
        else:
            target, target_vel = minimum_jerk_state(p0, v0, a0, goal, arrival, horizon)
            specs.append((target, target_vel, horizon, arrival, True))

    targets, target_vels, t_preds, arrivals, temporary = (np.array(v) for v in zip(*specs))
    envs = reaching_env_batch(body_params, legs, x0, q_chest_ref, targets, target_vels, t_preds / H,
                              [h.hand for h in hypotheses], params)
    solved = _solve_batch(envs, jnp.asarray(x0, dtype=jnp.float32), jnp.zeros((H, 19), dtype=jnp.float32), params,
                          max_iter=max_iter)
    L, X, outputs = jax.device_get(solved)   # one transfer
    preds = _predictions_batch(L, X, outputs, P0, t_preds)
    out = [HypothesisPrediction(h, p, targets[i], target_vels[i], float(arrivals[i]), bool(temporary[i]))
           for i, (h, p) in enumerate(zip(hypotheses, preds))]
    return out, hs


def goal_cue_logprior(hypotheses: List[Hypothesis], hs: HandState, head: Optional[np.ndarray],
                      gaze: Optional[np.ndarray], kappa_heading: float, kappa_gaze: float, v_ref: float = 0.3
                      ) -> np.ndarray:
    """Log prior of the hypotheses from instantaneous cues (von Mises-like): the direction of the wrist velocity
    towards the goal, weighted by the wrist speed (min(|v| / v_ref, 1): no heading when still), and the direction of
    the gaze (head position and unit direction, e.g. nose - mid-ears of the ZED skeleton; None = no gaze) towards the
    goal. Idle hypotheses: 0."""
    out = np.zeros(len(hypotheses))
    for i, hyp in enumerate(hypotheses):
        if hyp.idle:
            continue
        goal = np.asarray(hyp.goal)
        d = goal - hs.wrist[hyp.hand]
        v = hs.wrist_vel[hyp.hand]
        speed = float(np.linalg.norm(v))
        if speed > 1e-6 and np.linalg.norm(d) > 1e-6:
            out[i] += kappa_heading * min(speed / v_ref, 1.0) * float(v @ d) / (speed * np.linalg.norm(d))
        if gaze is not None and head is not None:
            g = goal - head
            out[i] += kappa_gaze * float(gaze @ g) / (np.linalg.norm(gaze) * np.linalg.norm(g) + 1e-9)
    return out


def wrist_logdensity(r: np.ndarray, S: np.ndarray, nu: float = 4.0) -> np.ndarray:
    """Log density of residuals r (..., 3) under a multivariate Student-t with scale matrices S (..., 3, 3) and nu
    degrees of freedom (heavy tails: an outlier of the IK, e.g. a limb solution flip, does not decide the goal)."""
    m = np.einsum("...i,...ij,...j->...", r, np.linalg.inv(S), r)
    logdet = np.linalg.slogdet(S)[1]
    c = gammaln((nu + 3) / 2) - gammaln(nu / 2) - 1.5 * np.log(nu * np.pi)
    return c - 0.5 * logdet - 0.5 * (nu + 3) * np.log1p(m / nu)


class GoalFilter:
    """Recursive Bayesian inference of the hypothesis (goal and hand, or idle) from the receding-horizon predictions.

    Hidden Markov model over the hypotheses: between updates the hypothesis switches with rate switch_rate (1/s),
    p <- (1 - eps) p + eps / K, eps = 1 - exp(-switch_rate dt). Evidence: the observed (Kalman-filtered) positions of
    both wrists now, under the prediction each hypothesis made `evidence_lag` s ago (wrist_logdensity: Student-t with
    the predicted covariance of the reaching and passive wrist + obs_noise^2), tempered by `temperature` (successive updates share most of their
    lag window). The cue prior (goal_cue_logprior) multiplies the filtered belief at output and is not accumulated.
    """

    def __init__(self, hypotheses: List[Hypothesis], pred_noise: float, switch_rate: float = 0.5,
                 evidence_lag: float = 0.3, temperature: float = 0.25, obs_noise: float = 0.01):
        self.hypotheses = list(hypotheses)
        self.pred_noise, self.switch_rate, self.lag = pred_noise, switch_rate, evidence_lag
        self.temperature, self.obs_noise = temperature, obs_noise
        self.reset()

    def reset(self):
        self.log_belief = np.full(len(self.hypotheses), -np.log(len(self.hypotheses)))
        self.history: List[Tuple[float, List[HypothesisPrediction]]] = []
        self.t_last: Optional[float] = None

    def loglikelihood(self, preds: List[HypothesisPrediction], tau: float, wrists: Dict[str, np.ndarray]) -> np.ndarray:
        """Log density of the observed wrists (side -> (3,)) tau s after the predictions."""
        out = np.zeros(len(preds))
        for i, hp_ in enumerate(preds):
            joints, cov = sample_prediction(hp_.prediction, np.array([tau]), self.pred_noise)
            hand = hp_.hypothesis.hand
            other = "left" if hand == "right" else "right"
            for side, key in ((hand, "wrist"), (other, "passive_wrist")):
                out[i] += wrist_logdensity(wrists[side] - joints[f"{side}_wrist"][0],
                                           cov[key][0] + self.obs_noise ** 2 * np.eye(3))
        return out

    def update(self, t: float, preds: List[HypothesisPrediction], log_prior: Optional[np.ndarray] = None
               ) -> np.ndarray:
        """Update with the predictions made at time t (their first sample is the filtered observation at t); returns
        the posterior over the hypotheses (cue prior included)."""
        if self.t_last is not None:
            eps = 1.0 - np.exp(-self.switch_rate * max(t - self.t_last, 0.0))
            p = np.exp(self.log_belief - self.log_belief.max())
            p = (1.0 - eps) * p / p.sum() + eps / len(p)
            self.log_belief = np.log(p)
            wrists = {s: preds[0].prediction.joints[f"{s}_wrist"][0] for s in ("right", "left")}
            past = [(tj, pj) for tj, pj in self.history if tj <= t - self.lag + 1e-6]
            tj, pj = past[-1] if past else self.history[0]
            ll = self.loglikelihood(pj, t - tj, wrists)
            self.log_belief = self.log_belief + self.temperature * (ll - ll.max())
            self.log_belief -= np.log(np.sum(np.exp(self.log_belief - self.log_belief.max()))) + self.log_belief.max()
        self.history = [(tj, pj) for tj, pj in self.history if tj >= t - self.lag - 0.5] + [(t, preds)]
        self.t_last = t
        post = self.log_belief + (0.0 if log_prior is None else log_prior)
        post = np.exp(post - post.max())
        return post / post.sum()


def resample_history(stamps: np.ndarray, q28: np.ndarray, t_end: float, duration: float, n: int) -> np.ndarray:
    """28-DOF configurations (n, 28) at n uniform times over [t_end - duration, t_end], linearly interpolated from
    measurements (stamps (m,), q28 (m, 28)) of irregular rate. The chest quaternions are brought to the hemisphere
    of the last one before the interpolation and normalized after it."""
    q28 = np.array(q28, dtype=np.float64)
    quat = q28[:, 3:7]
    quat[np.sum(quat * quat[-1], axis=1) < 0.0] *= -1.0
    times = np.linspace(t_end - duration, t_end, n)
    out = np.stack([np.interp(times, stamps, q28[:, j]) for j in range(q28.shape[1])], axis=1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out.astype(np.float32)


def sample_prediction(pred: KinematicPrediction, times: np.ndarray, pred_noise: float
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Joints {joint: (n, 3)} and wrist / elbow covariances {part: (n, 3, 3)} of a prediction at the times (n,) after
    the last observation; after the expected arrival (pred.t_pred) the final posture and covariance are held."""
    H = len(pred.joints[JOINTS[0]]) - 1
    t_model = np.linspace(0.0, pred.t_pred, H + 1)

    def at(a):
        flat = a.reshape(H + 1, -1)
        return np.stack([np.interp(times, t_model, flat[:, j]) for j in range(flat.shape[1])],
                        axis=1).reshape((len(times),) + a.shape[1:])

    return {j: at(v) for j, v in pred.joints.items()}, {k: at(v) for k, v in pred.cov(pred_noise).items()}


def coverage_fraction(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per sample: is the error (n, 3) inside the 95 % ellipsoid of cov (n, 3, 3)?"""
    m = np.einsum("ki,kij,kj->k", err, np.linalg.inv(cov + 1e-9 * np.eye(3)), err)
    return m <= CHI2_3_95
