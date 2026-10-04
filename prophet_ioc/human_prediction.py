"""Online prediction of human upper-body motion with the 19-DOF HumanKinematicReaching model.

`predict_motion` turns a 28-DOF joint history (IK of observed keypoints, sampled every dt) and a reaching target into
a prediction of the 9 upper-body joints over H steps up to the expected arrival, with the wrist / elbow position
covariance. It is the prediction used by the CARI v2 evaluation (evaluation/cari_kinematic.py); `predict_hypotheses`
is its online, multi-goal version (ROS 2 node in ros2/human_motion_predictor, evaluation/goal_inference.py).

Run-time prediction (methodology, "Run-time prediction"):
- handover: a Kalman filter over the observed joint history, under the model's ZOH dynamics (prophet_ioc.envs.zoh,
  damping params.damping), gives the state estimate x0_hat and its covariance P0 (the observer's uncertainty, not the
  agent's belief); the chest rotation vector is relative to the chest orientation at the last frame;
- the agent's policy is solved from x0_hat (fully observed model) or from its belief mean (partially observed) to the
  goal over the expected arrival time (ilqr_unrolled.solve: at most max_iter iterations, early stopping at tol);
- predictive distribution (PredictionSettings.covariance):
    "model" (default): the model's own: mean = the nominal of the policy, covariance propagated through the
        closed-loop linearization, Sigma_(k+1) = F_k Sigma_k F_k^T + E[V_k V_k^T] + W_k W_k^T, Sigma_0 = P0, with
        the motor noise V (signal dependent: at the nominal command plus the spread of the commands of the closed
        loop, gaussian.command_noise_second_moment), the max-ent decision noise W = B Gamma (temperature of the
        fit) and, by default, the likelihood-only residual noise of the fit (PredictionSettings.residual,
        propagation_params); partially observed: the joint (state, belief) Gaussian through the joint dynamics g
        (predictive_distribution);
    "random_walk" (ablation, the former default): cov_init + pred_noise^2 * cov_unit, P0 and a unit joint-velocity
        random walk through the closed loop A + B L, pred_noise calibrated on recorded data (train.py);
  keypoint / wrist / elbow covariances follow through the FK Jacobians, J Sigma J^T.
"""

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Dict, List, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import gammaln

import human_kinematic_model_jax as hkm
from prophet_ioc.control import ilqr_unrolled
from prophet_ioc.control.lqr import Gains
from prophet_ioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from prophet_ioc.data.cari import rts_smooth, upper_body_dofs
from prophet_ioc.envs.base import Env
from prophet_ioc.envs.human_kinematic_reaching import _ENV_LEAVES, HumanKinematicParams, HumanKinematicReaching
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.envs.zoh import zoh_coefficients, zoh_noise_chol, zoh_process_cov, zoh_state_matrices
from prophet_ioc.infer.inv_ilqg import SolvedModel, create_filtered_joint_dynamics, joint_predictive_moments
from prophet_ioc.infer.inv_ilqr import closed_loop_moments
from prophet_ioc.infer.multi_env import filter_gains, initial_belief, prefix_belief

JOINTS = ["head", "chest", "pelvis", "left_shoulder", "left_elbow", "left_wrist",
          "right_shoulder", "right_elbow", "right_wrist"]

_fk_batch = jax.jit(jax.vmap(hkm.fk, in_axes=(0, None)))


# =============================================================================
# Handover state estimation (Kalman filter over the observed joint history)
# =============================================================================
class JointKinematicsEnv(Env):
    """Joint-space damped double integrator for EKF state estimation: the deterministic part of HumanKinematicReaching
    (exact ZOH, damping b) driven by a white-noise acceleration of intensity params.motor_noise (covariance
    motor_noise^2 M(dt) per joint, prophet_ioc.envs.zoh), observed joint positions with std params.obs_noise."""
    def __init__(self, n_dof: int = 19, dt: float = 0.01, damping: float = 0.0):
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
        n = self.n_dof
        q, qd = state[:n], state[n:]
        e, a1, a2 = zoh_coefficients(self.dt, self.damping)
        l11, l21, l22 = zoh_noise_chol(self.dt)
        s = params.motor_noise
        return jnp.concatenate([q + a1 * qd + a2 * action + s * l11 * noise[:n],
                                e * qd + a1 * action + s * (l21 * noise[:n] + l22 * noise[n:])])

    def _observation(self, state, noise, params):
        return state[: self.n_dof] + params.obs_noise * noise

    def _cost(self, state, action, params):
        return 0.0

    def _final_cost(self, state, params):
        return 0.0

    def _reset(self, noise, params):
        return jnp.zeros(2 * self.n_dof, dtype=jnp.float32)


# Handover filter: prior std of the joint positions / velocities at the first frame, white-noise acceleration
# intensity, IK joint-angle noise (rad; m for the pelvis)
HANDOVER_SIGMA_Q0, HANDOVER_SIGMA_QD0, HANDOVER_SIGMA_A, HANDOVER_SIGMA_OBS = 0.03, 0.30, 2.0, 0.008


def kalman_filter_joint_history(
    q_obs_history: np.ndarray,
    dt: float,
    sigma_q0: float = HANDOVER_SIGMA_Q0,
    sigma_qd0: float = HANDOVER_SIGMA_QD0,
    sigma_a: float = HANDOVER_SIGMA_A,
    sigma_obs: float = HANDOVER_SIGMA_OBS,
    damping: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman-filtered joint state [q, qd] at the last frame of the observed joint history (n, n_dof).

    Dynamics: the exact ZOH damped double integrator of the model (JointKinematicsEnv, damping b) driven by a white-
    noise acceleration of intensity sigma_a; observation y_k = q_k + noise (sigma_obs). EKFWrapper.filter_step
    is the one-step predictor form b_{k+1|k} = f(b_k) + K (y_k - h(b_k)): it is run over all but the last
    observation, which then enters a measurement update (filtered estimate at t_obs, not the prediction after it).
    Jitted (compiled once per history length).

    Returns:
        x_hat: (2*n_dof,) state estimate, P: (2*n_dof, 2*n_dof) its covariance
    """
    b, P = _kalman_filter(jnp.asarray(q_obs_history, dtype=jnp.float32), jnp.float32(dt), sigma_q0, sigma_qd0,
                          sigma_a, sigma_obs, jnp.float32(damping))
    return np.array(b), np.array(P)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _kalman_filter(ys, dt, sigma_q0, sigma_qd0, sigma_a, sigma_obs, damping):
    n_dof = ys.shape[1]
    dim = 2 * n_dof
    b0 = jnp.zeros(dim, dtype=jnp.float32).at[:n_dof].set(ys[0])
    P0 = jnp.zeros((dim, dim), dtype=jnp.float32)
    P0 = P0.at[:n_dof, :n_dof].set((sigma_q0**2) * jnp.eye(n_dof))
    P0 = P0.at[n_dof:, n_dof:].set((sigma_qd0**2) * jnp.eye(n_dof))

    ekf = EKFWrapper(JointKinematicsEnv)(b0=(b0, P0), n_dof=n_dof, dt=dt, damping=damping)
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


def handover_state(q28_history: np.ndarray, body_params: np.ndarray, dt: float, damping: float = 0.0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimated 19-DOF state at the last frame of a 28-DOF joint history (n, 28) sampled every dt (Kalman filter
    under the model's dynamics with damping b, kalman_filter_joint_history).

    Returns x0 (38,), P (38, 38) and q_chest_ref (4,), the chest orientation at the last frame (rotation-vector
    origin of the chest DOFs, so the trunk angular velocity of the history is kept).
    """
    q_chest_ref = np.asarray(q28_history[-1][3:7])
    q_hist = upper_body_dofs(q28_history, body_params, q_chest_ref)
    x0, P = kalman_filter_joint_history(q_hist, dt=dt, damping=damping)
    return x0, P, q_chest_ref


def prefix_states(q_hist: np.ndarray, dt: float, x_last: np.ndarray, damping: float = 0.0) -> np.ndarray:
    """Estimated 19-DOF states (n, 38) along the observed joint history q_hist (n, 19) (partially observed
    prediction: the states over which the agent's belief is tracked).

    RTS smoother (prophet_ioc.data.cari.rts_smooth) with the noises and prior of the handover filter
    (kalman_filter_joint_history): causal with respect to the handover (only the history is used), and its last
    state is the handover estimate itself (the smoother's last state is the filter's), set to x_last exactly
    (they differ by float32 rounding)."""
    q, qd = rts_smooth(q_hist, dt, accel_noise=HANDOVER_SIGMA_A, obs_noise=HANDOVER_SIGMA_OBS, damping=damping,
                       vel0_std=HANDOVER_SIGMA_QD0, pos0_std=HANDOVER_SIGMA_Q0)
    states = np.concatenate([q, qd], axis=1)
    states[-1] = x_last
    return states


def belief_window(states: np.ndarray, dt: float, dt_sim: float, n_steps: int) -> np.ndarray:
    """The last n_steps steps of the estimated history states (n, d) (sampled every dt, the last one at the
    handover) on the prediction grid dt_sim: (n_steps + 1, d) at t = -n_steps dt_sim, ..., -dt_sim, 0 (linear
    interpolation; the last row is the handover state)."""
    t_hist = (np.arange(len(states)) - (len(states) - 1)) * dt
    t = -dt_sim * np.arange(n_steps, -1, -1)
    out = np.stack([np.interp(t, t_hist, states[:, j]) for j in range(states.shape[1])], axis=1)
    out[-1] = states[-1]
    return out


def belief_steps_available(n_hist: int, dt: float, dt_sim: float, max_steps: int) -> int:
    """Number of prediction steps (dt_sim) of the history (n_hist samples every dt) usable for the belief, at most
    max_steps."""
    return int(max(0, min(max_steps, np.floor((n_hist - 1) * dt / dt_sim + 1e-6))))


# =============================================================================
# Environment and prediction
# =============================================================================
def make_reaching_env(body_params: np.ndarray, legs_nominal: np.ndarray, q0: np.ndarray, q_chest_ref: np.ndarray,
                      target: np.ndarray, dt: float, hand: str,
                      target_left: Optional[np.ndarray] = None,
                      alpha_right: Optional[float] = None,
                      alpha_left: Optional[float] = None) -> HumanKinematicReaching:
    """Upper-body reaching environment (pelvis root, running cost scaled by dt). The cost weights, noises and damping
    are not part of the environment: they are the params of every call (HumanKinematicParams)."""
    return HumanKinematicReaching(
        mode="upper_body",
        dt=dt,
        target=target,
        q0=q0,
        body_params=body_params,
        q_chest_ref=q_chest_ref,
        legs_nominal=legs_nominal,
        reaching_hand=hand,
        root_joint="pelvis",
        dt_scaled_cost=True,
        target_left=target_left,
        alpha_right=alpha_right,
        alpha_left=alpha_left,
    )



def solver_tolerance(model_cfg) -> Optional[float]:
    """Early-stopping tolerance of the forward solver of a model configuration (config/model/human_kinematic.yaml):
    model.tol if model.early_stopping, else None (exactly model.max_iter iterations)."""
    return float(model_cfg.get("tol", 1e-3)) if model_cfg.get("early_stopping", False) else None


@partial(jax.jit, static_argnames=("max_iter", "tol"))
def solve_kinematic(env: HumanKinematicReaching, x0, U, params, max_iter: int, tol: Optional[float] = None):
    """gILQR solve (ilqr_unrolled.solve: at most max_iter iterations, early stopping with tol) with the environment as
    a traced pytree argument: compiled once for all trials of a hand. Returns (gains, X, U)."""
    return ilqr_unrolled.solve(env, x0, U, params, max_iter=max_iter, tol=tol)


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

# =============================================================================
# Predictive distribution of the model
# =============================================================================
@dataclass(frozen=True)
class PredictionSettings:
    """How the prediction is made (hashable: a static argument of the compiled predictors).

    covariance: "model" (the model's own predictive covariance, default), "random_walk" (calibrated joint-velocity
        random walk, ablation; needs pred_noise) or "both" (both computed, e.g. to calibrate the random walk and
        report the model's coverage in train.py);
    residual: add the likelihood-only residual noise of the fit (params.residual_noise) to the model covariance;
    observability: "full" (the agent knows its state) or "partial" (belief-space prediction, the trained model is
        partially observed);
    temperature: of the max-ent policy (decision noise W = B Gamma; 0 = deterministic policy);
    belief_steps: partially observed: prediction steps of the history over which the agent's belief is tracked.
    """
    covariance: str = "model"
    residual: bool = True
    observability: str = "full"
    temperature: float = 1e-6
    belief_steps: int = 10

    def __post_init__(self):
        if self.covariance not in ("model", "random_walk", "both"):
            raise ValueError(f"covariance must be model, random_walk or both, got {self.covariance}")
        if self.observability not in ("full", "partial"):
            raise ValueError(f"observability must be full or partial, got {self.observability}")

    @property
    def model_covariance(self) -> bool:
        return self.covariance != "random_walk"

    @property
    def random_walk(self) -> bool:
        return self.covariance != "model"


def prediction_settings(model_cfg: Optional[Mapping] = None, record: Optional[Mapping] = None,
                        ioc_cfg: Optional[Mapping] = None) -> PredictionSettings:
    """PredictionSettings of a configuration: model_cfg (config/model/human_kinematic.yaml: prediction_covariance,
    prediction_residual, prediction_observability, belief_steps), record (the train.py params.json of the fitted
    weights: observability and temperature of the fit; None = initial weights) and ioc_cfg (config ioc: temperature
    when the record has none).

    prediction_observability "auto" (default) predicts with the observability the weights were fitted with (partial
    only for a partially observed likelihood fit, which also fitted obs_noise); "full" / "partial" force it."""
    get = lambda cfg, key, default: default if cfg is None else cfg.get(key, default)
    record = record or {}
    observability = str(get(model_cfg, "prediction_observability", "auto"))
    if observability == "auto":
        fitted = record.get("objective") == "likelihood"
        observability = str(record.get("observability", "full")) if fitted else "full"
    return PredictionSettings(covariance=str(get(model_cfg, "prediction_covariance", "model")),
                              residual=bool(get(model_cfg, "prediction_residual", True)),
                              observability=observability,
                              temperature=float(record.get("temperature", get(ioc_cfg, "temperature", 1e-6))),
                              belief_steps=int(get(model_cfg, "belief_steps", 10)))


def propagation_params(params: HumanKinematicParams, settings: PredictionSettings) -> HumanKinematicParams:
    """Parameters of the covariance propagation: params, with the likelihood-only residual noise added when
    settings.residual.

    Why the residual: the fit explains the recorded transitions by the model's noises plus a white-noise acceleration
    of intensity residual_noise (model mismatch, HumanKinematicReaching.residual_covariance), with the controller's
    own motor noise fixed (fitted to the residuals it went to its bound and the plans stopped short). The predictive
    distribution of recorded motion consistent with that likelihood therefore contains the residual, fed back by the
    policy like any other disturbance; without it only the controller's small motor noise and the handover
    uncertainty remain (overconfident). It has the ZOH structure of the additive motor noise (residual_noise^2 M(dt)
    per joint), so it is folded into motor_noise_add: the variances add, and the gLQR gains do not depend on additive
    noise, so the policy is unchanged (it is in any case solved with params; and the agent's filter gains of the
    partially observed model are computed with params: the agent does not know about the residual)."""
    if not settings.residual:
        return params
    return params._replace(motor_noise_add=jnp.sqrt(params.motor_noise_add ** 2 + params.residual_noise ** 2))


def _plan_policy(gains: Gains, X: jnp.ndarray, U: jnp.ndarray, temperature: float):
    """The agent's policy around the solved nominal (X, U): u = U_t + L_t (x - X_t) (+ Gamma_t xi, max-ent).
    ilqr_unrolled.solve returns the gains of its last iteration with the accepted feedforward step already applied to
    (X, U), so l is dropped: the mean of the closed loop is then the nominal itself."""
    g = Gains(gains.L, jnp.zeros_like(gains.l), gains.H)
    return create_maxent_lqr_policy(g, X, U, temperature) if temperature > 0 else create_lqr_policy(g, X, U)


def _with_start(env: HumanKinematicReaching, x0: jnp.ndarray) -> HumanKinematicReaching:
    """The environment with another start state x0 (x0 and q0, the reference of the pelvis displacement cost)."""
    leaves, aux = env.tree_flatten()
    leaves = [x0[:env.n_dof] if name == "q0" else x0 if name == "x0" else leaf
              for name, leaf in zip(_ENV_LEAVES, leaves)]
    return HumanKinematicReaching.tree_unflatten(aux, leaves)


def predictive_distribution(env: HumanKinematicReaching, x_prefix: jnp.ndarray, P0: jnp.ndarray,
                            params: HumanKinematicParams, settings: PredictionSettings, H: int, max_iter: int,
                            tol: Optional[float] = None) -> Dict[str, jnp.ndarray]:
    """Predictive distribution of the state over H steps of env.dt from the handover (traceable: jit / vmap).

    x_prefix (n+1, d): the estimated states of the history on the prediction grid, the last one the handover estimate
    x0_hat (fully observed: only the last one is used; partially observed: belief_window); P0 (d, d): covariance of
    x0_hat. Returns the mean (H+1, d) of the state, the agent's nominal (H+1, d), the feedback gains L (H, n_dof, d)
    and, with the model covariance (always when partially observed), Sigma (H+1, d, d).

    Fully observed: the agent plans from x0_hat; mean = its nominal; Sigma by closed_loop_moments (closed-loop
    linearization F = A + B L, motor noise at the nominal command, decision noise B Gamma, Sigma_0 = P0).

    Partially observed: the agent planned from the start of the window (initial belief there as in the likelihood)
    and tracked its belief over the window's recorded states (multi_env.prefix_belief: Algorithm 1 over the prefix,
    conditioned on every state) -> joint Gaussian of (x0, b0) given the history. It now plans from its belief mean
    mu_b (not from x0_hat), and the joint Gaussian is propagated through the joint dynamics g of that plan (with the
    agent's filter continuing from its covariance at the handover) WITHOUT conditioning: nothing is observed in the
    future (inv_ilqg.joint_predictive_moments). The mean of x is not the nominal when mu_b differs from x0_hat (the
    agent acts on what it believes). With sigma_o -> 0 the belief is the state and this is the fully observed
    prediction."""
    d, n_u = x_prefix.shape[-1], env.n_dof
    prop = propagation_params(params, settings)
    U0 = jnp.zeros((H, n_u), dtype=x_prefix.dtype)
    if settings.observability == "full":
        gains, X, U = ilqr_unrolled.solve(env, x_prefix[-1], U0, params, max_iter=max_iter, tol=tol)
        out = dict(mean=X, nominal=X, L=gains.L)
        if settings.model_covariance:
            policy = _plan_policy(gains, X, U, settings.temperature)
            out["Sigma"] = closed_loop_moments(env, policy, prop, X, P0)
        return out

    n = x_prefix.shape[0] - 1
    if n >= 1:
        # belief over the history: the plan the agent made at the start of the window, over the window and the
        # prediction horizon (same goal, same end time), then Algorithm 1 over the window's states
        env_w = _with_start(env, x_prefix[0])
        gw, Xw, Uw = ilqr_unrolled.solve(env_w, x_prefix[0], jnp.zeros((n + H, n_u), dtype=x_prefix.dtype), params,
                                         max_iter=max_iter, tol=tol)
        mu0, S0, P_agent = prefix_belief(env_w, x_prefix, Xw, Uw, _plan_policy(gw, Xw, Uw, settings.temperature),
                                         params, P0)
    else:
        # no history: the belief is the handover state seen through one observation, b0 = x0 + w (initial_belief)
        P_agent = initial_belief(env, x_prefix[-1], params).Sigma
        mu0 = jnp.concatenate([x_prefix[-1], x_prefix[-1]])
        S0 = jnp.block([[P0, P0], [P0, P0 + P_agent]])
    mu_b = mu0[d:]
    env_f = _with_start(env, mu_b)
    gains, X, U = ilqr_unrolled.solve(env_f, mu_b, U0, params, max_iter=max_iter, tol=tol)   # plan from the belief
    policy = _plan_policy(gains, X, U, settings.temperature)
    K = filter_gains(env_f, X, U, params, P_agent)       # the agent's filter along its new plan (its own noises)
    model = SolvedModel(policy, create_filtered_joint_dynamics(env_f, K))
    mu, Sigma = joint_predictive_moments(env_f, model, prop, mu0, S0, H)
    return dict(mean=mu[:, :d], nominal=X, L=gains.L, Sigma=Sigma[:, :d, :d])


_COV_PARTS = ("wrist", "elbow", "passive_wrist")


def _prediction_outputs(env: HumanKinematicReaching, dist: Dict[str, jnp.ndarray], settings: PredictionSettings):
    """Keypoints along the predicted mean and the wrist / elbow / other-wrist covariances J Sigma J^T (model
    covariance) or the FK Jacobians (random walk, propagated on the host)."""
    kp, chest, Jw, Je, Jp = _kinematic_outputs(env, dist["mean"])
    out = dict(mean=dist["mean"], kp=kp, chest=chest, L=dist["L"])
    jac = dict(zip(_COV_PARTS, (Jw, Je, Jp)))
    if settings.random_walk:
        out["jac"] = jac
    if settings.model_covariance:
        sym = lambda C: 0.5 * (C + jnp.swapaxes(C, -1, -2))
        to_cart = lambda J: sym(jnp.einsum("kij,kjl,kml->kim", J, dist["Sigma"], J))
        out["cov"] = {k: to_cart(J) for k, J in jac.items()}
    return out


@partial(jax.jit, static_argnames=("settings", "H", "max_iter", "tol"))
def _predict_one(env, x_prefix, P0, params, settings: PredictionSettings, H: int, max_iter: int,
                 tol: Optional[float] = None):
    """predictive_distribution and its keypoint outputs for one environment (compiled once per hand, settings and
    history length)."""
    return _prediction_outputs(env, predictive_distribution(env, x_prefix, P0, params, settings, H, max_iter, tol),
                               settings)


@partial(jax.jit, static_argnames=("settings", "H", "max_iter", "tol"))
def _predict_batch(envs, x_prefix, P0, params, settings: PredictionSettings, H: int, max_iter: int,
                   tol: Optional[float] = None):
    """_predict_one over a batch of environments (leading axis) and their prefixes x_prefix (B, n+1, d), with the
    same handover covariance P0 (with tol, the batch solves stop when every element has converged)."""
    one = lambda env, xp: _prediction_outputs(env, predictive_distribution(env, xp, P0, params, settings, H,
                                                                            max_iter, tol), settings)
    return jax.vmap(one)(envs, x_prefix)


def _history_prefix(settings: PredictionSettings, q28_history: np.ndarray, body_params: np.ndarray, dt: float,
                    damping: float, x0: np.ndarray, q_chest_ref: np.ndarray, dt_sims: np.ndarray) -> np.ndarray:
    """x_prefix (B, n+1, d) of predictive_distribution for prediction steps dt_sims (B,): the handover state alone
    (fully observed), or the belief window of the estimated history states (partially observed; the same number of
    steps for the whole batch: compiled shapes)."""
    x0 = np.asarray(x0, dtype=np.float32)
    if settings.observability == "full":
        return np.broadcast_to(x0, (len(dt_sims), 1, len(x0))).copy()
    q_hist = upper_body_dofs(np.asarray(q28_history, dtype=np.float64), body_params, q_chest_ref)
    states = prefix_states(q_hist, dt, x0, damping)
    n = min(belief_steps_available(len(q_hist), dt, float(s), settings.belief_steps) for s in dt_sims)
    return np.stack([belief_window(states, dt, float(s), n) for s in dt_sims]).astype(np.float32)


@dataclass
class KinematicPrediction:
    """Kinematic prediction on the model time grid [0, t_pred] (H+1 steps).

    Position covariances (H+1, 3, 3) of the reaching wrist ("wrist"), its elbow ("elbow") and the other wrist
    ("passive_wrist"):
        cov_model: the model's predictive covariance (PredictionSettings covariance "model");
        cov_init, cov_unit: the random-walk ablation, cov_init + pred_noise^2 * cov_unit: cov_init propagates the
            Kalman covariance at the handover, cov_unit a unit-intensity random walk of the joint velocities (variance
            dt per step and per joint, positions integrated), both through the closed loop A + B L of the plan and the
            FK Jacobians; pred_noise (joint-velocity noise, units/s/sqrt(s)) is calibrated by train.py.
    cov(pred_noise) selects: None -> cov_model, a number -> the random walk with that noise level.
    """
    joints: Dict[str, np.ndarray]
    t_pred: float
    dt_sim: float
    cov_model: Optional[Dict[str, np.ndarray]] = None
    cov_init: Optional[Dict[str, np.ndarray]] = None
    cov_unit: Optional[Dict[str, np.ndarray]] = None

    def cov(self, pred_noise: Optional[float] = None) -> Dict[str, np.ndarray]:
        if pred_noise is None:
            if self.cov_model is None:
                raise ValueError("no model covariance in this prediction (PredictionSettings covariance "
                                 "random_walk): give pred_noise")
            return self.cov_model
        if self.cov_init is None:
            raise ValueError("no random-walk covariance in this prediction (PredictionSettings covariance model): "
                             "use pred_noise=None")
        return {k: self.cov_init[k] + pred_noise ** 2 * self.cov_unit[k] for k in self.cov_init}


def _random_walk_covariances(L: np.ndarray, jac: Dict[str, np.ndarray], P0: np.ndarray, dts: np.ndarray,
                             damping: float = 0.0) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """cov_init and cov_unit (B, H+1, 3, 3) of a batch (gains L (B, H, 19, 38), FK Jacobians (B, H+1, 3, 38), time
    steps dts (B,)): P0 and a unit joint-velocity random walk through the closed loop A + B L of the model's ZOH step
    (damping b)."""
    B, H = L.shape[:2]
    n = 19
    A = np.stack([zoh_state_matrices(dt, float(damping), n, xp=np)[0] for dt in dts])
    a2, a1 = (np.array([zoh_coefficients(dt, float(damping), xp=np)[k] for dt in dts]) for k in (2, 1))
    W1 = np.stack([zoh_process_cov(dt, n, xp=np) for dt in dts])   # unit random walk of the joint velocities
    S_init = [np.broadcast_to(np.asarray(P0, dtype=np.float64), (B, 2 * n, 2 * n))]
    S_unit = [np.zeros((B, 2 * n, 2 * n))]
    for k in range(H):
        F = A.copy()
        Lk = np.asarray(L[:, k], dtype=np.float64)
        F[:, :n, :] += a2[:, None, None] * Lk            # A + B L, B = [a2 I; a1 I] (ZOH)
        F[:, n:, :] += a1[:, None, None] * Lk
        S_init.append(F @ S_init[-1] @ F.transpose(0, 2, 1))
        S_unit.append(F @ S_unit[-1] @ F.transpose(0, 2, 1) + W1)
    S_init, S_unit = np.stack(S_init, axis=1), np.stack(S_unit, axis=1)   # (B, H+1, 38, 38)
    to_cart = lambda J, S: np.einsum("bkij,bkjl,bkml->bkim", np.asarray(J, dtype=np.float64), S,
                                     np.asarray(J, dtype=np.float64))
    return {k: to_cart(J, S_init) for k, J in jac.items()}, {k: to_cart(J, S_unit) for k, J in jac.items()}


def _assemble(out, P0: np.ndarray, t_preds: np.ndarray, settings: PredictionSettings, damping: float = 0.0
              ) -> List[KinematicPrediction]:
    """KinematicPrediction of each element of a batch of _predict_batch outputs (leading batch axis)."""
    out = jax.device_get(out)   # one transfer
    kp, chest = np.asarray(out["kp"], dtype=np.float64), np.asarray(out["chest"], dtype=np.float64)
    B, H = np.shape(out["L"])[:2]
    dts = np.asarray(t_preds, dtype=np.float64) / H
    c_init = c_unit = None
    if settings.random_walk:
        c_init, c_unit = _random_walk_covariances(np.asarray(out["L"]), out["jac"], P0, dts, damping)
    c_model = None
    if settings.model_covariance:
        c_model = {k: np.asarray(v, dtype=np.float64) for k, v in out["cov"].items()}
    pelvis = 0.5 * (kp[:, :, hkm.KP_INDEX["left_hip"]] + kp[:, :, hkm.KP_INDEX["right_hip"]])
    pick = lambda c, b: None if c is None else {k: v[b] for k, v in c.items()}
    preds = []
    for b in range(B):
        joints = {j: chest[b] if j == "chest" else pelvis[b] if j == "pelvis" else kp[b, :, hkm.KP_INDEX[j]]
                  for j in JOINTS}
        preds.append(KinematicPrediction(joints, float(t_preds[b]), float(dts[b]), cov_model=pick(c_model, b),
                                         cov_init=pick(c_init, b), cov_unit=pick(c_unit, b)))
    return preds


def predict_motion(q28_history: np.ndarray, dt: float, body_params: np.ndarray, target: np.ndarray,
                   params: HumanKinematicParams, H: int, max_iter: int, t_max: float, hand: Optional[str] = None,
                   legs_nominal: Optional[np.ndarray] = None, tol: Optional[float] = None,
                   settings: Optional[PredictionSettings] = None,
                   target_left: Optional[np.ndarray] = None,
                   alpha_right: Optional[float] = None,
                   alpha_left: Optional[float] = None) -> Tuple[KinematicPrediction, str]:
    """One complete prediction from a 28-DOF joint history (n, 28) sampled every dt (the last frame is the
    prediction start) to a known target, solved up to the expected arrival (offline evaluation; online with goal
    inference: predict_hypotheses): arrival-time estimate, Kalman-filtered handover state, environment, policy and
    predictive distribution (predictive_distribution with `settings`, default PredictionSettings(): fully observed,
    model covariance), keypoints and covariances. hand: reaching hand ("right" / "left"; None = the wrist that moved
    most over the history); legs_nominal: leg joints (8,) of the model (None = those of the first frame); tol: early
    stopping of the solver (None = exactly max_iter iterations). Returns the prediction and the reaching hand."""
    settings = settings or PredictionSettings()
    q28_history = np.asarray(q28_history, dtype=np.float32)
    body = jnp.asarray(body_params, dtype=jnp.float32)
    kp_hist = np.array(_fk_batch(jnp.asarray(q28_history), body))
    if hand is None:
        disp = {s: np.linalg.norm(kp_hist[-1, hkm.KP_INDEX[f"{s}_wrist"]] - kp_hist[0, hkm.KP_INDEX[f"{s}_wrist"]])
                for s in ("right", "left")}
        hand = "right" if disp["right"] >= disp["left"] else "left"
    legs = q28_history[0][18:26] if legs_nominal is None else np.asarray(legs_nominal)
    wrist_obs = kp_hist[:, hkm.KP_INDEX[f"{hand if hand != 'both' else 'right'}_wrist"]]
    v_obs = float(np.linalg.norm(sg_velocity(wrist_obs, dt))) if len(wrist_obs) > 1 else 0.0
    t_pred = arrival_time(float(np.linalg.norm(np.asarray(target) - wrist_obs[-1])), v_obs, t_max)
    dt_sim = t_pred / H

    x0, P0, q_chest_ref = handover_state(q28_history, body_params, dt, params.damping)
    env = make_reaching_env(body_params, legs, x0[:19], q_chest_ref, target, dt_sim, hand,
                            target_left=target_left, alpha_right=alpha_right, alpha_left=alpha_left)
    env.x0 = jnp.asarray(x0, dtype=jnp.float32)
    x_prefix = _history_prefix(settings, q28_history, body_params, dt, params.damping, x0, q_chest_ref,
                               np.array([dt_sim]))[0]
    out = _predict_one(env, jnp.asarray(x_prefix), jnp.asarray(P0, dtype=jnp.float32), params, settings, H,
                       max_iter, tol)
    out = jax.tree.map(lambda a: a[None], out)
    return _assemble(out, P0, np.array([t_pred]), settings, params.damping)[0], hand



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
    reached at the horizon (temporary = True). arrival: expected arrival at the goal (s, may exceed the horizon).
    goal: the wrist goal (wrist_goal of the hypothesis' goal location and the grasp offset; None for idle)."""
    hypothesis: Hypothesis
    prediction: KinematicPrediction
    target: np.ndarray
    target_vel: np.ndarray
    arrival: float
    temporary: bool
    goal: Optional[np.ndarray] = None


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
                             np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), 0.05, "any")


def reaching_env_batch(body_params: np.ndarray, legs_nominal: np.ndarray, x0: np.ndarray, q_chest_ref: np.ndarray,
                       targets: np.ndarray, target_vels: np.ndarray, dts: np.ndarray, hands: List[str],
                       targets_left: Optional[np.ndarray] = None,
                       target_vels_left: Optional[np.ndarray] = None,
                       alphas_right: Optional[np.ndarray] = None,
                       alphas_left: Optional[np.ndarray] = None,
                       ) -> HumanKinematicReaching:
    """Batch of B reaching environments (as make_reaching_env, leading axis = hypothesis) from the same state x0
    (38,) to targets (B, 3) with terminal wrist velocities (B, 3), time steps dts (B,) and reaching hands (B,)
    (reaching_hand "any": the hand is a leaf, so both hands share the batch). Built from the pytree leaves
    directly: the constructor evaluates the FK eagerly (~35 ms per environment)."""
    template = _env_template()
    leaves, aux = template.tree_flatten()
    B = len(targets)
    shared = {"q_chest_ref": q_chest_ref / np.linalg.norm(q_chest_ref), "legs_nominal": legs_nominal,
              "body_params": body_params, "q0": x0[:19], "x0": x0}

    # Default per-hypothesis arrays
    r_hand = np.array([1.0 if h == "right" else 0.0 for h in hands], dtype=np.float32)
    a_right = np.asarray(alphas_right if alphas_right is not None else [0.0 if h == "left" else 1.0 for h in hands], dtype=np.float32)
    a_left = np.asarray(alphas_left if alphas_left is not None else [1.0 if h in ("left", "both") else 0.0 for h in hands], dtype=np.float32)

    t_left = targets_left if targets_left is not None else targets
    tv_left = target_vels_left if target_vels_left is not None else target_vels

    per = {
        "dt": dts, "target": targets, "target_vel": target_vels,
        "right_hand": r_hand,
        "target_left": t_left, "target_vel_left": tv_left,
        "alpha_right": a_right, "alpha_left": a_left,
    }
    batch = []
    for name, leaf in zip(_ENV_LEAVES, leaves):
        if name in per:
            value = np.asarray(per[name], dtype=np.float32)
        else:
            value = np.broadcast_to(np.asarray(shared.get(name, leaf), dtype=np.float32),
                                    (B,) + np.shape(leaf)).copy()
        batch.append(jnp.asarray(value))
    return HumanKinematicReaching.tree_unflatten(aux, batch)



def wrist_goal(position: np.ndarray, wrist: np.ndarray, grasp_offset: float = 0.0) -> np.ndarray:
    """Wrist goal for a goal location (e.g. a detected object) at `position` (3,), seen from the current wrist
    position: the point grasp_offset m before it on the line from the wrist to it (the wrist stops short of the
    object it grasps by about the hand length; within grasp_offset of the object, the wrist position itself).
    grasp_offset = 0: the goal location itself, the convention of training, where the goal is the wrist position at
    the end of the reach (CARI v2 has no object positions)."""
    position = np.asarray(position, dtype=np.float64)
    d = position - np.asarray(wrist, dtype=np.float64)
    dist = float(np.linalg.norm(d))
    if grasp_offset <= 0.0 or dist < 1e-9:
        return position
    return position - min(grasp_offset, dist) * d / dist


def predict_hypotheses(q28_history: np.ndarray, dt: float, body_params: np.ndarray, hypotheses: List[Hypothesis],
                       params: HumanKinematicParams, H: int, max_iter: int, horizon: float, nominal_duration: float,
                       legs_nominal: Optional[np.ndarray] = None, stop_time: float = 0.3, tol: Optional[float] = None,
                       settings: Optional[PredictionSettings] = None, grasp_offset: float = 0.0
                       ) -> Tuple[List[HypothesisPrediction], HandState]:
    """Receding-horizon prediction of every hypothesis from a 28-DOF joint history (n, 28) sampled every dt.

    Shared Kalman handover state; per hypothesis, its wrist goal (wrist_goal: the goal location, or grasp_offset m
    before it when the goal locations are object positions), the expected arrival at it (minimum_jerk_remaining_time
    with the wrist speed towards the goal: a hypothesis assumes its reach is under way, or starts now) and the target
    of the optimal-control problem over the prediction horizon (HypothesisPrediction). An idle hypothesis comes to
    rest in stop_time s (free end position of the minimum-jerk stop: p0 + v0 T / 2 + a0 T^2 / 12). All hypotheses are
    predicted together (predictive_distribution with `settings`, default PredictionSettings(); vmap over both hands,
    compiled once per number of hypotheses), with at most max_iter solver iterations and early stopping at tolerance
    tol (None: exactly max_iter)."""
    settings = settings or PredictionSettings()
    q28_history = np.asarray(q28_history, dtype=np.float32)
    body_params = np.asarray(body_params, dtype=np.float32)
    kp_hist = np.array(_fk_batch(jnp.asarray(q28_history), jnp.asarray(body_params)))
    hs = HandState({}, {}, {})
    for side in ("right", "left"):
        w = kp_hist[:, hkm.KP_INDEX[f"{side}_wrist"]].astype(np.float64)
        hs.wrist[side] = w[-1]
        hs.wrist_vel[side], hs.wrist_acc[side] = _wrist_derivatives(w, dt)
    legs = q28_history[0][18:26] if legs_nominal is None else np.asarray(legs_nominal)
    x0, P0, q_chest_ref = handover_state(q28_history, body_params, dt, params.damping)
    specs = []
    for hyp in hypotheses:
        p0, v0, a0 = hs.wrist[hyp.hand], hs.wrist_vel[hyp.hand], hs.wrist_acc[hyp.hand]
        if hyp.idle:
            T = stop_time
            specs.append((p0 + 0.5 * v0 * T + a0 * T ** 2 / 12.0, np.zeros(3), T, T, False, None))
            continue
        goal = wrist_goal(hyp.goal, p0, grasp_offset)
        d = goal - p0
        dist = float(np.linalg.norm(d))
        v_towards = max(float(v0 @ d) / max(dist, 1e-6), 0.0)
        arrival = minimum_jerk_remaining_time(dist, v_towards, nominal_duration)
        if arrival <= horizon:
            specs.append((goal, np.zeros(3), arrival, arrival, False, goal))
        else:
            target, target_vel = minimum_jerk_state(p0, v0, a0, goal, arrival, horizon)
            specs.append((target, target_vel, horizon, arrival, True, goal))

    targets, target_vels, t_preds, arrivals, temporary = (np.array(v) for v in list(zip(*specs))[:5])
    goals = [s[5] for s in specs]
    envs = reaching_env_batch(body_params, legs, x0, q_chest_ref, targets, target_vels, t_preds / H,
                              [h.hand for h in hypotheses])
    x_prefix = _history_prefix(settings, q28_history, body_params, dt, params.damping, x0, q_chest_ref, t_preds / H)
    out = _predict_batch(envs, jnp.asarray(x_prefix), jnp.asarray(P0, dtype=jnp.float32), params, settings, H,
                         max_iter, tol)
    preds = _assemble(out, P0, t_preds, settings, params.damping)
    out = [HypothesisPrediction(h, p, targets[i], target_vels[i], float(arrivals[i]), bool(temporary[i]), goals[i])
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
    pred_noise: None = the predictions' model covariance, else the random-walk covariance with this noise level
    (KinematicPrediction.cov).
    """

    def __init__(self, hypotheses: List[Hypothesis], pred_noise: Optional[float], switch_rate: float = 0.5,
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


def sample_prediction(pred: KinematicPrediction, times: np.ndarray, pred_noise: Optional[float] = None
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Joints {joint: (n, 3)} and wrist / elbow covariances {part: (n, 3, 3)} of a prediction at the times (n,) after
    the last observation; after the expected arrival (pred.t_pred) the final posture and covariance are held.
    pred_noise: None = model covariance, else the random walk with this level (KinematicPrediction.cov)."""
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
