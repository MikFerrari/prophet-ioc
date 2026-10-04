"""Glue between CARI v2 trials and the 19-DOF HumanKinematicReaching predictor, shared by train.py and eval.py.

Conventions (the configuration that predicts best on CARI v2):
- every keypoint, ground truth and target in the model's frame: FK of the (Savitzky-Golay filtered) IK angles;
- the reaching target is the wrist position at the end of the reach (offset of the wrist speed profile);
- handover: Kalman-filtered 19-DOF joint state at t_obs from the observed joint history, with the chest rotation
  vector relative to the chest orientation at t_obs (so the trunk angular velocity is kept);
- running cost scaled by dt (HumanKinematicReaching dt_scaled_cost): the weights mean the same on any time grid,
  which lets weights fitted on one grid (train.py) be used on the prediction grid (eval.py);
- goal-directed baselines only get the reaching-wrist target, like the kinematic model (no per-keypoint goals).

IOC fit (`fit_params`): on the complete reaches of the training subjects (segments from several handover points to
the end of each reach), either the open-loop keypoint prediction error ("open_loop") or the one-step gILQR
likelihood with a likelihood-only residual noise ("likelihood") is optimized over the cost weights, with parallel
restarts of projected Adam in log10 space; the held-out test subjects (data.test_subjects) are only used by eval.py.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

import human_kinematic_model_jax as hkm
from prophet_ioc.data import CariDataset, CariTrial
from prophet_ioc.data.cari import sg_upper_body_state
from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams, HumanKinematicReaching, stack_envs
from prophet_ioc.infer import MultiTrialInverseGILQR, MultiTrialTrajectoryMatching
from prophet_ioc import human_prediction as hp
from prophet_ioc.human_prediction import (  # noqa: F401  (re-exported for train.py / eval.py / tests)
    CHI2_3_95, JOINTS, KinematicPrediction, JointKinematicsEnv, arrival_time, coverage_fraction,
    kalman_filter_joint_history, sg_velocity, solve_kinematic, to_timeline as to_gt_timeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_jax(device: str) -> jax.Device:
    """Exact float32 matmuls on the GPU (the default precision there is TF32), compiled programs cached in
    <repo>/jax_cache, and the default device ("cpu" or "gpu", falling back to the first available)."""
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(REPO_ROOT / "jax_cache"))
    try:
        return jax.devices(device)[0]
    except RuntimeError:
        print(f"no {device} device, using {jax.devices()[0]}")
        return jax.devices()[0]


def split_subjects(data_cfg) -> Tuple[List[str], List[str]]:
    """(training subjects, test subjects) of the data config: test = data.test_subjects (held out), training = the
    others; without test subjects, all subjects are both."""
    test = [str(s) for s in (data_cfg.get("test_subjects") or [])]
    train = [str(s) for s in data_cfg.subjects if str(s) not in test]
    return train, (test or train)


def load_trials(data_cfg, subjects: Optional[Sequence[str]] = None) -> List[CariTrial]:
    """The trials selected by the data config (subjects x instructions, one velocity), for `subjects` (default: all
    the subjects of the config)."""
    ds = CariDataset()
    trials = []
    for s in (subjects if subjects is not None else data_cfg.subjects):
        for i in data_cfg.instructions:
            try:
                trials.append(ds.load_trial(subject=s, velocity=data_cfg.velocity, instruction_id=int(i),
                                            v_thresh_ratio=0.12))
            except ValueError as exc:
                print(f"  skipping {s}/inst{i}: {exc}")
    return trials


def model_params(model_cfg) -> HumanKinematicParams:
    """Hand-tuned parameters of the model config (fields not in the config keep the class defaults)."""
    fields = HumanKinematicParams._fields
    return HumanKinematicParams(**{k: float(v) for k, v in model_cfg.items() if k in fields})


def params_to_dict(params: HumanKinematicParams) -> Dict[str, float]:
    return {k: float(v) for k, v in params._asdict().items()}


def save_params(path: Path, params: HumanKinematicParams, info: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params_to_dict(params), **info}, indent=2))


def load_params(path: Path) -> HumanKinematicParams:
    return HumanKinematicParams(**json.loads(Path(path).read_text())["params"])


# =============================================================================
# Trial frames
# =============================================================================
@dataclass
class TrialFrames:
    kp: np.ndarray       # (N, 13, 3) FK of the filtered IK angles, every frame of the trial
    chest: np.ndarray    # (N, 3) chest position
    target: np.ndarray   # (3,) reaching target: wrist at the end of the reach
    hand: str            # reaching hand (dataset metadata)

    def keypoint(self, name: str) -> np.ndarray:
        if name == "chest":
            return self.chest
        if name == "pelvis":
            return 0.5 * (self.kp[:, hkm.KP_INDEX["left_hip"]] + self.kp[:, hkm.KP_INDEX["right_hip"]])
        return self.kp[:, hkm.KP_INDEX[name]]

    def joints(self, idx) -> Dict[str, np.ndarray]:
        """The 9 MPJPE joints at frame(s) idx."""
        return {name: self.keypoint(name)[idx] for name in JOINTS}


_fk_batch = jax.jit(jax.vmap(hkm.fk, in_axes=(0, None)))


def trial_frames(trial: CariTrial) -> TrialFrames:
    kp = np.array(_fk_batch(jnp.asarray(trial.q28_filt), jnp.asarray(trial.body_params)))
    target = kp[trial.offset_idx, hkm.KP_INDEX[f"{trial.reaching_hand}_wrist"]]
    return TrialFrames(kp, np.asarray(trial.q28_filt[:, 0:3]), np.asarray(target, dtype=np.float32),
                       trial.reaching_hand)


def ratio_tag(obs_ratio: float) -> str:
    """Folder name of an observed fraction, e.g. obs30."""
    return f"obs{int(round(obs_ratio * 100)):02d}"


def obs_frame(trial: CariTrial, obs_ratio: float) -> int:
    """Index of the last observed frame (t_obs) for an observed fraction obs_ratio of the reach."""
    return trial.onset_idx + max(int(round(trial.reach_frames * obs_ratio)), 3)


def infer_reaching_hand(frames: TrialFrames, onset: int, f_obs: int) -> str:
    """Hand that moved most during the observation window."""
    disp = {s: np.linalg.norm(frames.keypoint(f"{s}_wrist")[f_obs] - frames.keypoint(f"{s}_wrist")[onset])
            for s in ("right", "left")}
    return "right" if disp["right"] >= disp["left"] else "left"


# =============================================================================
# Prediction (prophet_ioc.human_prediction) on CARI trials
# =============================================================================
def handover_state(trial: CariTrial, f_obs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman-filtered state at t_obs from the observation window [onset, t_obs] (see hp.handover_state)."""
    return hp.handover_state(trial.q28_filt[trial.onset_idx: f_obs + 1], trial.body_params, trial.dt)


def make_env(trial: CariTrial, q0: np.ndarray, q_chest_ref: np.ndarray, target: np.ndarray, dt: float, hand: str,
             params: HumanKinematicParams) -> HumanKinematicReaching:
    return hp.make_reaching_env(trial.body_params, trial.legs_nominal, q0, q_chest_ref, target, dt, hand, params)


def kinematic_inference(trial: CariTrial, frames: TrialFrames, f_obs: int, params: HumanKinematicParams, H: int,
                        max_iter: int, hand: str) -> KinematicPrediction:
    """One complete prediction from the observation window [onset, f_obs], as it runs online (hp.predict_motion),
    with the remaining duration of the reach as the upper bound of the arrival time."""
    pred, _ = hp.predict_motion(trial.q28_filt[trial.onset_idx: f_obs + 1], trial.dt, trial.body_params,
                                frames.target, params, H, max_iter, t_max=(trial.offset_idx - f_obs) * trial.dt,
                                hand=hand, legs_nominal=trial.legs_nominal)
    return pred


def prediction_error_samples(trial: CariTrial, obs_ratio: float, params: HumanKinematicParams, H: int,
                             max_iter: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrist and elbow errors (n, 3) of the kinematic prediction from t_obs (as eval.py: ground-truth timeline, H steps
    after the handover) and the two covariance terms (n, 3, 3) of KinematicPrediction."""
    frames = trial_frames(trial)
    f_obs = obs_frame(trial, obs_ratio)
    hand = infer_reaching_hand(frames, trial.onset_idx, f_obs)
    pred = kinematic_inference(trial, frames, f_obs, params, H, max_iter, hand)
    t_rem = (trial.offset_idx - f_obs) * trial.dt
    fut = np.round(np.linspace(f_obs, trial.offset_idx, H + 1)).astype(int)[1:]
    e, ci, cu = [], [], []
    for part in ("wrist", "elbow"):
        j = f"{trial.reaching_hand}_{part}"
        e.append(frames.keypoint(j)[fut] - to_gt_timeline(pred.joints[j], pred.t_pred, t_rem)[1:])
        ci.append(to_gt_timeline(pred.cov_init[part], pred.t_pred, t_rem)[1:])
        cu.append(to_gt_timeline(pred.cov_unit[part], pred.t_pred, t_rem)[1:])
    return np.concatenate(e), np.concatenate(ci), np.concatenate(cu)


def _noise_for_coverage(e, ci, cu, target: float) -> float:
    """Smallest noise level whose 95 % ellipsoids contain a fraction `target` of the errors (log10 bisection)."""
    frac = lambda sigma: float(np.mean(coverage_fraction(e, ci + sigma ** 2 * cu)))
    lo, hi = -4.0, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if frac(10 ** mid) < target else (lo, mid)
    return float(10 ** hi)


def calibrate_pred_noise(trials: Sequence[CariTrial], obs_ratio: float, params: HumanKinematicParams, H: int,
                         max_iter: int, target: float = 0.95) -> Tuple[float, Dict]:
    """Joint-velocity noise level of the prediction covariance such that a fraction `target` of the wrist and elbow
    prediction errors fall inside the 95 % ellipsoids, at the evaluated horizon (predictions from t_obs).

    The level of each subject is calibrated on the other subjects' trials only (leave-one-subject-out), so the future
    of the evaluated trials is never used; "pred_noise" (all subjects) is the value for a new person. A calibration on
    the observed prefix only does not transfer: there the predictions start near rest with an unconverged handover
    filter and short horizons, and the handover covariance alone already covers the errors (noise -> 0), while at
    t_obs the errors grow with the horizon (model bias)."""
    samples = {}
    for trial in trials:
        samples.setdefault(trial.subject, []).append(prediction_error_samples(trial, obs_ratio, params, H, max_iter))
    pooled = lambda subjects: tuple(np.concatenate([x[i] for s in subjects for x in samples[s]]) for i in range(3))
    by_subject = {s: _noise_for_coverage(*pooled([o for o in samples if o != s]), target) for s in samples}
    e, ci, cu = pooled(list(samples))
    sigma = _noise_for_coverage(e, ci, cu, target)
    return sigma, {"pred_noise": sigma, "pred_noise_by_subject": by_subject, "calibration": "leave-one-subject-out",
                   "calibration_samples": int(len(e)),
                   "coverage_handover_covariance_only": float(np.mean(coverage_fraction(e, ci)))}


# =============================================================================
# Goal-directed baselines
# =============================================================================
def predict_minimum_jerk_stop(obs_pos: np.ndarray, fut_times: np.ndarray, dt: float) -> np.ndarray:
    """Minimum jerk with a free end position: from the observed position, velocity and acceleration (Savitzky-Golay,
    as predict_minimum_jerk) to rest (zero velocity and acceleration) at the end of the horizon, wherever that is.
    With x(T) free, the optimality condition x^(5)(T) = 0 removes the 5th-order term: a quartic in tau = t / T with
    c3 = -c1 - 4/3 c2, c4 = (c1 + c2) / 2 (c1 = v0 T, c2 = a0 T^2 / 2)."""
    obs_pos = np.asarray(obs_pos)
    n = len(obs_pos)
    w = min(7, n if n % 2 == 1 else n - 1)
    if w >= 5:
        v0 = savgol_filter(obs_pos, w, 2, deriv=1, delta=dt, axis=0)[-1]
        a0 = savgol_filter(obs_pos, w, 2, deriv=2, delta=dt, axis=0)[-1]
    else:
        v0 = (obs_pos[-1] - obs_pos[-2]) / dt if n >= 2 else np.zeros(obs_pos.shape[-1])
        a0 = np.zeros(obs_pos.shape[-1])
    T = max(fut_times[-1] - fut_times[0], 1e-6)
    tau = np.clip((fut_times - fut_times[0]) / T, 0.0, 1.0)[:, None]
    c1, c2 = v0 * T, 0.5 * a0 * T ** 2
    c3, c4 = -c1 - 4.0 / 3.0 * c2, 0.5 * (c1 + c2)
    return obs_pos[-1] + c1 * tau + c2 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4


def predict_goal_directed(kind: str, obs_traj: np.ndarray, goal: Optional[np.ndarray], fut_times: np.ndarray,
                          dt: float) -> np.ndarray:
    """Min-jerk ("minjerk") or goal-directed constant velocity ("gcv") towards goal. A keypoint without a goal (every
    keypoint but the reaching wrist) continues at constant velocity (gcv) or comes smoothly to rest with a free end
    position (minjerk, predict_minimum_jerk_stop)."""
    from prophet_ioc.infer import predict_constant_velocity, predict_goal_directed_cv, predict_minimum_jerk

    if goal is None:
        if kind == "minjerk":
            return predict_minimum_jerk_stop(obs_traj, fut_times, dt)
        return predict_constant_velocity(obs_traj, fut_times, dt)
    if kind == "minjerk":
        return predict_minimum_jerk(obs_traj, goal, fut_times, dt)
    if kind == "gcv":
        return predict_goal_directed_cv(obs_traj[-1], goal, fut_times)
    raise ValueError(f"Unknown goal-directed baseline {kind}")


# =============================================================================
# IOC fit on the observed prefix of each trial
# =============================================================================
# Inferred parameters per objective. The noises of the controller (motor_noise, motor_noise_add) are not fitted:
# fitted to the one-step residuals, which are mostly model mismatch, the motor noise goes to its upper bound, and
# the controller, which plans against it, then stops short of the target.
COST_PARAMS = ("action_cost", "posture_cost", "base_disp_cost", "running_vel_cost", "velocity_cost",
               "w_act_trunk", "w_act_spine", "w_act_passive_arm", "w_act_head", "running_target_cost")
FIT_PARAMS = {"open_loop": COST_PARAMS, "likelihood": COST_PARAMS + ("residual_noise",)}
# Trials per batch inside a restart (lax.map): the likelihood gradient carries the noise Jacobians of the gILQR
# linearization and needs much more memory than the open-loop one (16 or 4 trials x 4 restarts ran out of 8 GB)
DEFAULT_BATCH = {"open_loop": 16, "likelihood": 2}

_UPPER_BODY_KP = jnp.array([hkm.KP_INDEX[n] for n in ("head", "left_shoulder", "left_elbow", "left_wrist",
                                                      "right_shoulder", "right_elbow", "right_wrist")])


_HIPS = jnp.array([hkm.KP_INDEX["left_hip"], hkm.KP_INDEX["right_hip"]])


def upper_body_output(env: HumanKinematicReaching, state: jnp.ndarray) -> jnp.ndarray:
    """The 9 MPJPE joints (head, shoulders, elbows, wrists, chest, pelvis = hip midpoint; 9 x 3, flattened): the space
    of the open-loop objective."""
    kp = env.all_keypoints(state)
    return jnp.concatenate([kp[_UPPER_BODY_KP], env.chest(state)[None], kp[_HIPS].mean(axis=0)[None]]).ravel()


def fit_segments(trial: CariTrial, params: HumanKinematicParams, starts: Sequence[float], T_fit: int
                 ) -> List[Tuple[np.ndarray, np.ndarray, HumanKinematicReaching]]:
    """Training segments of a complete reach (a demonstration of a training subject).

    For each start (fraction of the reach, e.g. the handover points of eval.py), the joint state from the start to
    the end of the reach on T_fit steps (dt = remaining time / T_fit), as a prediction from that start would be
    computed: positions and velocities are Savitzky-Golay estimates over the whole reach (sg_upper_body_state), the
    chest rotation vector is relative to the start orientation, the target is the wrist at the end of the reach.

    Returns a list of (x (T_fit+1, 38), mask (T_fit+1,) of ones, env).
    """
    target = trial_frames(trial).target
    segments = []
    for s in starts:
        s_idx = trial.onset_idx + int(round(s * (trial.offset_idx - trial.onset_idx)))
        q_ref = trial.q28_filt[s_idx][3:7]
        q, qd = sg_upper_body_state(trial, q_ref)
        t_grid = np.linspace(s_idx, trial.offset_idx, T_fit + 1)
        x = np.concatenate([q, qd], axis=1)
        x = np.stack([np.interp(t_grid, np.arange(len(q)), x[:, j]) for j in range(x.shape[1])], axis=1)
        dt = (trial.offset_idx - s_idx) * trial.dt / T_fit
        env = make_env(trial, x[0, :19], q_ref, target, dt, trial.reaching_hand, params)
        segments.append((x.astype(np.float32), np.ones(T_fit + 1, dtype=np.float32), env))
    return segments


def _groups_by_hand(segments):
    groups = []
    for hand in ("right", "left"):
        seg = [s for s in segments if s[2].reaching_hand == hand]
        if seg:
            groups.append((stack_envs([s[2] for s in seg]), jnp.asarray(np.stack([s[0] for s in seg])),
                           jnp.asarray(np.stack([s[1] for s in seg]))))
    return groups


def fit_params(trials: Sequence[CariTrial], base_params: HumanKinematicParams, ioc_cfg
               ) -> Tuple[HumanKinematicParams, Dict]:
    """IOC fit of the cost weights on the complete reaches of `trials` (fit_segments).

    ioc_cfg: objective ("open_loop" | "likelihood"), params (inferred names, null = FIT_PARAMS[objective]),
    T_fit, segment_starts, restarts, max_iter, lr, patience, tol, batch_size, seed. The parameters not inferred
    keep their base_params values. Restart 0 starts from base_params, the others uniformly at random in log10 space
    within HumanKinematicReaching.get_params_bounds(); all restarts run in parallel (projected Adam in log10 space,
    one batched gradient per iteration), with a progress bar. Returns the best point visited.
    """
    objective = ioc_cfg.objective
    infer = tuple(ioc_cfg.params) if ioc_cfg.get("params") else FIT_PARAMS[objective]
    segments = [seg for tr in trials for seg in fit_segments(tr, base_params, ioc_cfg.segment_starts, ioc_cfg.T_fit)]
    groups = _groups_by_hand(segments)
    batch = ioc_cfg.get("batch_size") or DEFAULT_BATCH[objective]
    if objective == "open_loop":
        ioc = MultiTrialTrajectoryMatching(groups, base_params, infer, output_fn=upper_body_output, batch_size=batch)
    elif objective == "likelihood":
        ioc = MultiTrialInverseGILQR(groups, base_params, infer, velocity_block=slice(19, 38), batch_size=batch)
    else:
        raise ValueError(f"objective must be 'open_loop' or 'likelihood', got {objective}")
    n_seg = len(segments)

    def to_params(theta):
        return ioc.full_params(base_params._replace(**{k: 10.0 ** theta[i] for i, k in enumerate(infer)}))

    loss = lambda theta: -ioc.loglikelihood(None, to_params(theta)) / n_seg   # per segment, for the tolerance
    lo, hi = HumanKinematicReaching.get_params_bounds()
    lo = jnp.log10(jnp.array([getattr(lo, k) for k in infer]))
    hi = jnp.log10(jnp.array([getattr(hi, k) for k in infer]))
    theta_base = jnp.clip(jnp.log10(jnp.array([max(getattr(base_params, k), 1e-12) for k in infer])), lo, hi)
    key = jax.random.PRNGKey(ioc_cfg.seed)
    theta0 = jax.random.uniform(key, (ioc_cfg.restarts, len(infer)), minval=lo, maxval=hi).at[0].set(theta_base)

    # Projected Adam in log10 space, all restarts in one batched value-and-gradient call per iteration (one
    # moderate compiled program; a line-search optimizer such as jaxopt.LBFGSB traces the objective several times
    # inside its loops, which took minutes to compile and ~15 s per iteration here)
    value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(loss)))
    lr, b1, b2, eps = float(ioc_cfg.lr), 0.9, 0.999, 1e-8
    m = jnp.zeros_like(theta0)
    v = jnp.zeros_like(theta0)
    theta = theta0
    best_theta, best_val = np.array(theta0), np.full(ioc_cfg.restarts, np.inf)
    history, history_restarts, loss_base = [], [], None

    t_start = time.perf_counter()
    print(f"  compiling the {objective} objective ({n_seg} segments, {len(infer)} parameters, "
          f"{ioc_cfg.restarts} parallel restarts)...", flush=True)
    bar = tqdm(range(1, ioc_cfg.max_iter + 1), desc=f"IOC fit ({objective})", unit="it")
    for it in bar:
        val, g = value_and_grad(theta)
        val, g = np.array(val), np.array(g)
        if loss_base is None:
            loss_base = float(val[0])
        ok = np.isfinite(val) & np.all(np.isfinite(g), axis=1)
        better = ok & (val < best_val)
        best_val[better], best_theta[better] = val[better], np.array(theta)[better]
        g = jnp.asarray(np.where(ok[:, None], g, 0.0))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        step = lr * (m / (1 - b1 ** it)) / (jnp.sqrt(v / (1 - b2 ** it)) + eps)
        # restarts with a non-finite loss go back to their best point (or to the model parameters)
        fallback = np.where(np.isfinite(best_val)[:, None], best_theta, np.array(theta_base)[None])
        theta = jnp.where(jnp.asarray(~ok)[:, None], jnp.asarray(fallback), jnp.clip(theta - step, lo, hi))
        history.append(float(np.min(best_val)))
        history_restarts.append(np.where(np.isfinite(val), val, np.nan).tolist())
        bar.set_postfix(best=f"{history[-1]:.4g}", base=f"{loss_base:.4g}")
        if it > ioc_cfg.patience and history[-ioc_cfg.patience - 1] - history[-1] < ioc_cfg.tol * abs(history[-1]):
            break  # the best loss improved by less than tol (relative) over the last `patience` iterations
    bar.close()
    values = best_val
    values = np.where(np.isfinite(values), values, np.inf)
    best = int(np.argmin(values))
    fitted = to_params(jnp.asarray(best_theta[best]))
    fitted = HumanKinematicParams(**params_to_dict(fitted))  # plain floats
    info = {"objective": objective, "infer": list(infer), "T_fit": ioc_cfg.T_fit,
            "segment_starts": list(ioc_cfg.segment_starts), "n_trials": len(trials), "n_segments": n_seg,
            "loss_base": loss_base, "loss_fit": float(values[best]), "loss_restarts": values.tolist(),
            "best_restart": best, "iterations": len(history), "loss_history": history,
            "loss_history_restarts": history_restarts,
            "theta_restarts": {k: (10.0 ** best_theta[:, i]).tolist() for i, k in enumerate(infer)},
            "fit_time_s": time.perf_counter() - t_start,
            "trials": [f"{tr.subject}/{tr.velocity}/inst{tr.instruction_id}" for tr in trials]}
    return fitted, info


def segment_errors(trials: Sequence[CariTrial], params: HumanKinematicParams, ioc_cfg) -> Dict[str, np.ndarray]:
    """Open-loop prediction error of every training segment (fit_segments) with `params`: RMS over the 9 joints and
    the T_fit steps (cm), with the subject, instruction and start of each segment."""
    from prophet_ioc.infer.multi_env import trial_open_loop_error
    rows, errs = [], []
    for tr in trials:
        for start, seg in zip(ioc_cfg.segment_starts, fit_segments(tr, params, ioc_cfg.segment_starts, ioc_cfg.T_fit)):
            rows.append((tr.subject, tr.instruction_id, float(start), seg))
    for hand in ("right", "left"):
        idx = [i for i, r in enumerate(rows) if r[3][2].reaching_hand == hand]
        if not idx:
            continue
        envs = stack_envs([rows[i][3][2] for i in idx])
        xs = jnp.asarray(np.stack([rows[i][3][0] for i in idx]))
        f = jax.jit(lambda e, x: jax.lax.map(lambda ex: trial_open_loop_error(ex[0], ex[1], params, upper_body_output),
                                             (e, x), batch_size=16))
        err = np.array(f(envs, xs))
        errs += list(zip(idx, err))
    err = np.zeros(len(rows))
    for i, e in errs:
        err[i] = e
    return {"rms_cm": 100.0 * np.sqrt(err / 9.0), "subject": np.array([r[0] for r in rows]),
            "instruction": np.array([r[1] for r in rows]), "start": np.array([r[2] for r in rows])}
