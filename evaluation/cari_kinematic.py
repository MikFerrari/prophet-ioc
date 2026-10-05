"""Glue between CARI v2 trials and the 19-DOF HumanKinematicReaching predictor, shared by train.py and eval.py.

Conventions (the configuration that predicts best on CARI v2):
- every keypoint, ground truth and target in the model's frame: FK of the (dataset-filtered) IK angles;
- the reaching target is the wrist position at the end of the reach (offset of the wrist speed profile);
- handover: Kalman-filtered 19-DOF joint state at t_obs from the observed joint history, with the chest rotation
  vector relative to the chest orientation at t_obs (so the trunk angular velocity is kept);
- running cost scaled by dt (HumanKinematicReaching dt_scaled_cost): the weights mean the same on any time grid,
  which lets weights fitted on one grid (train.py) be used on the prediction grid (eval.py);
- goal-directed baselines only get the reaching-wrist target, like the kinematic model (no per-keypoint goals).

IOC fit (`fit_params`): on the complete reaches of the training subjects (training windows from several handover
points to the end of each reach, joint states from the RTS smoother under the model's dynamics or Savitzky-Golay,
data.state_estimator), the negative log-likelihood of the probabilistic IOC model ("likelihood": policy solved per
window, fully or partially observed, prophet_ioc.infer.multi_env) or the open-loop keypoint prediction error
("open_loop", ablation) is minimized over the learnable cost weights (HumanKinematicParams.LEARNABLE, minus the
switched-off terms), with parallel restarts of projected Adam in log10 space; the held-out test subjects
(data.test_subjects) are only used by eval.py.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

import human_kinematic_model_jax as hkm
from prophet_ioc.data import CariDataset, CariTrial
from prophet_ioc.data import cari
from prophet_ioc.data.cari import INSTRUCTION_METADATA
from prophet_ioc.data.cari import RTS_ACCEL_NOISE, RTS_OBS_NOISE, rts_upper_body_state, sg_upper_body_state
from prophet_ioc.envs.human_kinematic_reaching import (HumanKinematicParams, HumanKinematicReaching, learnable_params,
                                                       params_from_config, stack_envs)
from prophet_ioc.infer import MultiTrialLikelihood, MultiTrialTrajectoryMatching
from prophet_ioc import human_prediction as hp
from prophet_ioc.human_prediction import (  # noqa: F401  (re-exported for train.py / eval.py / tests)
    CHI2_3_95, JOINTS, KinematicPrediction, JointKinematicsEnv, PredictionSettings, arrival_time, coverage_fraction,
    kalman_filter_joint_history, prediction_settings, sg_velocity, solve_kinematic, solver_tolerance,
    to_timeline as to_gt_timeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hydra_argparse_compat() -> None:
    """Python >= 3.14 argparse validates help strings when an argument is added; Hydra 1.3 passes a lazy, non-string
    help object (--shell-completion), so @hydra.main fails with "badly formed help string" before the script starts.
    Skip the check for non-string help (train.py, eval.py and goal_inference.py import this module first)."""
    import argparse
    check = getattr(argparse.ArgumentParser, "_check_help", None)
    if check is None or getattr(check, "_hydra_compat", False):
        return

    def _check_help(self, action):
        if action.help is None or isinstance(action.help, str):
            check(self, action)

    _check_help._hydra_compat = True
    argparse.ArgumentParser._check_help = _check_help


_hydra_argparse_compat()


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


def load_trials(data_cfg, subjects: Optional[Sequence[str]] = None, role: str = "") -> List[CariTrial]:
    """The trials selected by the data config (subjects x instructions, one or multiple velocities), for `subjects`
    (default: all the subjects of the config). data.head_keypoint selects the dataset (prophet_ioc.data.cari
    HEAD_KEYPOINT_CACHES) and becomes the default of every CariDataset of the process (cari_sessions).
    Prints the selection and how the number of trials adds up (requested, skipped by reason, kept); role labels the
    summary (e.g. "training", "test")."""
    cari.HEAD_KEYPOINT = str(getattr(data_cfg, "head_keypoint", None) or cari.HEAD_KEYPOINT)
    ds = CariDataset()
    trials = []
    skipped = {}   # reason -> ["sub/instI/VEL", ...]
    raw = getattr(data_cfg, "velocities", getattr(data_cfg, "velocity", "FAST"))
    vels = [raw] if isinstance(raw, str) else list(raw)
    subjects = list(subjects if subjects is not None else data_cfg.subjects)
    instructions = [int(i) for i in data_cfg.instructions]
    for vel in vels:
        for s in subjects:
            for i in instructions:
                try:
                    tr = ds.load_trial(subject=s, velocity=str(vel), instruction_id=int(i), v_thresh_ratio=0.12)
                except ValueError as exc:
                    print(f"  skipping {s}/inst{i}/{vel}: {exc}")
                    skipped.setdefault("not loadable", []).append(f"{s}/inst{i}/{vel}")
                    continue
                # IK failures (e.g. sub_3/inst3 SLOW, MEDIUM: 20-30 % of the reach): NaN keypoints for the baselines,
                # a gap bridged by the RTS smoother in the IOC windows, and the wrong reaching hand (NaN wrist
                # displacement in CariDataset.load_trial)
                n_bad = int((~np.isfinite(tr.q28_filt[tr.onset_idx:tr.offset_idx + 1]).all(axis=1)).sum())
                if n_bad:
                    print(f"  skipping {s}/inst{i}/{vel}: {n_bad} of {tr.offset_idx - tr.onset_idx + 1} reach frames "
                          f"without joint angles (IK failure)")
                    skipped.setdefault("NaN joint angles (IK failure)", []).append(f"{s}/inst{i}/{vel}")
                    continue
                trials.append(tr)
    n_req = len(subjects) * len(instructions) * len(vels)
    n_skip = sum(len(v) for v in skipped.values())
    names = lambda ids: ", ".join(f"{i} ({INSTRUCTION_METADATA[i]['name']})" if i in INSTRUCTION_METADATA else str(i)
                                  for i in ids)
    print(f"--- {role + ' ' if role else ''}data: head keypoint {cari.HEAD_KEYPOINT}\n"
          f"    instructions: {names(instructions)}\n"
          f"    velocities:   {', '.join(map(str, vels))}\n"
          f"    subjects:     {', '.join(subjects)}\n"
          f"    {len(subjects)} subjects x {len(instructions)} instructions x {len(vels)} velocities = {n_req} reaches"
          + "".join(f"\n    - {len(v):3d} skipped, {reason}: {', '.join(v)}" for reason, v in skipped.items())
          + f"\n    = {len(trials)} reaches kept" + (f" ({n_skip} skipped)" if n_skip else ""), flush=True)
    return trials


def model_params(model_cfg) -> HumanKinematicParams:
    """Parameters of the model config (initial weights and hand-tuned fixed values; fields not in the config keep the
    class defaults; switched-off terms have weight 0): the one place where config/model is turned into parameters
    (prophet_ioc.envs.human_kinematic_reaching.params_from_config, also used by the ROS 2 node)."""
    return params_from_config(dict(model_cfg))


def params_to_dict(params: HumanKinematicParams) -> Dict[str, float]:
    return {k: float(v) for k, v in params._asdict().items()}


def save_params(path: Path, params: HumanKinematicParams, info: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # params last: a "params" entry of info (e.g. a loaded record) must not replace them
    path.write_text(json.dumps({**{k: v for k, v in info.items() if k != "params"}, "params": params_to_dict(params)},
                               indent=2))


def load_record(path) -> Tuple[HumanKinematicParams, Dict, Path]:
    """(parameters, the whole params.json record, its path) of a train.py run folder or params.json."""
    path = Path(path)
    path = path / "params.json" if path.is_dir() else path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run train.py first, or use the initial weights (params=null)")
    return load_params(path), json.loads(path.read_text()), path


def predictor_settings(cfg, record: Optional[Dict] = None) -> PredictionSettings:
    """PredictionSettings of the configuration (model.prediction_covariance, prediction_residual,
    prediction_observability, belief_steps) and of the train.py record of the weights (observability and temperature
    of the fit; None = initial weights: fully observed, ioc.temperature)."""
    return prediction_settings(cfg.model, record, cfg.get("ioc"))


def pred_noise_of(settings: PredictionSettings, record: Optional[Dict], obs_ratio: Optional[float], default: float
                  ) -> Optional[float]:
    """Noise level of the random-walk covariance (KinematicPrediction.cov): None with the model covariance; else the
    level train.py calibrated for obs_ratio (None: the median over the observed fractions, online), or `default`
    (model.pred_noise) for the initial weights."""
    if not settings.random_walk:
        return None
    noise = (record or {}).get("pred_noise", default)
    if not isinstance(noise, dict):
        return float(noise)
    if obs_ratio is None:
        return float(np.median(list(noise.values())))
    return float(noise.get(ratio_tag(obs_ratio), default))


def load_params(path: Path) -> HumanKinematicParams:
    """Parameters of a train.py params.json: the fitted weights and the fixed values (damping, joint-limit weight,
    ...) the fit used."""
    saved = json.loads(Path(path).read_text())["params"]
    unknown = sorted(set(saved) - set(HumanKinematicParams._fields))
    if unknown:
        raise ValueError(f"{path} was written by an older model version (unknown parameters {unknown}; the cost was "
                         "rescaled to a unit terminal weight): run train.py again")
    return HumanKinematicParams(**saved)


# =============================================================================
# Trial frames
# =============================================================================
@dataclass
class TrialFrames:
    kp: np.ndarray       # (N, 13, 3) FK of the filtered IK angles, every frame of the trial
    chest: np.ndarray    # (N, 3) chest position
    target: np.ndarray   # (3,) reaching target (reaching wrist, or right wrist if both)
    hand: str            # reaching hand (dataset metadata)
    target_right: Optional[np.ndarray] = None
    target_left: Optional[np.ndarray] = None

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
    tgt_r = np.asarray(kp[trial.offset_idx, hkm.KP_INDEX["right_wrist"]], dtype=np.float32)
    tgt_l = np.asarray(kp[trial.offset_idx, hkm.KP_INDEX["left_wrist"]], dtype=np.float32)
    tgt = tgt_r if trial.reaching_hand in ("right", "both") else tgt_l
    return TrialFrames(kp, np.asarray(trial.q28_filt[:, 0:3]), tgt,
                       trial.reaching_hand, target_right=tgt_r, target_left=tgt_l)


def ratio_tag(obs_ratio: float) -> str:
    """Folder name of an observed fraction, e.g. obs30."""
    return f"obs{int(round(obs_ratio * 100)):02d}"


def obs_frame(trial: CariTrial, obs_ratio: float) -> int:
    """Index of the last observed frame (t_obs) for an observed fraction obs_ratio of the reach."""
    return trial.onset_idx + max(int(round(trial.reach_frames * obs_ratio)), 3)


def infer_reaching_hand(frames: TrialFrames, onset: int, f_obs: int, allow_both: bool = False) -> str:
    """Predict reaching hand from kinematic motion energy across the entire arm chain (shoulder, elbow, wrist)."""
    energies = {}
    for side in ("right", "left"):
        w = frames.keypoint(f"{side}_wrist")[onset: f_obs + 1]
        e = frames.keypoint(f"{side}_elbow")[onset: f_obs + 1]
        s = frames.keypoint(f"{side}_shoulder")[onset: f_obs + 1]
        if len(w) < 2:
            disp = {side: float(np.linalg.norm(frames.keypoint(f"{side}_wrist")[f_obs] -
                                               frames.keypoint(f"{side}_wrist")[onset]))
                    for side in ("right", "left")}
            return "right" if disp["right"] >= disp["left"] else "left"
        vw, ve, vs = np.diff(w, axis=0), np.diff(e, axis=0), np.diff(s, axis=0)
        # Weight by segment mass/leverage: wrist (1.0), forearm/elbow (1.5), upper-arm/shoulder (2.0)
        energies[side] = float(np.sum(vw ** 2) + 1.5 * np.sum(ve ** 2) + 2.0 * np.sum(vs ** 2))

    if allow_both:
        e_tot = energies["right"] + energies["left"]
        if e_tot > 1e-6:
            r_ratio = energies["right"] / e_tot
            if 0.35 <= r_ratio <= 0.65:
                return "both"

    return "right" if energies["right"] >= energies["left"] else "left"


# =============================================================================
# Prediction (prophet_ioc.human_prediction) on CARI trials
# =============================================================================
def handover_state(trial: CariTrial, f_obs: int, damping: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman-filtered state at t_obs from the observation window [onset, t_obs] (see hp.handover_state)."""
    return hp.handover_state(trial.q28_filt[trial.onset_idx: f_obs + 1], trial.body_params, trial.dt, damping)


def make_env(trial: CariTrial, q0: np.ndarray, q_chest_ref: np.ndarray, target: np.ndarray, dt: float,
             hand: str, target_left: Optional[np.ndarray] = None,
             alpha_right: Optional[float] = None, alpha_left: Optional[float] = None) -> HumanKinematicReaching:
    return hp.make_reaching_env(trial.body_params, trial.legs_nominal, q0, q_chest_ref, target, dt, hand,
                                target_left=target_left, alpha_right=alpha_right, alpha_left=alpha_left)


def kinematic_inference(trial: CariTrial, frames: TrialFrames, f_obs: int, params: HumanKinematicParams, H: int,
                        max_iter: int, hand: str, tol: Optional[float] = None,
                        settings: Optional[PredictionSettings] = None,
                        reaching_mode: str = "dual") -> KinematicPrediction:
    """One complete prediction from the observation window [onset, f_obs], as it runs online (hp.predict_motion; tol:
    early stopping of the solver; settings: predictive distribution, default hp.PredictionSettings()), with the
    remaining duration of the reach as the upper bound of the arrival time.

    Each wrist gets its own goal: the environment's `target` is the right wrist's, target_left the left wrist's, and
    `hand` switches them on (right / left / both). frames.target is the goal of the dataset's reaching hand, which is
    not the right wrist's when `hand` differs from it (e.g. a left reach predicted with hand="both")."""
    target_right = frames.target_right if frames.target_right is not None else frames.target
    target_left = frames.target_left if (reaching_mode == "dual" or hand == "both") else None
    # predict_motion estimates the arrival time from `target` and the wrist of `hand` (the right one for "both")
    target = frames.target_left if hand == "left" and frames.target_left is not None else target_right
    ar = 1.0 if hand in ("right", "both") else 0.0
    al = 1.0 if hand in ("left", "both") else 0.0
    pred, _ = hp.predict_motion(trial.q28_filt[trial.onset_idx: f_obs + 1], trial.dt, trial.body_params,
                                target, params, H, max_iter, t_max=(trial.offset_idx - f_obs) * trial.dt,
                                hand=hand, legs_nominal=trial.legs_nominal, tol=tol, settings=settings,
                                target_left=target_left, alpha_right=ar, alpha_left=al)
    return pred



def prediction_error_samples(trial: CariTrial, obs_ratio: float, params: HumanKinematicParams, H: int,
                             max_iter: int, tol: Optional[float] = None, settings: Optional[PredictionSettings] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrist and elbow errors (n, 3) of the kinematic prediction from t_obs (as eval.py: ground-truth timeline, H steps
    after the handover), the two random-walk covariance terms and the model covariance (n, 3, 3) of
    KinematicPrediction (settings with covariance "both")."""
    settings = replace(settings or PredictionSettings(), covariance="both")
    frames = trial_frames(trial)
    f_obs = obs_frame(trial, obs_ratio)
    hand = trial.reaching_hand   # the dataset's reaching hand, as eval.py (eval.hand: known)
    pred = kinematic_inference(trial, frames, f_obs, params, H, max_iter, hand, tol, settings)
    t_rem = (trial.offset_idx - f_obs) * trial.dt
    fut = np.round(np.linspace(f_obs, trial.offset_idx, H + 1)).astype(int)[1:]
    e, ci, cu, cm = [], [], [], []
    for part in ("wrist", "elbow"):
        j = f"{trial.reaching_hand}_{part}"
        e.append(frames.keypoint(j)[fut] - to_gt_timeline(pred.joints[j], pred.t_pred, t_rem)[1:])
        ci.append(to_gt_timeline(pred.cov_init[part], pred.t_pred, t_rem)[1:])
        cu.append(to_gt_timeline(pred.cov_unit[part], pred.t_pred, t_rem)[1:])
        cm.append(to_gt_timeline(pred.cov_model[part], pred.t_pred, t_rem)[1:])
    return np.concatenate(e), np.concatenate(ci), np.concatenate(cu), np.concatenate(cm)


def _noise_for_coverage(e, ci, cu, target: float) -> float:
    """Smallest noise level whose 95 % ellipsoids contain a fraction `target` of the errors (log10 bisection)."""
    frac = lambda sigma: float(np.mean(coverage_fraction(e, ci + sigma ** 2 * cu)))
    lo, hi = -4.0, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if frac(10 ** mid) < target else (lo, mid)
    return float(10 ** hi)


def calibrate_pred_noise(trials: Sequence[CariTrial], obs_ratio: float, params: HumanKinematicParams, H: int,
                         max_iter: int, target: float = 0.95, tol: Optional[float] = None,
                         settings: Optional[PredictionSettings] = None) -> Tuple[float, Dict]:
    """Joint-velocity noise level of the random-walk prediction covariance (PredictionSettings covariance
    "random_walk", the ablation of the model's own covariance) such that a fraction `target` of the wrist and elbow
    prediction errors fall inside the 95 % ellipsoids, at the evaluated horizon (predictions from t_obs). The 95 %
    coverage of the model covariance on the same errors is reported too ("coverage_model_covariance"; it is not
    calibrated).

    The level of each subject is calibrated on the other subjects' trials only (leave-one-subject-out), so the future
    of the evaluated trials is never used; "pred_noise" (all subjects) is the value for a new person. A calibration on
    the observed prefix only does not transfer: there the predictions start near rest with an unconverged handover
    filter and short horizons, and the handover covariance alone already covers the errors (noise -> 0), while at
    t_obs the errors grow with the horizon (model bias)."""
    samples = {}
    for trial in trials:
        samples.setdefault(trial.subject, []).append(prediction_error_samples(trial, obs_ratio, params, H, max_iter,
                                                                              tol, settings))
    pooled = lambda subjects: tuple(np.concatenate([x[i] for s in subjects for x in samples[s]]) for i in range(4))
    by_subject = {s: _noise_for_coverage(*pooled([o for o in samples if o != s])[:3], target) for s in samples}
    e, ci, cu, cm = pooled(list(samples))
    sigma = _noise_for_coverage(e, ci, cu, target)
    return sigma, {"pred_noise": sigma, "pred_noise_by_subject": by_subject, "calibration": "leave-one-subject-out",
                   "calibration_samples": int(len(e)),
                   "coverage_handover_covariance_only": float(np.mean(coverage_fraction(e, ci))),
                   "coverage_model_covariance": float(np.mean(coverage_fraction(e, cm)))}


PREDICTIVE_NOISE_PARAMS = ("residual_noise", "handover_cov_scale")


def predictive_covariance_parts(trial: CariTrial, obs_ratio: float, params: HumanKinematicParams, H: int,
                                max_iter: int, tol: Optional[float] = None,
                                settings: Optional[PredictionSettings] = None, ref_noise: float = 1.0
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Errors e (n, 3) of the reaching wrist and elbow predicted from obs_ratio of `trial` (as eval.py) and the three
    parts of their model covariance (n, 3, 3) each:  S = handover_cov_scale * A + B + residual_noise^2 * C.

    With the cost weights fixed, the plan, its feedback gains and the kinematic Jacobians do not depend on these two
    noise parameters (the residual noise is not seen by the controller, the handover covariance only starts the
    propagation), and the closed-loop covariance recursion is linear in its initial covariance and in its noise
    covariances: A = the handover covariance propagated, B = the controller's own noises (motor, decision), C = the
    residual noise at unit intensity. All three come from the model covariance itself (three predictions: handover
    scale 1 and 2 without residual noise, scale 1 with residual noise ref_noise), so that S reproduces the published
    covariance exactly (cov_init, the host-side propagation of the random-walk ablation, differs by a few %)."""
    settings = replace(settings or PredictionSettings(), covariance="model", residual=True)
    base = params._replace(residual_noise=0.0, handover_cov_scale=1.0)
    e, _, _, S00 = prediction_error_samples(trial, obs_ratio, base, H, max_iter, tol, settings)
    _, _, _, S20 = prediction_error_samples(trial, obs_ratio, base._replace(handover_cov_scale=2.0), H, max_iter, tol,
                                            settings)
    _, _, _, S01 = prediction_error_samples(trial, obs_ratio, base._replace(residual_noise=float(ref_noise)), H,
                                            max_iter, tol, settings)
    A = S20 - S00
    return e, A, S00 - A, (S01 - S00) / float(ref_noise) ** 2


def predictive_nll(e: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, scale: float, residual: float,
                   jitter: float = 1e-8) -> np.ndarray:
    """Per-sample Gaussian negative log-likelihood (up to a constant) of the errors under S = scale A + B + residual^2 C
    (a composite likelihood: wrist and elbow, every prediction step, as separate 3-D marginals)."""
    S = scale * A + B + residual ** 2 * C
    S = 0.5 * (S + np.swapaxes(S, -1, -2)) + jitter * np.eye(3)
    sign, logdet = np.linalg.slogdet(S)
    q = np.einsum("ni,ni->n", e, np.linalg.solve(S, e[..., None])[..., 0])
    return 0.5 * (logdet + q) + np.where(sign > 0, 0.0, 1e6)


def fit_predictive_noise(trials: Sequence[CariTrial], params: HumanKinematicParams, obs_ratios: Sequence[float], H: int,
                         max_iter: int, tol: Optional[float] = None, settings: Optional[PredictionSettings] = None,
                         names: Sequence[str] = PREDICTIVE_NOISE_PARAMS) -> Tuple[HumanKinematicParams, Dict]:
    """Stage 2 of train.py (ioc.noise_fit.objective: predictive): the noise parameters `names` (residual_noise,
    handover_cov_scale) that maximize the likelihood of the recorded wrist and elbow positions under the model's own
    predictive distribution, i.e. the open-loop predictions from every handover point obs_ratios of the training
    reaches, exactly as eval.py makes them (the covariance the prediction publishes, not the one-step transitions of
    the IOC likelihood). The cost weights of `params` are fixed. Thanks to the linearity of predictive_covariance_parts
    the predictions are computed once and the fit (L-BFGS-B in log10 space, within get_params_bounds) takes seconds.
    Returns the parameters and a record (values, negative log-likelihood and 95 % coverage before / after, per
    observed fraction)."""
    from scipy.optimize import minimize
    names = tuple(names)
    unknown = [n for n in names if n not in PREDICTIVE_NOISE_PARAMS]
    if unknown:
        raise ValueError(f"predictive noise fit: {unknown} not in {PREDICTIVE_NOISE_PARAMS}")
    t0 = time.perf_counter()
    parts = {}
    for r in obs_ratios:
        rows = [predictive_covariance_parts(tr, float(r), params, H, max_iter, tol, settings)
                for tr in tqdm(trials, desc=f"  predictions from {float(r):.0%} observed", unit="reach", leave=False)]
        parts[float(r)] = tuple(np.concatenate([x[i] for x in rows]) for i in range(4))
    pooled = tuple(np.concatenate([parts[r][i] for r in parts]) for i in range(4))
    lo, hi = HumanKinematicReaching.get_params_bounds()
    cur = {"residual_noise": float(params.residual_noise), "handover_cov_scale": float(params.handover_cov_scale)}
    # linearity check on the data: the parts must reproduce the model covariance of an actual prediction at other
    # values (e.g. it would not hold for a partially observed prediction, where P0 also enters the agent's belief)
    r0, chk = float(obs_ratios[0]), {"residual_noise": 0.7 * cur["residual_noise"] + 0.05, "handover_cov_scale": 2.0}
    e_c, _, _, S_c = prediction_error_samples(trials[0], r0, params._replace(**chk), H, max_iter, tol,
                                              replace(settings or PredictionSettings(), covariance="model",
                                                      residual=True))
    e_p, A_p, B_p, C_p = predictive_covariance_parts(trials[0], r0, params, H, max_iter, tol, settings)
    S_p = chk["handover_cov_scale"] * A_p + B_p + chk["residual_noise"] ** 2 * C_p
    lin_err = float(np.max(np.abs(S_p - S_c)) / max(np.max(np.abs(S_c)), 1e-12))
    if lin_err > 1e-3:
        print(f"  WARNING: predictive covariance not linear in the noise parameters (relative error {lin_err:.2e}): "
              f"the fitted values are approximate", flush=True)

    def unpack(z):
        v = dict(cur)
        v.update({n: float(10.0 ** zi) for n, zi in zip(names, z)})
        return v

    nll = lambda v, P=pooled: float(np.mean(predictive_nll(*P, v["handover_cov_scale"], v["residual_noise"])))
    z0 = np.log10([cur[n] for n in names])
    bounds = [(np.log10(getattr(lo, n)), np.log10(getattr(hi, n))) for n in names]
    res = minimize(lambda z: nll(unpack(z)), z0, method="L-BFGS-B", bounds=bounds)
    fitted = unpack(res.x)

    def coverage(v, P):
        S = v["handover_cov_scale"] * P[1] + P[2] + v["residual_noise"] ** 2 * P[3]
        return float(np.mean(coverage_fraction(P[0], S)))

    per_ratio = {ratio_tag(r): {"coverage_before": coverage(cur, P), "coverage_after": coverage(fitted, P),
                                "nll_before": nll(cur, P), "nll_after": nll(fitted, P), "samples": int(len(P[0]))}
                 for r, P in parts.items()}
    info = {"objective": "predictive", "infer": list(names), "before": {n: cur[n] for n in names},
            "after": {n: fitted[n] for n in names}, "loss_base": nll(cur), "loss_fit": nll(fitted),
            "iterations": int(res.nit), "converged": bool(res.success), "samples": int(len(pooled[0])),
            "linearity_error": lin_err,
            "coverage_before": coverage(cur, pooled), "coverage_after": coverage(fitted, pooled),
            "per_ratio": per_ratio, "fit_time_s": time.perf_counter() - t0}
    return params._replace(**{n: fitted[n] for n in names}), info


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
# Inferred parameters: the learnable cost weights the model configuration keeps (learnable_params) and, for the
# likelihood, the likelihood-only residual noise (plus the observation noise of the partially observed model). The
# noises of the controller (motor_noise, motor_noise_add) are not fitted: fitted to the one-step residuals, which are
# mostly model mismatch, the motor noise went to its upper bound, and the controller, which plans against it, then
# stopped short of the target.
def fit_param_names(ioc_cfg, model_cfg) -> Tuple[str, ...]:
    """Names of the inferred parameters: ioc.params if given, else the learnable cost weights of the model config
    (+ residual_noise for the likelihood, + obs_noise when partially observed; + motor_noise for the open-loop
    objective with model.learn_motor_noise)."""
    if ioc_cfg.get("params"):
        return tuple(ioc_cfg.params)
    names = learnable_params(dict(model_cfg))
    if ioc_cfg.objective == "open_loop" and model_cfg.get("learn_motor_noise", False):
        # it shapes the plan (signal-dependent noise in the gILQR backward pass: smoother, slower reaches); not with
        # the one-step likelihood, where fitting it to the residuals pushed it to its bound
        names += ("motor_noise",)
    if ioc_cfg.objective == "likelihood":
        names += ("residual_noise",) + (("obs_noise",) if ioc_cfg.get("observability", "full") == "partial" else ())
    return names


# Windows evaluated at once on the GPU = restarts (vmap) x batch (lax.map chunk inside a restart). Budgets that fit the
# 8 GB GPU: the likelihood gradient carries the noise Jacobians of the gILQR linearization and needs much more memory
# than the open-loop one (likelihood: 16 or 4 windows x 4 restarts ran out; open loop: 16 x 4 restarts ran out, 16 x 1
# fits). With ioc.batch_size null the batch is budget / restarts (at least 1).
WINDOWS_IN_FLIGHT = {"open_loop": 16, "likelihood": 2}


def default_batch_size(objective: str, restarts: int) -> int:
    return max(1, WINDOWS_IN_FLIGHT[objective] // max(1, int(restarts)))

_UPPER_BODY_KP = jnp.array([hkm.KP_INDEX[n] for n in ("head", "left_shoulder", "left_elbow", "left_wrist",
                                                      "right_shoulder", "right_elbow", "right_wrist")])


_HIPS = jnp.array([hkm.KP_INDEX["left_hip"], hkm.KP_INDEX["right_hip"]])


def upper_body_output(env: HumanKinematicReaching, state: jnp.ndarray) -> jnp.ndarray:
    """The 9 MPJPE joints (head, shoulders, elbows, wrists, chest, pelvis = hip midpoint; 9 x 3, flattened): the space
    of the open-loop objective."""
    kp = env.all_keypoints(state)
    return jnp.concatenate([kp[_UPPER_BODY_KP], env.chest(state)[None], kp[_HIPS].mean(axis=0)[None]]).ravel()


def trial_state(trial: CariTrial, q_ref: np.ndarray, data_cfg=None, damping: float = 0.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """19-DOF joint positions and velocities of the whole trial (chest rotation vector relative to q_ref), estimated
    by data.state_estimator: "rts" (default; RTS smoother under the model's ZOH dynamics with damping, noise levels
    data.rts_accel_noise / data.rts_obs_noise) or "savgol" (Savitzky-Golay with the dataset's settings)."""
    data_cfg = data_cfg or {}
    estimator = data_cfg.get("state_estimator", "rts")
    if estimator == "rts":
        return rts_upper_body_state(trial, q_ref, accel_noise=float(data_cfg.get("rts_accel_noise", RTS_ACCEL_NOISE)),
                                    obs_noise=float(data_cfg.get("rts_obs_noise", RTS_OBS_NOISE)), damping=damping)
    if estimator == "savgol":
        return sg_upper_body_state(trial, q_ref)
    raise ValueError(f"data.state_estimator must be 'rts' or 'savgol', got {estimator}")


def window_starts(data_cfg) -> List[float]:
    """Starts of the training windows: the observed fractions data.obs_ratios, i.e. the handover points of eval.py
    (training windows aligned with the predictions)."""
    return [float(r) for r in data_cfg.obs_ratios]


def fit_segments(trial: CariTrial, starts: Sequence[float], T_fit: int, data_cfg=None, damping: float = 0.0,
                 window_start: Optional[str] = None) -> List[Tuple[np.ndarray, np.ndarray, HumanKinematicReaching]]:
    """Training windows of a complete reach (a demonstration of a training subject).

    For each start (observed fraction of the reach, data.obs_ratios: the handover frame obs_frame of eval.py), the
    joint state from the start to
    the end of the reach on T_fit steps (dt = remaining time / T_fit), as a prediction from that start would be
    computed: positions and velocities over the whole reach from trial_state (RTS smoother or Savitzky-Golay), the
    chest rotation vector is relative to the start orientation, the target is the wrist at the end of the reach.

    window_start (default data.window_start, else "kalman"): the first state of each window.
        kalman    the handover estimate the prediction starts from (handover_state: Kalman filter of the joint history
                  from the onset to the start, causal), in the same chest frame (orientation at the start frame); the
                  rest of the window is the smoothed recording the open-loop prediction is compared with
        smoothed  the smoothed state (trial_state, which also uses the future of the reach): consistent transitions,
                  as the one-step likelihood needs (fit_params uses it for that objective)

    Returns a list of (x (T_fit+1, 38), mask (T_fit+1,) of ones, env).
    """
    if window_start is None:
        window_start = str((data_cfg or {}).get("window_start", "kalman") if hasattr(data_cfg or {}, "get")
                           else getattr(data_cfg, "window_start", "kalman"))
    if window_start not in ("kalman", "smoothed"):
        raise ValueError(f"window_start must be kalman or smoothed, got {window_start!r}")
    frames = trial_frames(trial)
    target = frames.target
    target_left = frames.target_left
    hand = trial.reaching_hand
    ar = 1.0 if hand in ("right", "both") else 0.0
    al = 1.0 if hand in ("left", "both") else 0.0

    segments = []
    for s in starts:
        s_idx = obs_frame(trial, s)
        q_ref = trial.q28_filt[s_idx][3:7]
        q, qd = trial_state(trial, q_ref, data_cfg, damping)
        t_grid = np.linspace(s_idx, trial.offset_idx, T_fit + 1)
        x = np.concatenate([q, qd], axis=1)
        x = np.stack([np.interp(t_grid, np.arange(len(q)), x[:, j]) for j in range(x.shape[1])], axis=1)
        if window_start == "kalman":   # as the prediction: the causal estimate at the handover, same chest frame
            x[0] = handover_state(trial, s_idx, damping)[0]
        dt = (trial.offset_idx - s_idx) * trial.dt / T_fit
        env = make_env(trial, x[0, :19], q_ref, target, dt, hand,
                       target_left=target_left, alpha_right=ar, alpha_left=al)
        segments.append((x.astype(np.float32), np.ones(T_fit + 1, dtype=np.float32), env))
    return segments


def _groups_by_hand(segments):
    groups = []
    for hand in ("right", "left", "both"):
        seg = [s for s in segments if s[2].reaching_hand == hand]
        if seg:
            groups.append((stack_envs([s[2] for s in seg]), jnp.asarray(np.stack([s[0] for s in seg])),
                           jnp.asarray(np.stack([s[1] for s in seg]))))
    return groups



def make_objective(groups, base_params: HumanKinematicParams, infer: Sequence[str], ioc_cfg):
    """The IOC objective of ioc_cfg over the stacked training windows `groups` (loglikelihood(None, params) to
    maximize): "likelihood" (MultiTrialLikelihood with ioc.observability, ioc.linearization, ioc.temperature,
    ioc.likelihood_block, ioc.solve_iters, ioc.checkpoint) or "open_loop" (MultiTrialTrajectoryMatching)."""
    objective = ioc_cfg.objective
    restarts = int(ioc_cfg.get("restarts", 1))
    batch = int(ioc_cfg.get("batch_size") or default_batch_size(objective, restarts))
    print(f"  GPU batch: {restarts} restarts x {batch} windows = {restarts * batch} windows at once"
          + (f" (above the {WINDOWS_IN_FLIGHT[objective]} known to fit 8 GB: lower ioc.restarts or ioc.batch_size if "
             f"it runs out of GPU memory)" if restarts * batch > WINDOWS_IN_FLIGHT[objective] else ""), flush=True)
    iters = int(ioc_cfg.get("solve_iters", 8))
    checkpoint = bool(ioc_cfg.get("checkpoint", False))
    if objective == "open_loop":
        return MultiTrialTrajectoryMatching(groups, base_params, infer, output_fn=upper_body_output, batch_size=batch,
                                            solve_iters=iters, checkpoint=checkpoint)
    if objective == "likelihood":
        block = ioc_cfg.get("likelihood_block", "full")
        if block not in ("full", "velocity"):
            raise ValueError(f"ioc.likelihood_block must be 'full' or 'velocity', got {block}")
        return MultiTrialLikelihood(groups, base_params, infer,
                                    velocity_block=slice(19, 38) if block == "velocity" else None,
                                    linearization=ioc_cfg.get("linearization", "solve"), solve_iters=iters,
                                    batch_size=batch, observability=ioc_cfg.get("observability", "full"),
                                    temperature=float(ioc_cfg.get("temperature", 1e-6)), checkpoint=checkpoint)
    raise ValueError(f"objective must be 'likelihood' or 'open_loop', got {objective}")



# =============================================================================
# Compilation with progress (fit_params)
# =============================================================================
COMPILE_LOG = REPO_ROOT / "output" / "compile_times.jsonl"   # one record per compiled IOC objective


class _CacheLog(logging.Handler):
    """Collects the persistent-cache messages of JAX (hit, write, or why an entry was not written)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "persistent" in msg.lower() or "cache" in msg.lower():
            self.messages.append(msg if len(msg) < 300 else msg[:300] + "...")


def _compile_estimate(records: List[Dict], key: Dict, size_mb: Optional[float], field: str) -> Optional[float]:
    """Median of `field` (seconds) over the earlier records of the same objective and device (scaled by the program
    size for the XLA compilation); None without such records."""
    same = [r for r in records if r.get("objective") == key["objective"] and r.get("device") == key["device"]
            and not r.get("cache_hit") and r.get(field)]
    if not same:
        return None
    if field == "compile_s" and size_mb:
        return float(np.median([r[field] * size_mb / r["size_mb"] for r in same if r.get("size_mb")]))
    return float(np.median([r[field] for r in same]))


def _wait_with_progress(fn, desc: str, estimate: Optional[float]):
    """fn() while a progress bar shows the elapsed time against the estimate (s; elapsed only if None) and the RAM
    of the process. Returns (result, seconds)."""
    if os.environ.get("PROPHET_PROGRESS", "1") == "0":   # no progress thread (diagnostics): fn() and its duration
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        print(f"{desc}: done in {dt:.0f} s", flush=True)
        return out, dt
    try:
        from resources import memory
    except ImportError:
        memory = lambda: {}
    done = threading.Event()
    estimate = estimate if estimate and estimate >= 1.0 else None
    fmt = ("{desc}: {percentage:3.0f}% of estimate |{bar}| {elapsed} {postfix}" if estimate
           else "{desc}: {elapsed} {postfix}")
    bar = tqdm(total=estimate, desc=desc, bar_format=fmt, unit="s", leave=True)
    t0 = time.perf_counter()

    def tick():
        while not done.wait(1.0):
            el = time.perf_counter() - t0
            bar.n = min(el, 0.99 * bar.total) if bar.total else el
            m = memory()
            left = "" if not bar.total else (f"~{(bar.total - el) / 60:.1f} min left of ~{bar.total / 60:.1f} min"
                                             if el < bar.total else f"over the estimate of ~{bar.total / 60:.1f} min")
            bar.set_postfix_str(", ".join(x for x in (left, f"RAM {m['rss']:.1f} GB" if m else "") if x))
    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        out = fn()
    finally:
        done.set()
        th.join()
        dt = time.perf_counter() - t0
        if bar.total:
            bar.n = bar.total
        bar.set_postfix_str(f"done in {dt:.0f} s")
        bar.close()
    return out, dt


def compile_with_progress(jitted, args: Tuple, key: Dict):
    """Ahead-of-time compilation of jitted(*args) in its phases, with progress: tracing and lowering to StableHLO
    (Python), XLA compilation (or loading from the persistent cache in jax_cache/), first evaluation. The phases cannot
    report their own progress: the bars compare the elapsed time with the earlier compilations of the same objective
    and device in output/compile_times.jsonl (scaled by the program size), and the run is appended there. key:
    objective, device and the sizes of the program (segments, restarts, ...). Returns (compiled, first result)."""
    records = []
    if COMPILE_LOG.exists():
        for line in COMPILE_LOG.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    lowered, t_lower = _wait_with_progress(lambda: jitted.lower(*args), "  1/3 tracing + lowering",
                                           _compile_estimate(records, key, None, "lower_s"))
    size_mb = len(lowered.as_text()) / 1e6
    est = _compile_estimate(records, key, size_mb, "compile_s")
    n_prev = sum(1 for r in records if r.get("objective") == key["objective"] and r.get("device") == key["device"])
    print(f"      program {size_mb:.1f} MB of StableHLO; XLA compilation estimate "
          + (f"~{est / 60:.1f} min (from {n_prev} earlier compilations, scaled by size)" if est else
             "unavailable (first compilation of this objective on this device)"), flush=True)
    hits = []
    listener = lambda event, **kw: hits.append(event) if event == "/jax/compilation_cache/cache_hits" else None
    jax.monitoring.register_event_listener(listener)
    cache_log = _CacheLog()
    loggers = [logging.getLogger(n) for n in ("jax._src.compiler", "jax._src.compilation_cache")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(cache_log)
        lg.setLevel(logging.DEBUG)
    try:
        compiled, t_comp = _wait_with_progress(lowered.compile, "  2/3 XLA compilation", est)
    finally:
        jax.monitoring.unregister_event_listener(listener)
        for lg, lv in zip(loggers, levels):
            lg.removeHandler(cache_log)
            lg.setLevel(lv)
    hit = bool(hits)
    print("      " + ("loaded from the persistent cache (jax_cache/)" if hit else "compiled (not in the persistent cache)"),
          flush=True)
    for msg in cache_log.messages:
        if not hit or "hit" in msg.lower():
            print(f"      jax cache: {msg}", flush=True)
    first, t_first = _wait_with_progress(lambda: jax.block_until_ready(compiled(*args)), "  3/3 first evaluation",
                                         _compile_estimate(records, key, None, "first_eval_s"))
    rec = {"date": datetime.now().isoformat(timespec="seconds"), **key, "size_mb": round(size_mb, 2),
           "lower_s": round(t_lower, 1), "compile_s": round(t_comp, 1), "first_eval_s": round(t_first, 1),
           "cache_hit": hit}
    try:
        COMPILE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(COMPILE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    print(f"  ✓ ready in {t_lower + t_comp + t_first:.0f} s (tracing {t_lower:.0f} s, XLA "
          f"{'cache load' if hit else 'compilation'} {t_comp:.0f} s, first evaluation {t_first:.0f} s)", flush=True)
    return compiled, first

def fit_params(trials: Sequence[CariTrial], base_params: HumanKinematicParams, ioc_cfg, model_cfg, data_cfg=None,
               callback: Optional[Callable[[int, Dict], None]] = None, segments: Optional[List] = None
               ) -> Tuple[HumanKinematicParams, Dict]:
    """IOC fit of the cost weights on the complete reaches of `trials` (fit_segments), or on the given training
    windows `segments` ([(x, mask, env)] as fit_segments returns them, e.g. simulated: synthetic_recovery.py).

    ioc_cfg: objective ("likelihood" | "open_loop"), observability, linearization, temperature, likelihood_block,
    solve_iters, checkpoint, params (inferred names, null = fit_param_names), T_fit, restarts, max_iter, lr, patience,
    tol, batch_size, seed; model_cfg: the model configuration (learnable terms); data_cfg: the window starts
    (data.obs_ratios, the handover points of eval.py) and the state estimator of the training windows. The parameters not inferred keep their base_params values. Restart 0
    starts from base_params, the others uniformly at random in log10 space within
    HumanKinematicReaching.get_params_bounds(); all restarts run in parallel (projected Adam in log10 space, one
    batched gradient per iteration), with a progress bar. Returns the best point visited.
    callback(iteration, record), if given, is called after every iteration with the loss and gradient norm of each
    restart, the best loss so far, the weights (log10) of each restart and of the best point, and the iteration time
    (train.py logs it to wandb).
    """
    objective = ioc_cfg.objective
    infer = fit_param_names(ioc_cfg, model_cfg)
    if segments is None:
        # the one-step likelihood needs consistent (smoothed) transitions; the open-loop objective starts each window
        # from the Kalman handover estimate, as the prediction (data.window_start)
        start = "smoothed" if objective == "likelihood" else None
        segments = [seg for tr in trials for seg in fit_segments(tr, window_starts(data_cfg), ioc_cfg.T_fit, data_cfg,
                                                                 base_params.damping, window_start=start)]
    groups = _groups_by_hand(segments)
    ioc = make_objective(groups, base_params, infer, ioc_cfg)
    n_seg = len(segments)

    def to_params(theta):
        return ioc.full_params(base_params._replace(**{k: 10.0 ** theta[i] for i, k in enumerate(infer)}))

    # per segment, for the tolerance; the data are an argument of the compiled function, not constants (see
    # MultiTrialLikelihood)
    loss = lambda theta, groups: -ioc.loglikelihood(None, to_params(theta), groups) / n_seg
    lo, hi = HumanKinematicReaching.get_params_bounds()
    lo = jnp.log10(jnp.array([getattr(lo, k) for k in infer]))
    hi = jnp.log10(jnp.array([getattr(hi, k) for k in infer]))
    theta_base = jnp.clip(jnp.log10(jnp.array([max(getattr(base_params, k), 1e-12) for k in infer])), lo, hi)
    key = jax.random.PRNGKey(ioc_cfg.seed)
    theta0 = jax.random.uniform(key, (ioc_cfg.restarts, len(infer)), minval=lo, maxval=hi).at[0].set(theta_base)

    # Projected Adam in log10 space, all restarts in one batched value-and-gradient call per iteration (one
    # moderate compiled program; a line-search optimizer such as jaxopt.LBFGSB traces the objective several times
    # inside its loops, which took minutes to compile and ~15 s per iteration here)
    value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, None)))
    lr, b1, b2, eps = float(ioc_cfg.lr), 0.9, 0.999, 1e-8
    m = jnp.zeros_like(theta0)
    v = jnp.zeros_like(theta0)
    theta = theta0
    best_theta, best_val = np.array(theta0), np.full(ioc_cfg.restarts, np.inf)
    history, history_restarts, loss_base = [], [], None
    stopped_early, stop_iter = False, None

    t_start = time.perf_counter()
    print(f"  compiling the {objective} objective ({n_seg} segments in {len(ioc.groups)} hand groups, {len(infer)} "
          f"parameters, {ioc_cfg.restarts} parallel restarts)...", flush=True)
    key = {"objective": objective, "device": next(iter(theta.devices())).platform, "n_seg": n_seg,
           "n_groups": len(ioc.groups), "restarts": int(ioc_cfg.restarts), "n_params": len(infer),
           "T_fit": int(ioc_cfg.T_fit), "solve_iters": int(ioc_cfg.get("solve_iters", 8))}
    value_and_grad, (val, g) = compile_with_progress(value_and_grad, (theta, ioc.groups), key)
    print(f"  starting the optimization ({ioc_cfg.max_iter} iterations at most)\n", flush=True)

    val, g = np.array(val), np.array(g)
    loss_base = float(val[0])
    bar = tqdm(range(1, ioc_cfg.max_iter + 1), desc=f"IOC fit ({objective})", unit="it")
    t_it = time.perf_counter()
    for it in bar:
        if it > 1:
            val, g = value_and_grad(theta, ioc.groups)
            val, g = np.array(val), np.array(g)
        theta_it = np.array(theta)
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
        b = int(np.argmin(np.where(np.isfinite(best_val), best_val, np.inf)))
        now = time.perf_counter()
        step_dt = now - t_it
        t_it = now
        bar.set_postfix(
            best=f"{history[-1]:.4g}",
            base=f"{loss_base:.4g}",
            r=b,
            gnorm=f"{float(np.linalg.norm(g[b])):.2e}",
            step=f"{step_dt:.1f}s",
        )
        if it % 10 == 0 or it == 1 or better.any():
            bar.write(f"  [Iter {it:3d}/{ioc_cfg.max_iter}] best: {history[-1]:10.4f} (restart {b}) | "
                      f"base: {loss_base:10.4f} | grad_norm: {float(np.linalg.norm(g[b])):8.2e} | "
                      f"step: {step_dt:5.1f}s | elapsed: {(now - t_start)/60:4.1f} min")
        if callback is not None:
            callback(it, {"loss": val.tolist(), "grad_norm": np.linalg.norm(np.where(np.isfinite(g), g, 0.0),
                                                                          axis=1).tolist(),
                          "best_loss": history[-1], "best_restart": b, "infer": list(infer),
                          "log10_params": theta_it.tolist(), "best_log10_params": best_theta[b].tolist(),
                          "iteration_time_s": step_dt, "elapsed_s": now - t_start})
        if it > ioc_cfg.patience and history[-ioc_cfg.patience - 1] - history[-1] < ioc_cfg.tol * abs(history[-1]):
            stopped_early = True
            stop_iter = it
            break
    bar.close()
    if stopped_early:
        print(f"\n  ==========================================================================", flush=True)
        print(f"  ✓ EARLY STOPPING triggered at iteration {stop_iter}/{ioc_cfg.max_iter}", flush=True)
        print(f"    Best loss improved by < {ioc_cfg.tol:.1e} over the last {ioc_cfg.patience} iterations.", flush=True)
        print(f"  ==========================================================================\n", flush=True)
    else:
        print(f"\n  ✓ Completed all {ioc_cfg.max_iter} iterations successfully.\n", flush=True)
    values = best_val
    values = np.where(np.isfinite(values), values, np.inf)
    best = int(np.argmin(values))
    fitted = to_params(jnp.asarray(best_theta[best]))
    fitted = HumanKinematicParams(**params_to_dict(fitted))  # plain floats
    info = {"objective": objective, "observability": ioc_cfg.get("observability", "full"),
            "linearization": ioc_cfg.get("linearization", "solve"),
            "temperature": float(ioc_cfg.get("temperature", 1e-6)), "infer": list(infer), "T_fit": ioc_cfg.T_fit,
            "segment_starts": window_starts(data_cfg), "n_trials": len(trials), "n_segments": n_seg,
            "loss_base": loss_base, "loss_fit": float(values[best]), "loss_restarts": values.tolist(),
            "best_restart": best, "iterations": len(history), "loss_history": history,
            "loss_history_restarts": history_restarts,
            "theta_restarts": {k: (10.0 ** best_theta[:, i]).tolist() for i, k in enumerate(infer)},
            "fit_time_s": time.perf_counter() - t_start,
            "trials": [f"{tr.subject}/{tr.velocity}/inst{tr.instruction_id}" for tr in trials]}
    return fitted, info


def segment_errors(trials: Sequence[CariTrial], params: HumanKinematicParams, ioc_cfg, data_cfg=None
                   ) -> Dict[str, np.ndarray]:
    """Open-loop prediction error of every training segment with `params` (segment_errors_many)."""
    return segment_errors_many(trials, [params], ioc_cfg, data_cfg)[0]


def segment_errors_many(trials: Sequence[CariTrial], params_sets: Sequence[HumanKinematicParams], ioc_cfg,
                        data_cfg=None, labels: Optional[Sequence[str]] = None) -> List[Dict[str, np.ndarray]]:
    """Open-loop prediction error of every training segment (fit_segments) for each parameter set: RMS over the 9
    joints and the T_fit steps (cm), with the subject, instruction and start of each segment.

    The parameters are an argument of the compiled program, not constants baked into it: one compilation per hand group
    serves every parameter set (and, independent of the weights, the program can come from the persistent cache in a
    later run with the same windows). Progress: compile_with_progress for the first set, a timed bar for the others."""
    from prophet_ioc.infer.multi_env import trial_open_loop_error
    labels = list(labels) if labels is not None else [f"set {k + 1}" for k in range(len(params_sets))]
    rows = []
    starts = window_starts(data_cfg)
    for tr in trials:   # the windows depend on the fixed damping only, the same for every parameter set
        for start, seg in zip(starts, fit_segments(tr, starts, ioc_cfg.T_fit, data_cfg, params_sets[0].damping)):
            rows.append((tr.subject, tr.instruction_id, float(start), seg))
    as_arg = lambda p: jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=jnp.float32), p)   # same types each set
    run = jax.jit(lambda e, x, p: jax.lax.map(lambda ex: trial_open_loop_error(ex[0], ex[1], p, upper_body_output),
                                              (e, x), batch_size=16))
    err = np.zeros((len(params_sets), len(rows)))
    for hand in ("right", "left", "both"):
        idx = [i for i, r in enumerate(rows) if r[3][2].reaching_hand == hand]
        if not idx:
            continue
        envs = stack_envs([rows[i][3][2] for i in idx])
        xs = jnp.asarray(np.stack([rows[i][3][0] for i in idx]))
        print(f"  open-loop error, {hand} hand: {len(idx)} windows x {len(params_sets)} weight sets "
              f"({', '.join(labels)}), one compiled program", flush=True)
        key = {"objective": "segment_errors", "device": next(iter(xs.devices())).platform, "n_seg": len(idx),
               "hand": hand, "T_fit": int(ioc_cfg.T_fit), "solve_iters": int(ioc_cfg.get("solve_iters", 8))}
        compiled, first = compile_with_progress(run, (envs, xs, as_arg(params_sets[0])), key)
        err[0, idx] = np.array(first)
        t_first = None
        for k in range(1, len(params_sets)):
            out, t_first = _wait_with_progress(
                lambda k=k: jax.block_until_ready(compiled(envs, xs, as_arg(params_sets[k]))),
                f"  evaluation ({labels[k]})", t_first)
            err[k, idx] = np.array(out)
    meta = {"subject": np.array([r[0] for r in rows]), "instruction": np.array([r[1] for r in rows]),
            "start": np.array([r[2] for r in rows])}
    return [{"rms_cm": 100.0 * np.sqrt(e / 9.0), **meta} for e in err]


# =============================================================================
# Data-driven baselines (prophet_ioc.baselines: ProMP, DMP), learned from the training reaches
# =============================================================================
def training_reaches(trials: Sequence[CariTrial]) -> List:
    """Complete reaches (onset -> offset, every frame) of `trials` as demonstrations of the data-driven baselines: the
    9 joints of the evaluation (FK of the filtered IK angles, trial_frames) and the reaching hand of the metadata."""
    from prophet_ioc.baselines import Reach
    reaches = []
    for tr in trials:
        fr = trial_frames(tr)
        reaches.append(Reach(fr.joints(slice(tr.onset_idx, tr.offset_idx + 1)), tr.reaching_hand, tr.dt))
    return reaches


def fit_baselines(names: Sequence[str], trials: Sequence[CariTrial], options: Optional[Dict] = None) -> Dict:
    """{name: fitted predictor} of the data-driven baselines `names` (keys of prophet_ioc.baselines.BASELINES),
    learned from the complete reaches of `trials` (the training subjects), with options {name: {fit kwargs}}."""
    from prophet_ioc.baselines import BASELINES
    unknown = [n for n in names if n not in BASELINES]
    if unknown:
        raise ValueError(f"unknown baselines {unknown}, available: {list(BASELINES)}")
    reaches = training_reaches(trials)
    return {n: BASELINES[n].fit(reaches, **dict((options or {}).get(n) or {})) for n in names}
