#!/usr/bin/env python3
"""Evaluation of the 19-DOF kinematic predictor and of the baselines on CARI v2 reaches.

For every reach of the held-out test subjects (data.test_subjects, never seen by train.py; all subjects if none) and
every observed fraction in data.obs_ratios, the first part of the reach is observed and the rest is predicted
(H = model.horizon steps) by
    kin      the kinematic model (gILQR in joint space, rigid bones) with the IOC-fitted weights of a train.py run
             (eval.params, default output/latest_train) and its own predictive covariance (model.prediction_covariance:
             model; random_walk = the calibrated joint-velocity random walk of the run, ablation); partially observed
             (belief-space prediction) when the run fitted the partially observed model
    kin_init the same model with the initial weights of config/model (the starting point of the IOC fit), for
             reference (eval.compare_initial)
    cart     a damped Cartesian point mass per joint (LQR); the reaching wrist is driven to the target
    minjerk  minimum jerk: wrist to the target, the other joints to rest with a free end position
    gcv      goal-directed constant velocity: wrist straight to the target, the other joints at constant velocity
    cv       constant velocity of every joint (Savitzky-Golay velocity)
    promp    Probabilistic Movement Primitives learned from the training subjects' complete reaches, conditioned on the
             observed prefix and the wrist goal, with covariance (95 % coverage)   (eval.baselines, prophet_ioc.baselines)
    dmp      Dynamic Movement Primitives learned from the same reaches, from the handover state to the wrist goal
Only the reaching wrist has a goal (the target); all joints are predicted by every method.

Results: output/eval_<run>/ (the name of the evaluated train.py run output/train_<run>; eval_initial_<timestamp>
for the initial weights; _2, _3, ... if evaluated again) with summary.html / summary.csv over all ratios and, per ratio, obs<XX>/ with
summary.html (best method per metric in bold), summary.csv, per_trial.csv, results.json, html/ (figures of
eval.plot_trials) and frames/<trial>/{skeleton,skeleton_tube}/ (PNG frames + GIF); figures/ with the errors vs
observed fraction, per instruction and along the prediction.

    python eval.py                                     # weights and noise of output/latest_train
    python eval.py weights=train_20261004_200005       # a train.py run: name under output/ (train_ prefix optional),
    python eval.py weights=output/train_<ts>/params.json   #   run folder or params.json (same as eval.params=...)
    python eval.py weights=initial                     # initial weights of config/model only
    python eval.py model.prediction_covariance=random_walk   # calibrated random-walk covariance (ablation)
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Single-threaded OpenBLAS (the LAPACK of JAX's CPU linear algebra): the small factorizations of the prediction
# (Kalman gains of the belief-space prediction) were 100x slower with its thread pool on a loaded machine
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Triton GEMM autotuning tries kernels with GiB-sized workspaces, which ran out of the 8 GB GPU memory
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))  # cari_kinematic, save_results

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

import cari_kinematic as ck
from prophet_ioc.control import gilqr
from prophet_ioc.envs.cartesian_reaching import CartesianMultiPointReaching3D, CartesianMultiPointReaching3DParams
from prophet_ioc.infer import predict_constant_velocity
from plot_utils import UPPER_BODY_BONES
from plot_utils.cari_plots import plot_trial
from plot_utils import ioc_plots
from save_results import METHODS, print_summary, save_overview, save_results, summarize
from tracking import Tracker

MIN_DISTANCE_M = 0.02  # floor of the percentage normalizer (keypoints that barely move)


def timed(fn, n: int = 15) -> float:
    """Mean wall time of fn() in ms (fn must block until its result is ready)."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def predict_kinematic(trial, frames, f_obs, params, pred_noise, H, max_iter, hand, tol=None, settings=None, reaching_mode="dual"):
    """Kinematic prediction and wrist / elbow covariances on the ground-truth timeline, and the inference latency
    (ms, one complete prediction from the observation window after compilation, ck.kinematic_inference; tol: early
    stopping of the solver, model.early_stopping / model.tol; settings: ck.predictor_settings; pred_noise: None =
    model covariance, else the random walk with this level)."""
    t_rem = (trial.offset_idx - f_obs) * trial.dt
    pred = ck.kinematic_inference(trial, frames, f_obs, params, H, max_iter, hand, tol, settings, reaching_mode=reaching_mode)  # compiles
    latency = timed(lambda: ck.kinematic_inference(trial, frames, f_obs, params, H, max_iter, hand, tol, settings, reaching_mode=reaching_mode),
                    n=10)
    joints = {j: ck.to_gt_timeline(v, pred.t_pred, t_rem) for j, v in pred.joints.items()}
    cov = {k: ck.to_gt_timeline(v, pred.t_pred, t_rem) for k, v in pred.cov(pred_noise).items()}
    return joints, cov, latency, pred.t_pred, pred.dt_sim



def predict_cartesian(obs, hand, target, dt_sim, t_pred, t_rem, H, dt):
    """A damped Cartesian point mass per joint (CartesianMultiPointReaching3D, LQR): the reaching elbow / wrist pair
    with the target on the wrist, the other joints in pairs without target. Returns joints (H+1, 3), latency."""
    params = CartesianMultiPointReaching3DParams(action_cost=jnp.float32(1e-4), velocity_cost=jnp.float32(1e-2),
                                                 motor_noise=jnp.float32(0.1), obs_noise=jnp.float32(0.5))
    w, e = f"{hand}_wrist", f"{hand}_elbow"
    others = [j for j in ck.JOINTS if j not in (w, e)]
    pairs = [(e, w)] + [(others[i], others[min(i + 1, len(others) - 1)]) for i in range(0, len(others), 2)]

    def state(a, b):
        return jnp.asarray(np.concatenate([obs[a][-1], obs[b][-1], ck.sg_velocity(obs[a], dt),
                                           ck.sg_velocity(obs[b], dt)]), dtype=jnp.float32)

    U0 = jnp.zeros((H, 6), dtype=jnp.float32)
    solvers = {}
    for goal in (True, False):
        env = CartesianMultiPointReaching3D(dt=dt_sim, target_hand=target, target_elbow=obs[e][-1],
                                            x0=state(e, w), w_target_hand=100.0 if goal else 0.0,
                                            w_target_elbow=0.0)
        solvers[goal] = jax.jit(lambda x, env=env: gilqr.solve(p=env, x0=x, U_init=U0, params=params, max_iter=1)[1])
    states = [state(a, b) for a, b in pairs]

    def run():
        return [solvers[k == 0](x).block_until_ready() for k, x in enumerate(states)]

    Xs = run()
    latency = timed(run)
    out = {}
    for (a, b), X in zip(pairs, Xs):  # state = [pos a, pos b, vel a, vel b]; the last pair may repeat a joint
        X = ck.to_gt_timeline(np.array(X), t_pred, t_rem)
        out.setdefault(a, X[:, 0:3])
        out.setdefault(b, X[:, 3:6])
    return out, latency


def metrics(pred, gt, hand, nom_bones):
    """Errors in cm and in % of each keypoint's remaining distance (from its position at the first predicted step to
    its final position, i.e. to the target for the reaching wrist; at least MIN_DISTANCE_M), and the maximum
    bone-length distortion in % over the upper-body bones."""
    err = {j: np.linalg.norm(pred[j] - gt[j], axis=1) for j in ck.JOINTS}
    dist = {j: max(float(np.linalg.norm(gt[j][-1] - gt[j][0])), MIN_DISTANCE_M) for j in ck.JOINTS}
    pct = {j: err[j] / dist[j] * 100.0 for j in ck.JOINTS}
    distortion = max(float(np.max(np.abs(np.linalg.norm(pred[a] - pred[b], axis=1) - L)) / max(L, 1e-4) * 100.0)
                     for (a, b), L in nom_bones.items())
    w, e = f"{hand}_wrist", f"{hand}_elbow"
    return {"mpjpe_cm": float(np.mean([err[j].mean() for j in ck.JOINTS]) * 100.0),
            "mpjpe_pct": float(np.mean([pct[j].mean() for j in ck.JOINTS])),
            "wrist_ade_cm": float(err[w].mean() * 100.0), "wrist_ade_pct": float(pct[w].mean()),
            "wrist_fde_cm": float(err[w][-1] * 100.0), "wrist_fde_pct": float(pct[w][-1]),
            "elbow_ade_cm": float(err[e].mean() * 100.0), "elbow_ade_pct": float(pct[e].mean()),
            "elbow_fde_cm": float(err[e][-1] * 100.0), "elbow_fde_pct": float(pct[e][-1]),
            "bone_distortion_pct": distortion}


def evaluate_trial(trial, obs_ratio, params, pred_noise, cfg, initial=None, learned=None, settings=None):
    H, dt, tol = int(cfg.model.horizon), trial.dt, ck.solver_tolerance(cfg.model)
    frames = ck.trial_frames(trial)
    f_obs = ck.obs_frame(trial, obs_ratio)
    fut = np.round(np.linspace(f_obs, trial.offset_idx, H + 1)).astype(int)
    t_obs = (f_obs - trial.onset_idx) * dt
    fut_time = np.linspace(t_obs, trial.reach_frames * dt, H + 1)
    t_rem = (trial.offset_idx - f_obs) * dt
    obs = frames.joints(slice(trial.onset_idx, f_obs + 1))
    gt = frames.joints(fut)
    onset_pose = frames.joints(trial.onset_idx)
    nom_bones = {(a, b): float(np.linalg.norm(onset_pose[a] - onset_pose[b])) for a, b in UPPER_BODY_BONES}
    hand = trial.reaching_hand
    w = f"{hand}_wrist"

    reaching_mode = str(cfg.model.get("reaching_mode", "dual"))
    inferred_hand = ck.infer_reaching_hand(frames, trial.onset_idx, f_obs, allow_both=(reaching_mode == "dual"))
    kin, cov, lat_kin, t_pred, dt_sim = predict_kinematic(trial, frames, f_obs, params, pred_noise, H,
                                                          int(cfg.model.max_iter),
                                                          inferred_hand, tol,
                                                          settings, reaching_mode=reaching_mode)
    cart, lat_cart = predict_cartesian(obs, hand, frames.target, dt_sim, t_pred, t_rem, H, dt)
    methods, latency, covs = {"kin": kin, "cart": cart}, {"kin": lat_kin, "cart": lat_cart}, {"kin": cov}
    if initial is not None:   # (params, pred_noise, settings) of the initial weights
        methods["kin_init"], covs["kin_init"], latency["kin_init"], _, _ = predict_kinematic(
            trial, frames, f_obs, initial[0], initial[1], H, int(cfg.model.max_iter),
            inferred_hand, tol, initial[2], reaching_mode=reaching_mode)

    goal = lambda j: frames.target if j == w else None
    for name in ("minjerk", "gcv"):
        methods[name] = {j: ck.predict_goal_directed(name, obs[j], goal(j), fut_time, dt) for j in ck.JOINTS}
        latency[name] = timed(lambda: [ck.predict_goal_directed(name, obs[j], goal(j), fut_time, dt)
                                       for j in ck.JOINTS], 20)
    methods["cv"] = {j: predict_constant_velocity(obs[j], fut_time, dt) for j in ck.JOINTS}
    latency["cv"] = timed(lambda: [predict_constant_velocity(obs[j], fut_time, dt) for j in ck.JOINTS], 20)
    # data-driven baselines (ck.fit_baselines), with the inputs of minjerk / gcv: observed prefix, wrist target, times
    for name, model in (learned or {}).items():
        run = lambda model=model: model.predict(obs, hand, frames.target, fut_time, dt)
        pred = run()
        methods[name] = pred.joints
        if pred.cov is not None:
            covs[name] = {part: pred.cov[f"{hand}_{part}"] for part in ("wrist", "elbow")}
        latency[name] = timed(run, 10)

    rows, mets = [], {}
    for m, pred in methods.items():
        mets[m] = metrics(pred, gt, hand, nom_bones)
        mets[m]["latency_ms"] = latency[m]
        mets[m]["rate_hz"] = 1000.0 / latency[m]
        if m in covs:
            for part in ("wrist", "elbow"):
                j = f"{hand}_{part}"
                mets[m][f"coverage_{part}_pct"] = float(np.mean(ck.coverage_fraction(gt[j] - pred[j],
                                                                                     covs[m][part])) * 100)
        rows.append({"subject": trial.subject, "instruction": trial.instruction_id, "method": m, **mets[m]})
    record = {"subject": trial.subject, "velocity": trial.velocity, "instruction": trial.instruction_id,
              "task": trial.task_description, "obs_ratio": obs_ratio, "pred_dur": t_rem, "H": H, "hand": hand,
              "dt": dt, "target": frames.target, "obs": obs, "gt": gt, "methods": methods, "metrics": mets,
              "cov": cov, "covs": covs, "nom_bone_lens": nom_bones,
              "wrist_error_curve": {m: np.linalg.norm(p[w] - gt[w], axis=1) * 100.0 for m, p in methods.items()}}
    return rows, record


def initial_parameters(cfg, obs_ratio):
    """(params, pred_noise, settings) of the initial weights of config/model (fully observed; model.pred_noise for
    the random walk)."""
    settings = ck.predictor_settings(cfg)
    return ck.model_params(cfg.model), ck.pred_noise_of(settings, None, obs_ratio, float(cfg.model.pred_noise)), \
        settings


def get_weights_path(cfg) -> Optional[Path]:
    """Weights to evaluate: `weights` (command-line shorthand) if set, else eval.params. None (initial weights of
    config/model) for null / none / initial; latest -> output/latest_train; otherwise a run folder or params.json,
    tried as given, under output/ and under output/train_<name>. Relative to the repository root."""
    raw = cfg.get("weights")
    if raw is None:
        raw = cfg.eval.get("params")
    if raw is None or str(raw).strip().lower() in ("null", "none", "initial"):
        return None
    raw = str(raw).strip()
    if raw.lower() in ("latest", "latest_train"):
        raw = "output/latest_train"
    output = ck.REPO_ROOT / "output"
    for p in (Path(raw), output / raw, output / f"train_{raw}"):
        p = p if p.is_absolute() else ck.REPO_ROOT / p
        if p.exists():
            return p
    runs = sorted(r.name for r in output.glob("train_*")) if output.is_dir() else []
    # Hydra parses a bare timestamp such as 20261004_200005 as the integer 20261004200005
    match = [r for r in runs if r.removeprefix("train_").replace("_", "") == raw]
    if len(match) == 1:
        return output / match[0]
    raise FileNotFoundError(f"weights {raw!r} not found (tried it as a path, under output/ and as output/train_{raw}); "
                            f"train.py runs in output/: {', '.join(runs) or 'none'}; weights=initial for config/model")


def default_output_dir(weights_path: Optional[Path]) -> Path:
    """output/eval_<run> for the weights of output/train_<run> (latest_train resolved to its run), so that a train.py
    run and its evaluation share the name; output/eval_initial_<timestamp> for the initial weights. If the folder
    already exists (the run evaluated again), a suffix _2, _3, ... is added."""
    if weights_path is None:
        name = f"eval_initial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run = weights_path.resolve()
        run = run.parent if run.is_file() else run
        name = "eval_" + run.name.removeprefix("train_")
    root = ck.REPO_ROOT / "output" / name
    k = 2
    while root.exists():
        root = ck.REPO_ROOT / "output" / f"{name}_{k}"
        k += 1
    return root


def parameters_for(cfg, obs_ratio):
    """(params, pred_noise, source, settings): the IOC-fitted weights of a train.py run (eval.params / params / weights:
    run folder or params.json) with the prediction settings of the run (observability, temperature) and of config/model
    (covariance) and, for the random-walk covariance, the noise level calibrated for obs_ratio; else the initial
    weights of config/model (model.pred_noise). pred_noise is None with the model covariance."""
    target_path = get_weights_path(cfg)
    if target_path is None:
        params, pred_noise, settings = initial_parameters(cfg, obs_ratio)
        return params, pred_noise, "initial weights (config/model)", settings
    params, record, path = ck.load_record(target_path)
    settings = ck.predictor_settings(cfg, record)
    return params, ck.pred_noise_of(settings, record, obs_ratio, float(cfg.model.pred_noise)), \
        f"IOC-fitted, {path}", settings


def describe(settings, pred_noise) -> str:
    """Short description of the predictive distribution, for the logs."""
    cov = "model covariance" + (" + residual" if settings.residual else "") if pred_noise is None \
        else f"random-walk covariance (noise {pred_noise:.3g})"
    return f"{settings.observability} observability, {cov}"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    os.chdir(ck.REPO_ROOT)  # relative paths (config, output/) refer to the repository root
    device = ck.setup_jax(cfg.eval.device)
    weights_path = get_weights_path(cfg)
    print(f"weights: {weights_path or 'initial weights of config/model'}", flush=True)
    root = Path(cfg.eval.output_dir) if cfg.eval.output_dir else default_output_dir(weights_path)
    train_subjects, test_subjects = ck.split_subjects(cfg.data)
    trials = ck.load_trials(cfg.data, test_subjects, role="test")
    tracker = Tracker(cfg.get("wandb"), "eval", root.name, OmegaConf.to_container(cfg, resolve=True))
    learned = {}
    if cfg.eval.get("baselines"):   # data-driven baselines, learned once from the training subjects' complete reaches
        if not cfg.data.get("test_subjects"):
            print("WARNING: no data.test_subjects: the data-driven baselines are trained on the evaluated reaches")
        names = [str(b) for b in cfg.eval.baselines]
        options = OmegaConf.to_container(cfg.eval.get("baseline_options") or {}, resolve=True)
        learned = ck.fit_baselines(names, ck.load_trials(cfg.data, train_subjects, role="baseline training"), options)
        print(f"data-driven baselines {', '.join(names)}: trained on the reaches of {', '.join(train_subjects)}")
    plot_keys = {str(t) for t in cfg.eval.plot_trials or []}
    overview, all_rows, curves = {}, [], {}
    for obs_ratio in [float(r) for r in cfg.data.obs_ratios]:
        params, pred_noise, source, settings = parameters_for(cfg, obs_ratio)
        initial = initial_parameters(cfg, obs_ratio) if cfg.eval.compare_initial and (weights_path is not None) else None
        out = root / ck.ratio_tag(obs_ratio)
        raw_vel = cfg.data.get("velocities") if hasattr(cfg.data, "get") else getattr(cfg.data, "velocities", None)
        if raw_vel is None:
            raw_vel = cfg.data.get("velocity", "FAST") if hasattr(cfg.data, "get") else getattr(cfg.data, "velocity", "FAST")
        vel_str = raw_vel if isinstance(raw_vel, str) else ", ".join(str(v) for v in raw_vel)
        print(f"\n=== {obs_ratio:.0%} observed | {len(trials)} {vel_str} reaches of {', '.join(test_subjects)}"
              f" | parameters: {source} | {describe(settings, pred_noise)} | device {device}", flush=True)
        rows, figures = [], []
        with jax.default_device(device):
            for k, trial in enumerate(trials, 1):
                r, record = evaluate_trial(trial, obs_ratio, params, pred_noise, cfg, initial, learned, settings)
                rows += r
                for m, c in record["wrist_error_curve"].items():
                    curves.setdefault(m, []).append(c)
                kin = record["metrics"]["kin"]
                print(f"  [{k:2d}/{len(trials)}] {trial.subject:6s} inst{trial.instruction_id} | kin MPJPE "
                      f"{kin['mpjpe_cm']:5.2f} cm, wrist FDE {kin['wrist_fde_cm']:5.2f} cm, wrist coverage "
                      f"{kin['coverage_wrist_pct']:3.0f}% | best baseline MPJPE "
                      f"{min(record['metrics'][m]['mpjpe_cm'] for m in record['metrics'] if m != 'kin'):5.2f} cm",
                      flush=True)
                if f"{trial.subject}/{trial.instruction_id}" in plot_keys:
                    figures += plot_trial(record, out)
        vel_meta = list(raw_vel) if not isinstance(raw_vel, str) else raw_vel
        meta = {"n_trials": len(trials), "velocities": vel_meta, "velocity": vel_meta, "subjects": test_subjects,
                "instructions": list(cfg.data.instructions), "obs_ratio": obs_ratio, "params_source": source,
                "params": ck.params_to_dict(params), "pred_noise": pred_noise,
                "prediction": dict(vars(settings)),
                "config": OmegaConf.to_container(cfg, resolve=True)}
        save_results(rows, meta, out)
        all_rows += rows
        overview[obs_ratio] = summarize(rows)
        print_summary(overview[obs_ratio])
        tag = ck.ratio_tag(obs_ratio)
        tracker.summary({f"{tag}/{m}/{key}": v["mean"] for m, s in overview[obs_ratio].items()
                         for key, v in s.items()})
        if figures:
            print(f"  figures: {out / 'html'}, frames: {out / 'frames'}")
    save_overview(overview, root)
    labels = {m: label for m, label in METHODS}
    fig_dir = root / "figures"
    ioc_plots.plot_eval_overview(overview, labels, fig_dir / "errors_vs_observed_fraction.png")
    ioc_plots.plot_by_instruction(all_rows, labels, fig_dir / "errors_by_instruction.png")
    ioc_plots.plot_error_vs_time({m: np.mean(c, axis=0) for m, c in curves.items()}, labels,
                                 fig_dir / "wrist_error_along_prediction.png")
    keys = sorted({key for s in overview.values() for ms in s.values() for key in ms})
    tracker.table("results", ["obs_ratio", "method"] + keys,
                  [[r, m] + [ms.get(key, {}).get("mean") for key in keys] for r, s in overview.items()
                   for m, ms in s.items()])
    # Figures are saved locally to figures/ (offline) and not uploaded online
    tracker.finish()
    latest = Path("output/latest")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(root.resolve(), target_is_directory=True)
    print(f"\nResults in {root} (output/latest): summary.html, summary.csv and one folder per observed fraction")


if __name__ == "__main__":
    main()
