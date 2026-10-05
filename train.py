#!/usr/bin/env python3
"""IOC fit of the 19-DOF kinematic predictor's cost weights on CARI v2 demonstrations, and calibration of its
prediction uncertainty.

The training subjects are config data.subjects minus data.test_subjects (held out for eval.py); every selected
instruction of every training subject is a demonstration (complete reach).
- ioc.objective = likelihood (default) | open_loop: the learnable cost weights are fitted on the training windows of
  every demonstration, from the handover points data.obs_ratios to the end of the reach (GPU, parallel restarts
  from the initial weights of config/model and random points, progress bar): likelihood = negative log-likelihood of
  the probabilistic IOC model with the policy solved per window (ioc.observability full | partial,
  ioc.linearization solve | data), open_loop = open-loop keypoint error (ablation); none: the initial weights are
  kept. The window states come from data.state_estimator (rts | savgol).
- The predictions use the model's own predictive covariance (model.prediction_covariance: model, see
  prophet_ioc.human_prediction); for the random-walk ablation (model.prediction_covariance: random_walk) the
  joint-velocity noise of that covariance (pred_noise) is calibrated for every observed fraction in data.obs_ratios
  so that 95 % of the wrist / elbow prediction errors of the training subjects fall in the 95 % ellipsoids
  (leave-one-subject-out among them; the value for a new person is the pooled one), and the coverage of the model
  covariance on the same errors is reported.
- Two stages for ioc.objective = open_loop (ioc.noise_fit): the open-loop objective fits the cost weights only, so the
  noise of the predictive distribution is fitted next, with those weights held fixed; otherwise the residual noise
  keeps its initial value and the predicted covariance is inflated. noise_fit.objective: predictive (default:
  residual_noise and handover_cov_scale maximize the likelihood of the recorded wrist / elbow positions under the
  published predictive distribution, ck.fit_predictive_noise, seconds after one pass of predictions) | likelihood
  (residual_noise, + obs_noise when partially observed, by the one-step IOC likelihood).
Resources (ioc.resources, evaluation/resources.py): lower priority, a few cores left free and a memory watchdog that
stops the run (exit code 137, with the reason) when its RAM exceeds ioc.resources.max_ram_gb (default 80 % of the
physical RAM) or the system's available RAM falls below ioc.resources.min_available_gb, before the machine swaps.
Results: output/train_<timestamp>/ with params.json (fitted weights, pred_noise per observed fraction, fit history),
figures/ (convergence, parameters, open-loop error before / after the fit, example predictions) and
output/latest_train; eval.py uses the latest run by default.

    python train.py                                        # fully observed likelihood (solve-based)
    python train.py ioc.observability=partial              # partially observed model (belief, Algorithm 1)
    python train.py ioc.objective=open_loop                # open-loop error (ablation)
    python train.py ioc.objective=none                     # initial weights, calibration only
    python train.py 'data.test_subjects=[sub_4]'           # another held-out subject
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Triton GEMM autotuning tries kernels with GiB-sized workspaces, which ran out of the 8 GB GPU memory
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
# Triton GEMM autotuning tries kernels with GiB-sized workspaces, which ran out of the 8 GB GPU memory.
# Multi-threaded XLA compilation parallelism utilizes available CPU cores to speed up LLVM code generation.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false --xla_gpu_force_compilation_parallelism=16")
# Single-threaded OpenBLAS (LAPACK of JAX's CPU linear algebra, calibration predictions): see eval.py
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))  # cari_kinematic, save_results

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

import cari_kinematic as ck
import resources
from plot_utils import ioc_plots
from tracking import Tracker
from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicReaching


def example_figure(trial, base, params, ratios, H, max_iter, path, tol=None, settings=None):
    """Wrist path of one demonstration predicted from each observed fraction, initial vs fitted weights."""
    frames = ck.trial_frames(trial)
    w = f"{trial.reaching_hand}_wrist"
    gt = frames.keypoint(w)[trial.onset_idx: trial.offset_idx + 1]
    f_obs = [ck.obs_frame(trial, r) for r in ratios]
    preds = {}
    for name, p in (("init", base), ("fit", params)):
        preds[name] = [ck.kinematic_inference(trial, frames, f, p, H, max_iter, trial.reaching_hand, tol,
                                              settings).joints[w] for f in f_obs]
    return ioc_plots.plot_example(gt, [f - trial.onset_idx for f in f_obs], preds["init"], preds["fit"],
                                  f"{trial.subject} instruction {trial.instruction_id}", path)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    os.chdir(ck.REPO_ROOT)  # relative paths (config, output/) refer to the repository root
    # memory watchdog, priority and CPU affinity before the first JAX computation (XLA sizes its thread pools from
    # the affinity): the run stops itself instead of pushing the machine into swap (evaluation/resources.py)
    res = cfg.ioc.get("resources") or {}
    resources.guard(max_ram_gb=res.get("max_ram_gb"), max_ram_fraction=float(res.get("max_ram_fraction", 0.8)),
                    min_available_gb=float(res.get("min_available_gb", 3.0)), free_cpus=int(res.get("free_cpus", 2)),
                    nice=int(res.get("nice", 10)))
    fit_device = ck.setup_jax(cfg.ioc.device)
    cpu = ck.setup_jax("cpu")
    root = Path(cfg.ioc.output_dir or Path("output") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    train_subjects, test_subjects = ck.split_subjects(cfg.data)
    trials = ck.load_trials(cfg.data, train_subjects, role="training")
    base = ck.model_params(cfg.model)
    H, max_iter, tol = int(cfg.model.horizon), int(cfg.model.max_iter), ck.solver_tolerance(cfg.model)
    raw_vel = cfg.data.get("velocities") if hasattr(cfg.data, "get") else getattr(cfg.data, "velocities", None)
    if raw_vel is None:
        raw_vel = cfg.data.get("velocity", "FAST") if hasattr(cfg.data, "get") else getattr(cfg.data, "velocity", "FAST")
    vel_str = raw_vel if isinstance(raw_vel, str) else ", ".join(str(v) for v in raw_vel)
    print(f"=== IOC fit ({cfg.ioc.objective}) on {len(trials)} {vel_str} demonstrations of "
          f"{len(train_subjects)} subjects ({', '.join(train_subjects)}); held out: {', '.join(test_subjects)}",
          flush=True)
    starts = ck.window_starts(cfg.data)
    print(f"    training windows from the observed fractions {', '.join(f'{r:.0%}' for r in starts)} of each reach to its "
          f"end ({cfg.ioc.T_fit} steps): {len(trials)} reaches x {len(starts)} = {len(trials) * len(starts)} windows",
          flush=True)

    tracker = Tracker(cfg.get("wandb"), "train", root.name, OmegaConf.to_container(cfg, resolve=True))

    def make_logger(stage: str, obj: str, step0: int = 0):
        """Per-iteration record of ck.fit_params -> wandb under stage/ (fit, noise_fit): loss and gradient norm of
        every restart, best loss, weights of the best point (and of every restart, log10). step0: offset of the
        steps (wandb steps must increase across the stages)."""
        def log_iteration(it, rec):
            data = {
                f"{stage}/{obj}_best_loss": rec["best_loss"],
                f"{stage}/best_loss": rec["best_loss"],
                f"{stage}/best_restart": rec["best_restart"],
                f"{stage}/iteration_time_s": rec["iteration_time_s"],
                f"{stage}/elapsed_min": rec["elapsed_s"] / 60.0,
            }
            for r, (v, gn) in enumerate(zip(rec["loss"], rec["grad_norm"])):
                data[f"{stage}/{obj}_loss_restart{r}"] = v
                data[f"{stage}/loss_restart{r}"], data[f"{stage}/grad_norm_restart{r}"] = v, gn
            for i, k in enumerate(rec["infer"]):
                data[f"params/{k}"] = 10.0 ** rec["best_log10_params"][i]
                for r, th in enumerate(rec["log10_params"]):
                    data[f"log10_params_restart{r}/{k}"] = th[i]
            tracker.log(data, step=step0 + it)
        return log_iteration

    vel_meta = list(raw_vel) if not isinstance(raw_vel, str) else raw_vel
    info = {"objective": cfg.ioc.objective, "train_subjects": train_subjects, "test_subjects": test_subjects,
            "instructions": list(cfg.data.instructions), "velocities": vel_meta, "velocity": vel_meta,
            "temperature": float(cfg.ioc.get("temperature", 1e-6)), "initial_params": ck.params_to_dict(base)}
    params = base
    if cfg.ioc.objective != "none":
        with jax.default_device(fit_device):
            params, fit_info = ck.fit_params(trials, base, cfg.ioc, cfg.model, cfg.data,
                                             callback=make_logger("fit", cfg.ioc.objective) if tracker.enabled
                                             else None)
        tracker.summary({"loss_base": fit_info["loss_base"], "loss_fit": fit_info["loss_fit"],
                         "iterations": fit_info["iterations"], "fit_time_min": fit_info["fit_time_s"] / 60.0,
                         **{f"fitted/{k}": ck.params_to_dict(params)[k] for k in fit_info["infer"]}})
        info.update(fit_info)
        print(f"  loss per segment: initial weights {fit_info['loss_base']:.4g} -> fitted {fit_info['loss_fit']:.4g} "
              f"(restart {fit_info['best_restart']}, {fit_info['iterations']} iterations, "
              f"{fit_info['fit_time_s']:.0f} s)")
        print(f"  {'parameter':22s} {'initial':>10s} {'fitted':>10s}")
        for k in fit_info["infer"]:
            print(f"  {k:22s} {ck.params_to_dict(base)[k]:10.3g} {ck.params_to_dict(params)[k]:10.3g}")
        resources.release()   # the compiled objective (several GB of host RAM for the likelihood) is not needed any more
        print(f"  after the fit: {resources.status()}", flush=True)

    # Stage 2 (open-loop fits): the noise levels of the predictive distribution by the likelihood, with the fitted cost
    # weights held fixed. The open-loop objective fits no noise (the residual noise would keep its initial value and
    # inflate the predicted covariance); the likelihood of the recorded transitions does.
    nf = cfg.ioc.get("noise_fit") or {}
    if cfg.ioc.objective == "open_loop" and nf.get("enabled", True) and str(nf.get("objective", "predictive")) == \
            "predictive":
        # predictive: residual noise and handover covariance scale maximize the likelihood of the recorded wrist and
        # elbow positions under the predictive distribution the model publishes (open-loop predictions from every
        # handover point of the training reaches, as eval.py), not the one-step transitions of the IOC likelihood
        names = list(nf.get("params") or ck.PREDICTIVE_NOISE_PARAMS)
        ratios = [float(x) for x in cfg.data.obs_ratios]
        print(f"=== stage 2: noise levels ({', '.join(names)}) by the predictive likelihood of the wrist and elbow "
              f"(predictions from {', '.join(f'{r:.0%}' for r in ratios)} observed of the {len(trials)} training "
              f"reaches, CPU), cost weights of the open-loop fit held fixed", flush=True)
        with jax.default_device(cpu):
            params, noise_info = ck.fit_predictive_noise(trials, params, ratios, H, max_iter, tol,
                                                         ck.predictor_settings(cfg, info), names)
        info["noise_fit"] = noise_info
        tracker.summary({f"fitted/{k}": v for k, v in noise_info["after"].items()})
        print(f"  negative log-likelihood per sample: {noise_info['loss_base']:.4g} -> {noise_info['loss_fit']:.4g}; "
              f"95 % coverage of the wrist / elbow (training reaches): {noise_info['coverage_before']:.0%} -> "
              f"{noise_info['coverage_after']:.0%} ({noise_info['samples']} samples, {noise_info['fit_time_s']:.0f} s)")
        for tag, v in noise_info["per_ratio"].items():
            print(f"    {tag}: coverage {v['coverage_before']:.0%} -> {v['coverage_after']:.0%}")
        for k in names:
            print(f"  {k:22s} {noise_info['before'][k]:10.3g} {noise_info['after'][k]:10.3g}")
    elif cfg.ioc.objective == "open_loop" and nf.get("enabled", True):
        names = list(nf.get("params") or (["residual_noise"] + (["obs_noise"] if cfg.ioc.get("observability", "full")
                                                                == "partial" else [])))
        ncfg = OmegaConf.merge(cfg.ioc, OmegaConf.create({
            "objective": "likelihood", "params": names, "restarts": int(nf.get("restarts", 1)),
            "max_iter": int(nf.get("max_iter", 60)), "batch_size": nf.get("batch_size")}))
        print(f"=== stage 2: noise levels ({', '.join(names)}) by the likelihood, cost weights of the open-loop fit "
              f"held fixed ({ncfg.restarts} restart(s), at most {ncfg.max_iter} iterations)", flush=True)
        before = ck.params_to_dict(params)
        with jax.default_device(fit_device):
            params, noise_info = ck.fit_params(trials, params, ncfg, cfg.model, cfg.data,
                                               callback=make_logger("noise_fit", "likelihood",
                                                                    step0=info.get("iterations", 0) + 1)
                                               if tracker.enabled else None)
        info["noise_fit"] = {k: noise_info[k] for k in ("infer", "loss_base", "loss_fit", "iterations", "fit_time_s",
                                                         "best_restart", "loss_history") if k in noise_info}
        tracker.summary({f"fitted/{k}": ck.params_to_dict(params)[k] for k in names})
        print(f"  negative log-likelihood per segment: {noise_info['loss_base']:.4g} -> {noise_info['loss_fit']:.4g} "
              f"({noise_info['iterations']} iterations, {noise_info['fit_time_s']:.0f} s)")
        for k in names:
            print(f"  {k:22s} {before[k]:10.3g} {ck.params_to_dict(params)[k]:10.3g}")
        resources.release()
        print(f"  after the noise fit: {resources.status()}", flush=True)

    noise = {}
    settings = ck.predictor_settings(cfg, info)   # observability and temperature of this fit
    info["prediction"] = dict(vars(settings))
    with jax.default_device(cpu):
        for r in [float(x) for x in cfg.data.obs_ratios]:
            sigma, cal = ck.calibrate_pred_noise(trials, r, params, H, max_iter, tol=tol, settings=settings)
            noise[ck.ratio_tag(r)] = {"pred_noise": sigma, **{k: v for k, v in cal.items() if k != "pred_noise"}}
            print(f"  {r:.0%} observed: model covariance coverage {cal['coverage_model_covariance']:.0%}; random-walk "
                  f"noise {sigma:.3g} ({cal['calibration_samples']} samples, "
                  f"{cal['coverage_handover_covariance_only']:.0%} coverage with the handover covariance only)",
                  flush=True)
    tracker.summary({f"pred_noise/{k}": v["pred_noise"] for k, v in noise.items()})
    tracker.summary({f"coverage_model_covariance/{k}": v["coverage_model_covariance"] for k, v in noise.items()})
    info["pred_noise"] = {k: v["pred_noise"] for k, v in noise.items()}
    info["calibration"] = noise
    ck.save_params(root / "params.json", params, info)

    # figures
    fig_dir = root / "figures"
    figures = []
    if cfg.ioc.objective != "none":
        figures.append(ioc_plots.plot_convergence(info, fig_dir / "ioc_convergence.png"))
        figures.append(ioc_plots.plot_parameters(ck.params_to_dict(base), ck.params_to_dict(params), info,
                                                 HumanKinematicReaching.get_params_bounds(),
                                                 fig_dir / "ioc_parameters.png"))
        print("  diagnostics: open-loop error of the training windows with the initial and the fitted weights",
              flush=True)
        with jax.default_device(fit_device):
            err_init, err_fit = ck.segment_errors_many(trials, [base, params], cfg.ioc, cfg.data,
                                                       labels=["initial", "fitted"])
        (root / "segment_errors.json").write_text(json.dumps(
            {"init": {k: v.tolist() for k, v in err_init.items()}, "fit": {k: v.tolist() for k, v in err_fit.items()}}))
        tracker.summary({"train_rms_cm/initial": float(err_init["rms_cm"].mean()),
                         "train_rms_cm/fitted": float(err_fit["rms_cm"].mean())})
        print(f"  open-loop RMS joint error of the training segments: initial {err_init['rms_cm'].mean():.2f} cm -> "
              f"fitted {err_fit['rms_cm'].mean():.2f} cm")
        figures.append(ioc_plots.plot_fit_quality(err_init, err_fit, fig_dir / "ioc_fit_quality.png"))
        with jax.default_device(cpu):
            for tr in [t for t in trials if t.instruction_id in (1, 3)][:2]:
                print(f"  diagnostics: example predictions of {tr.subject} instruction {tr.instruction_id} "
                      f"(initial and fitted weights, CPU; the first one compiles the predictor)", flush=True)
                figures.append(example_figure(tr, base, params, [float(x) for x in cfg.data.obs_ratios], H, max_iter,
                                              fig_dir / f"example_{tr.subject}_inst{tr.instruction_id}.png", tol,
                                              settings))

    (root / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    latest = Path("output/latest_train")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(root.resolve(), target_is_directory=True)
    # Figures are saved locally to figures/ (offline) and not uploaded online
    tracker.finish()
    print(f"\nSaved to {root} (output/latest_train): params.json, figures/ ({len(figures)} figures)")


if __name__ == "__main__":
    main()
