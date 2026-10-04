#!/usr/bin/env python3
"""IOC fit of the 19-DOF kinematic predictor's cost weights on CARI v2 demonstrations, and calibration of its
prediction uncertainty.

The training subjects are config data.subjects minus data.test_subjects (held out for eval.py); every selected
instruction of every training subject is a demonstration (complete reach).
- ioc.objective = open_loop | likelihood: the cost weights are fitted on the segments of every demonstration from
  the handover points ioc.segment_starts to the end of the reach (GPU, parallel restarts from the initial weights of
  config/model and random points, progress bar); none: the initial weights are kept.
- The joint-velocity noise of the prediction covariance (pred_noise) is calibrated for every observed fraction in
  data.obs_ratios so that 95 % of the wrist / elbow prediction errors of the training subjects fall in the 95 %
  ellipsoids (leave-one-subject-out among them; the value for a new person is the pooled one).
Results: output/train_<timestamp>/ with params.json (fitted weights, pred_noise per observed fraction, fit history),
figures/ (convergence, parameters, open-loop error before / after the fit, example predictions) and
output/latest_train; eval.py uses the latest run by default.

    python train.py                                        # open-loop objective
    python train.py ioc.objective=none                     # initial weights, calibration only
    python train.py 'data.test_subjects=[sub_4]'           # another held-out subject
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Triton GEMM autotuning tries kernels with GiB-sized workspaces, which ran out of the 8 GB GPU memory
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

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
from plot_utils import ioc_plots
from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicReaching


def example_figure(trial, base, params, ratios, H, max_iter, path):
    """Wrist path of one demonstration predicted from each observed fraction, initial vs fitted weights."""
    frames = ck.trial_frames(trial)
    w = f"{trial.reaching_hand}_wrist"
    gt = frames.keypoint(w)[trial.onset_idx: trial.offset_idx + 1]
    f_obs = [ck.obs_frame(trial, r) for r in ratios]
    preds = {}
    for name, p in (("init", base), ("fit", params)):
        preds[name] = [ck.kinematic_inference(trial, frames, f, p, H, max_iter, trial.reaching_hand).joints[w]
                       for f in f_obs]
    return ioc_plots.plot_example(gt, [f - trial.onset_idx for f in f_obs], preds["init"], preds["fit"],
                                  f"{trial.subject} instruction {trial.instruction_id}", path)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    os.chdir(ck.REPO_ROOT)  # relative paths (config, output/) refer to the repository root
    fit_device = ck.setup_jax(cfg.ioc.device)
    cpu = ck.setup_jax("cpu")
    root = Path(cfg.ioc.output_dir or Path("output") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    train_subjects, test_subjects = ck.split_subjects(cfg.data)
    trials = ck.load_trials(cfg.data, train_subjects)
    base = ck.model_params(cfg.model)
    H, max_iter = int(cfg.model.horizon), int(cfg.model.max_iter)
    print(f"=== IOC fit ({cfg.ioc.objective}) on {len(trials)} {cfg.data.velocity} demonstrations of "
          f"{len(train_subjects)} subjects ({', '.join(train_subjects)}); held out: {', '.join(test_subjects)}",
          flush=True)

    info = {"objective": cfg.ioc.objective, "train_subjects": train_subjects, "test_subjects": test_subjects,
            "instructions": list(cfg.data.instructions), "velocity": cfg.data.velocity,
            "initial_params": ck.params_to_dict(base)}
    params = base
    if cfg.ioc.objective != "none":
        with jax.default_device(fit_device):
            params, fit_info = ck.fit_params(trials, base, cfg.ioc)
        info.update(fit_info)
        print(f"  loss per segment: initial weights {fit_info['loss_base']:.4g} -> fitted {fit_info['loss_fit']:.4g} "
              f"(restart {fit_info['best_restart']}, {fit_info['iterations']} iterations, "
              f"{fit_info['fit_time_s']:.0f} s)")
        print(f"  {'parameter':22s} {'initial':>10s} {'fitted':>10s}")
        for k in fit_info["infer"]:
            print(f"  {k:22s} {ck.params_to_dict(base)[k]:10.3g} {ck.params_to_dict(params)[k]:10.3g}")

    noise = {}
    with jax.default_device(cpu):
        for r in [float(x) for x in cfg.data.obs_ratios]:
            sigma, cal = ck.calibrate_pred_noise(trials, r, params, H, max_iter)
            noise[ck.ratio_tag(r)] = {"pred_noise": sigma, **{k: v for k, v in cal.items() if k != "pred_noise"}}
            print(f"  {r:.0%} observed: prediction noise {sigma:.3g} ({cal['calibration_samples']} samples, "
                  f"{cal['coverage_handover_covariance_only']:.0%} coverage with the handover covariance only)",
                  flush=True)
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
        with jax.default_device(fit_device):
            err_init = ck.segment_errors(trials, base, cfg.ioc)
            err_fit = ck.segment_errors(trials, params, cfg.ioc)
        (root / "segment_errors.json").write_text(json.dumps(
            {"init": {k: v.tolist() for k, v in err_init.items()}, "fit": {k: v.tolist() for k, v in err_fit.items()}}))
        print(f"  open-loop RMS joint error of the training segments: initial {err_init['rms_cm'].mean():.2f} cm -> "
              f"fitted {err_fit['rms_cm'].mean():.2f} cm")
        figures.append(ioc_plots.plot_fit_quality(err_init, err_fit, fig_dir / "ioc_fit_quality.png"))
        with jax.default_device(cpu):
            for tr in [t for t in trials if t.instruction_id in (1, 3)][:2]:
                figures.append(example_figure(tr, base, params, [float(x) for x in cfg.data.obs_ratios], H, max_iter,
                                              fig_dir / f"example_{tr.subject}_inst{tr.instruction_id}.png"))

    (root / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    latest = Path("output/latest_train")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(root.resolve(), target_is_directory=True)
    print(f"\nSaved to {root} (output/latest_train): params.json, figures/ ({len(figures)} figures)")


if __name__ == "__main__":
    main()
