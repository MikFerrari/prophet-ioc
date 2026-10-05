#!/usr/bin/env python3
"""Parameter recovery on data simulated from the kinematic model itself (config `synthetic`): can the IOC objectives
of train.py recover known cost weights when the model is exactly right?

The training windows of train.py (fit_segments: start state, reaching wrist target, time step and hand of each window
of the training subjects' reaches) are kept, but their trajectories are replaced by simulations of the fully observed
model (multi_env.simulate_trial: the agent solves its policy from the window start with the gILQR of the likelihood,
then acts with it through the signal-dependent motor noise and the max-ent policy noise) with known true weights: the
initial weights of config/model, each shifted by U(-offset, offset) decades (seeded; restart 0 of every fit starts from
the unshifted initial weights, so no fit starts at the truth). The true residual noise is its lower bound (no model
mismatch).

Every objective of synthetic.objectives is then fitted with ck.fit_params (ioc.* settings, as train.py), and
evaluated on held-out windows of the test subjects simulated with the same true weights:
- recovery: fitted vs true weights, error in decades (log10);
- open-loop RMS (cm, the 9 MPJPE joints, as eval.py and the open-loop objective): the true weights give the noise floor;
- negative log-likelihood per window (fully observed, policy solved per window), with the true residual noise for the
  parameter sets that do not fit it.
If the likelihood recovers the weights here but predicts worse than the open-loop fit on CARI, the gap there comes
from model mismatch, not from the objective or the optimizer.

    python evaluation/synthetic_recovery.py                          # -> output/synthetic_<timestamp>/
    python evaluation/synthetic_recovery.py synthetic.n_trials=20 ioc.max_iter=60   # quicker
    python evaluation/synthetic_recovery.py synthetic.objectives=[likelihood]

Progress: timestamped stage lines with elapsed time and RAM (also in <output>/log.txt), a progress bar over the
simulation chunks and over the fit iterations (ck.fit_params); the true weights and each fit are saved as soon as they
are known (true.json, fit_<objective>.json), so a stopped run keeps what it finished: run it again with the same
synthetic.output_dir (and settings) and the saved fits are loaded instead of refitted (the simulations are seeded).
Resources (synthetic.resources, evaluation/resources.py): lower priority, a few cores left free and a memory watchdog
that stops the run before the desktop starts swapping (a run of 4 October grew to 10.7 GB and froze the machine).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# as train.py: Triton GEMM autotuning tries GiB-sized workspaces on the 8 GB GPU; single-threaded OpenBLAS (must be set
# before numpy is imported)
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import cari_kinematic as ck
import resources
from prophet_ioc.envs.human_kinematic_reaching import (HumanKinematicParams, HumanKinematicReaching,
                                                     learnable_params, stack_envs)
from prophet_ioc.infer import MultiTrialLikelihood
from prophet_ioc.infer.multi_env import simulate_trial, trial_open_loop_error


def true_parameters(base, names, offset: float, seed: int, margin: float = 0.25):
    """The initial weights shifted by U(-offset, offset) decades each, kept `margin` decades inside the fit bounds;
    residual noise at its lower bound (the simulated data have no model mismatch)."""
    lo, hi = HumanKinematicReaching.get_params_bounds()
    rng = np.random.default_rng(seed)
    true = {}
    for k in names:
        lo_k, hi_k = np.log10(getattr(lo, k)) + margin, np.log10(getattr(hi, k)) - margin
        theta = np.clip(np.log10(max(getattr(base, k), 1e-12)), lo_k, hi_k) + rng.uniform(-offset, offset)
        true[k] = float(10.0 ** np.clip(theta, lo_k, hi_k))
    return base._replace(**true, residual_noise=float(lo.residual_noise))


def by_hand(segments):
    """{hand: indices of the segments} (stack_envs needs the same static configuration)."""
    groups = {}
    for i, s in enumerate(segments):
        groups.setdefault(s[2].reaching_hand, []).append(i)
    return groups


def simulate(segments, params, key, T: int, temperature: float, solve_iters: int, chunk: int = 8, desc: str = ""):
    """The segments with their trajectories replaced by simulations of the model from the same start states; windows
    whose simulation is not finite are dropped (their number is returned).

    Simulated `chunk` windows at a time (the last chunk padded by repetition, so that every chunk reuses one compiled
    program): memory bounded by the chunk instead of the number of windows, and a progress bar."""
    sim = jax.jit(jax.vmap(lambda env, x0, k: simulate_trial(env, params, k, T, temperature, solve_iters, x0=x0)))
    out = [None] * len(segments)
    groups = list(by_hand(segments).values())
    bar = tqdm(total=len(segments), desc=f"simulate {desc}".strip(), unit="window")
    for idx in groups:
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, len(idx))
        for c in range(0, len(idx), chunk):
            part = idx[c:c + chunk]
            pad = part + [part[-1]] * (chunk - len(part))
            envs = stack_envs([segments[i][2] for i in pad])
            x0 = jnp.asarray(np.stack([segments[i][0][0] for i in pad]))
            k = jnp.concatenate([keys[c:c + len(part)], jnp.repeat(keys[c + len(part) - 1][None], chunk - len(part), 0)])
            xs = np.array(sim(envs, x0, k))[:len(part)]
            for i, x in zip(part, xs):
                out[i] = (x.astype(np.float32), segments[i][1], segments[i][2])
            bar.update(len(part))
    bar.close()
    kept = [s for s in out if np.all(np.isfinite(s[0]))]
    return kept, len(out) - len(kept)


class TestMetrics:
    """Open-loop RMS error (cm, 9 joints; ck.segment_errors on given segments: the model's optimal trajectory from the
    window start vs the simulated one) and negative log-likelihood per window of the test segments, for any parameter
    set. The parameters are an argument of the compiled functions, not constants baked into them, so each function is
    compiled once and reused for every parameter set (one compilation of the likelihood takes several GB of host RAM
    and about a minute)."""

    def __init__(self, segments, params0, ioc_cfg):
        self.n = len(segments)
        self.idx = list(by_hand(segments).values())
        self.groups = [(stack_envs([segments[i][2] for i in idx]), jnp.asarray(np.stack([segments[i][0] for i in idx])))
                       for idx in self.idx]
        self._ol = jax.jit(lambda e, x, p: jax.lax.map(
            lambda ex: trial_open_loop_error(ex[0], ex[1], p, ck.upper_body_output), (e, x), batch_size=16))
        # infer = every field: loglikelihood(None, p) evaluates p itself (with infer=() it would use params0)
        self.lik = MultiTrialLikelihood(self.groups, params0, tuple(params0._fields),
                                        solve_iters=int(ioc_cfg.solve_iters), batch_size=4,
                                        temperature=float(ioc_cfg.temperature))
        self._nll = jax.jit(lambda p, g: self.lik.loglikelihood(None, p, g))

    def open_loop_rms_cm(self, params) -> np.ndarray:
        rms = np.zeros(self.n)
        for idx, (env, x) in zip(self.idx, self.groups):
            rms[idx] = 100.0 * np.sqrt(np.array(self._ol(env, x, params)) / 9.0)
        return rms

    def nll_per_window(self, params) -> float:
        return -float(self._nll(params, self.lik.groups)) / self.n


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    os.chdir(ck.REPO_ROOT)
    syn = cfg.synthetic
    root = Path(syn.output_dir or Path("output") / f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)
    log = resources.StageLog(root / "log.txt")
    res = syn.get("resources") or {}
    # before the first JAX computation: XLA sizes its CPU thread pools from the CPU affinity when the backend starts
    resources.guard(max_ram_gb=res.get("max_ram_gb"), max_ram_fraction=float(res.get("max_ram_fraction", 0.4)),
                    min_available_gb=float(res.get("min_available_gb", 3.0)), free_cpus=int(res.get("free_cpus", 4)),
                    nice=int(res.get("nice", 10)), on_exit=log.write)
    device = ck.setup_jax(syn.device)
    log(f"output {root}, device {device}")
    (root / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    train_subjects, test_subjects = ck.split_subjects(cfg.data)
    trials = ck.load_trials(cfg.data, train_subjects, role="training")
    if syn.n_trials:
        pick = np.random.default_rng(syn.seed).choice(len(trials), min(int(syn.n_trials), len(trials)), replace=False)
        trials = [trials[i] for i in sorted(pick)]
    test_trials = ck.load_trials(cfg.data, test_subjects, role="test")
    log(f"data: {len(trials)} training reaches ({', '.join(train_subjects)}), {len(test_trials)} test reaches "
        f"({', '.join(test_subjects)})")

    base = ck.model_params(cfg.model)
    weights = learnable_params(dict(cfg.model))
    true = true_parameters(base, weights, float(syn.offset), int(syn.seed))
    T, temp, iters = int(cfg.ioc.T_fit), float(cfg.ioc.temperature), int(cfg.ioc.solve_iters)
    key_train, key_test = jax.random.split(jax.random.PRNGKey(int(syn.seed)))

    with jax.default_device(device):
        windows = lambda trs: [seg for tr in trs for seg in ck.fit_segments(tr, ck.window_starts(cfg.data), T, cfg.data,
                                                                             base.damping)]
        chunk = int(syn.get("sim_chunk", 8))
        log("simulating the training windows with the true weights (the first chunk compiles)")
        train, dropped_train = simulate(windows(trials), true, key_train, T, temp, iters, chunk, "train")
        resources.release()
        log("simulating the test windows")
        test, dropped_test = simulate(windows(test_trials), true, key_test, T, temp, iters, chunk, "test")
        resources.release()
        (root / "true.json").write_text(json.dumps({"true": ck.params_to_dict(true), "initial": ck.params_to_dict(base),
                                                    "n_train_windows": len(train), "n_test_windows": len(test)},
                                                   indent=2))
        log(f"=== synthetic recovery: {len(train)} training windows ({len(trials)} reaches of "
              f"{', '.join(train_subjects)}), {len(test)} test windows ({', '.join(test_subjects)}); "
              f"dropped non-finite simulations: {dropped_train} train, {dropped_test} test")
        print(f"  {'weight':22s} {'initial':>10s} {'true':>10s}  (offset {syn.offset} decades)")
        for k in weights:
            print(f"  {k:22s} {getattr(base, k):10.3g} {getattr(true, k):10.3g}")

        fitted, infos = {}, {}
        for n_obj, obj in enumerate(syn.objectives, 1):
            ioc_cfg = OmegaConf.merge(cfg.ioc, {"objective": str(obj)})
            saved = root / f"fit_{obj}.json"
            if saved.exists():   # a run stopped later (e.g. by the memory guard), restarted with the same output_dir
                rec = json.loads(saved.read_text())
                fitted[str(obj)], infos[str(obj)] = HumanKinematicParams(**rec["params"]), rec["info"]
                log(f"--- fit {n_obj}/{len(syn.objectives)}: {obj} loaded from {saved} (delete it to refit)")
                continue
            log(f"--- fit {n_obj}/{len(syn.objectives)}: {obj} ({len(train)} windows, max {ioc_cfg.max_iter} "
                f"iterations, {ioc_cfg.restarts} restarts)")
            fitted[str(obj)], infos[str(obj)] = ck.fit_params(trials, base, ioc_cfg, cfg.model, cfg.data,
                                                              segments=train)
            info = infos[str(obj)]
            log(f"fit {obj} done: loss {info['loss_base']:.4g} -> {info['loss_fit']:.4g}, {info['iterations']} "
                f"iterations, {info['fit_time_s'] / 60:.1f} min")
            resources.release()   # the compiled objective of this fit is not needed any more
            (root / f"fit_{obj}.json").write_text(json.dumps({"params": ck.params_to_dict(fitted[str(obj)]),
                                                              "info": info}, indent=2))

        sets = {"true": true, "initial": base._replace(residual_noise=true.residual_noise),
                **{f"fit_{o}": p if "residual_noise" in infos[o]["infer"] else p._replace(residual_noise=true.residual_noise)
                   for o, p in fitted.items()}}
        ol, nll = {}, {}
        metrics = TestMetrics(test, true, cfg.ioc)
        for n_set, (name, p) in enumerate(sets.items(), 1):
            log(f"test windows, parameters {n_set}/{len(sets)} ({name}): open-loop error"
                + (" (compiling)" if n_set == 1 else ""))
            ol[name] = metrics.open_loop_rms_cm(p)
            log(f"test windows, parameters {n_set}/{len(sets)} ({name}): negative log-likelihood"
                + (" (compiling)" if n_set == 1 else ""))
            nll[name] = metrics.nll_per_window(p)
            log(f"  {name}: open-loop RMS {ol[name].mean():.3f} cm, NLL/window {nll[name]:.2f}")

    names = list(weights) + ["residual_noise"]
    lo = HumanKinematicReaching.get_params_bounds()[0]   # a weight of 0 (switched-off initial value) counts as its bound
    log10 = lambda p, k: float(np.log10(max(getattr(p, k), getattr(lo, k))))
    err = {name: {k: log10(p, k) - log10(true, k) for k in names} for name, p in sets.items() if name != "true"}
    lines = ["| weight | true | initial | " + " | ".join(f"fit {o}" for o in fitted) + " |",
             "|---|---|---|" + "---|" * len(fitted)]
    for k in names:
        cells = [f"{getattr(sets[s], k):.3g} ({err[s][k]:+.2f})" for s in ["initial"] + [f"fit_{o}" for o in fitted]]
        lines.append(f"| {k} | {getattr(true, k):.3g} | " + " | ".join(cells) + " |")
    lines += ["", "(in brackets: log10 error vs the true value, decades)", "",
              "| parameters | mean abs log10 error (weights) | open-loop RMS test (cm) | NLL / window test |",
              "|---|---|---|---|"]
    for s in sets:
        mae = np.mean([abs(err[s][k]) for k in weights]) if s in err else 0.0
        lines.append(f"| {s} | {mae:.3f} | {ol[s].mean():.3f} ± {ol[s].std():.3f} | {nll[s]:.2f} |")
    summary = "\n".join(lines)
    log("summary\n" + summary)
    (root / "summary.md").write_text(summary + "\n")
    (root / "results.json").write_text(json.dumps({
        "true": ck.params_to_dict(true), "initial": ck.params_to_dict(base),
        "fitted": {o: ck.params_to_dict(p) for o, p in fitted.items()}, "fit_info": infos,
        "log10_error": err, "open_loop_rms_cm": {s: v.tolist() for s, v in ol.items()}, "nll_per_window": nll,
        "n_train_windows": len(train), "n_test_windows": len(test),
        "config": OmegaConf.to_container(cfg, resolve=True)}, indent=2))
    log(f"saved to {root}")


if __name__ == "__main__":
    main()
