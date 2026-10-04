"""Figures of the IOC fit (train.py) and of the held-out evaluation (eval.py), as PNG files.

Colors follow one rule: the IOC-fitted model is blue (slot 1), the initial weights (the starting point of the fit)
orange (slot 2), the ground truth near-black, baselines and secondary marks in grays (the data-driven baselines ProMP
and DMP: green and purple, LEARNED); text in text tones only.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIT, INIT, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
GT = "#0b0b0b"
GRAYS = ["#5f5e5a", "#8a8984", "#a9a8a2", "#c4c3bd"]
LEARNED = {"promp": THIRD, "dmp": "#8a5cc2"}   # data-driven baselines (eval.baselines): own colors, thin lines
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
INSTRUCTIONS = {0: "0 hands home", 1: "1 object 1 (R)", 2: "2 home", 3: "3 object 2 (L)", 4: "4 home",
                5: "5 object 3", 6: "6 home", 7: "7 robot (both)", 8: "8 hands home"}


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ============================================================================= train.py
def plot_convergence(info: Dict, path: Path) -> Path:
    """Loss per segment of every restart over the projected-Adam iterations (best restart in blue)."""
    hist = np.array(info["loss_history_restarts"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    it = np.arange(1, len(hist) + 1)
    for r in range(hist.shape[1]):
        best = r == info["best_restart"]
        label = f"restart {r}" + (" (from the initial weights)" if r == 0 else "") + (", best" if best else "")
        ax.plot(it, hist[:, r], color=FIT if best else GRAYS[min(r, 3)], linewidth=2.2 if best else 1.3,
                label=label, zorder=3 if best else 2)
    ax.axhline(info["loss_base"], color=INIT, linewidth=1.2, linestyle="--", label="initial weights")
    if np.all(hist > 0) and info["loss_base"] > 0:
        ax.set_yscale("log")
    _style(ax, f"IOC fit: {info.get('objective', '')} loss per segment ({info['n_segments']} segments, {info['n_trials']} reaches)",
           "iteration", "loss (log)" if (np.all(hist > 0) and info["loss_base"] > 0) else "loss")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_parameters(base: Dict[str, float], fitted: Dict[str, float], info: Dict, bounds, path: Path) -> Path:
    """Fitted cost weights against the initial ones, within the search bounds (log scale); the end points of the
    other restarts show how well each weight is determined by the data."""
    names = list(info["infer"])
    lo, hi = bounds
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.6), facecolor=SURFACE)
    y = np.arange(len(names))[::-1]
    for yi, k in zip(y, names):
        ax.plot([getattr(lo, k), getattr(hi, k)], [yi, yi], color=GRID, linewidth=6, solid_capstyle="round", zorder=1)
        for r, v in enumerate(info["theta_restarts"][k]):
            if r != info["best_restart"]:
                ax.scatter(v, yi, s=22, color=GRAYS[2], zorder=2, label="other restarts" if (yi == y[0] and r in (0, 1)
                                                                                           and r != info["best_restart"]) else None)
        ax.scatter(base[k], yi, s=60, facecolors=SURFACE, edgecolors=INIT, linewidths=2, zorder=3,
                   label="initial" if yi == y[0] else None)
        ax.scatter(fitted[k], yi, s=60, color=FIT, zorder=4, label="IOC-fitted" if yi == y[0] else None)
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    _style(ax, "Cost weights: initial vs IOC-fitted (gray band: search bounds)", "value (log)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3)
    return _save(fig, path)


def plot_fit_quality(err_init: Dict, err_fit: Dict, path: Path, title_suffix: str = "") -> Path:
    """Open-loop error of the training segments with the initial and the fitted weights: by handover point and by
    instruction (mean, with the per-segment values as dots)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 2]})
    for ax, key, labels_fn, title in (
            (axes[0], "start", lambda v: f"{100 * v:.0f}%", "By handover point (fraction of the reach)"),
            (axes[1], "instruction", lambda v: INSTRUCTIONS.get(int(v), str(v)), "By instruction")):
        cats = sorted(set(err_init[key].tolist()))
        x = np.arange(len(cats))
        for j, (err, color, name) in enumerate(((err_init, INIT, "initial weights"), (err_fit, FIT, "IOC-fitted"))):
            means = [err["rms_cm"][err[key] == c].mean() for c in cats]
            ax.bar(x + (j - 0.5) * 0.38, means, width=0.38, color=color, label=name, edgecolor=SURFACE, linewidth=2)
            for xi, c in zip(x, cats):
                vals = err["rms_cm"][err[key] == c]
                ax.scatter(np.full(len(vals), xi + (j - 0.5) * 0.38) + np.random.default_rng(0).uniform(-0.08, 0.08,
                           len(vals)), vals, s=6, color=GRAYS[0], alpha=0.5, zorder=3)
        ax.set_xticks(x, [labels_fn(c) for c in cats], rotation=0 if key == "start" else 25, ha="center" if key ==
                      "start" else "right", fontsize=8)
        _style(ax, title, "", "RMS joint error (cm)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Open-loop error of the training reaches (9 joints, from the handover to the end of the reach)"
                 f"{title_suffix}", fontsize=11, color=TEXT, x=0.01, ha="left")
    return _save(fig, path)


def plot_example(gt: np.ndarray, t_obs: Sequence[int], preds_init: List[np.ndarray], preds_fit: List[np.ndarray],
                 title: str, path: Path) -> Path:
    """Wrist path of one reach (top and side views of the cell): ground truth, and the predictions from several
    handover points with the initial and the fitted weights."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    for ax, (i, j, xl, yl) in zip(axes, ((0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)"))):
        ax.plot(gt[:, i], gt[:, j], color=GT, linewidth=2.2, label="ground truth", zorder=4)
        for k, (f, pi, pf) in enumerate(zip(t_obs, preds_init, preds_fit)):
            ax.plot(pi[:, i], pi[:, j], color=INIT, linewidth=1.4, linestyle="--",
                    label="initial weights" if k == 0 else None)
            ax.plot(pf[:, i], pf[:, j], color=FIT, linewidth=1.8, label="IOC-fitted" if k == 0 else None)
            ax.scatter(gt[f, i], gt[f, j], s=40, color=SURFACE, edgecolors=GT, linewidths=1.5, zorder=5,
                       label="handover points" if k == 0 else None)
        ax.set_aspect("equal", adjustable="datalim")
        _style(ax, ("top view" if j == 1 else "side view") + f": {title}", xl, yl)
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, path)


# ============================================================================= eval.py
def plot_eval_overview(summary: Dict[float, Dict], labels: Dict[str, str], path: Path) -> Path:
    """MPJPE, wrist ADE and wrist 95 % coverage vs observed fraction, per method (fitted blue, initial orange,
    baselines gray)."""
    ratios = sorted(summary)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURFACE)
    color = {"kin": FIT, "kin_init": INIT}
    for ax, metric, title, ylab in ((axes[0], "mpjpe_cm", "MPJPE, 9 upper-body joints", "cm"),
                                    (axes[1], "wrist_ade_cm", "Reaching-wrist ADE", "cm"),
                                    (axes[2], "coverage_wrist_pct", "Wrist inside its 95 % region", "% of samples")):
        g = 0
        for m, label in labels.items():
            ys = [summary[r].get(m, {}).get(metric, {}).get("mean", np.nan) for r in ratios]
            if not np.any(np.isfinite(ys)):
                continue
            c = color.get(m) or LEARNED.get(m)
            if c is None:
                c, g = GRAYS[min(g, 3)], g + 1
            ax.plot([100 * r for r in ratios], ys, color=c, linewidth=2.4 if m in color else 1.3,
                    marker="o", markersize=6 if m in color else 4, zorder=3 if m in color else 2, label=label)
        if metric == "coverage_wrist_pct":
            ax.axhline(95, color=TEXT2, linewidth=1, linestyle=":")
        _style(ax, title, "observed fraction of the reach (%)", ylab)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_by_instruction(rows: List[Dict], labels: Dict[str, str], path: Path, metric: str = "mpjpe_cm") -> Path:
    """Error per instruction (mean over the evaluated reaches and observed fractions) for each method."""
    insts = sorted({r["instruction"] for r in rows})
    methods = [m for m in labels if any(r["method"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(14, 4.8), facecolor=SURFACE)
    w = 0.8 / len(methods)
    g = 0
    for j, m in enumerate(methods):
        c = {"kin": FIT, "kin_init": INIT}.get(m) or LEARNED.get(m)
        if c is None:
            c, g = GRAYS[min(g, 3)], g + 1
        vals = [np.mean([r[metric] for r in rows if r["method"] == m and r["instruction"] == i]) for i in insts]
        ax.bar(np.arange(len(insts)) + (j - (len(methods) - 1) / 2) * w, vals, width=w, color=c, label=labels[m],
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(np.arange(len(insts)), [INSTRUCTIONS.get(i, str(i)) for i in insts], rotation=20, ha="right")
    _style(ax, "Held-out subject: error per instruction (mean over the observed fractions)", "", "MPJPE (cm)")
    ax.legend(frameon=False, fontsize=8, ncol=len(methods))
    return _save(fig, path)


def plot_error_vs_time(curves: Dict[str, np.ndarray], labels: Dict[str, str], path: Path) -> Path:
    """Wrist error along the prediction (fraction of the remaining reach), mean over reaches and observed fractions."""
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    g = 0
    for m, label in labels.items():
        if m not in curves:
            continue
        c = {"kin": FIT, "kin_init": INIT}.get(m) or LEARNED.get(m)
        if c is None:
            c, g = GRAYS[min(g, 3)], g + 1
        y = curves[m]
        x = np.linspace(0, 100, len(y))
        ax.plot(x, y, color=c, linewidth=2.2 if m in ("kin", "kin_init") else 1.3, label=label)
    _style(ax, "Held-out subject: reaching-wrist error along the prediction", "prediction time (% of the remaining reach)",
           "error (cm)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)
