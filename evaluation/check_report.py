#!/usr/bin/env python3
"""Overview of the latest runs, for a quick check of the whole pipeline: output/check_<ts>/ with overview.png and an
index.html linking every run's own tables and figures.

Panels: eval.py (MPJPE and wrist ADE of the held-out subject vs observed fraction: IOC-fitted and initial weights
against the baselines), train.py (fitted / initial cost weights), goal_inference.py (goal accuracy over the movement,
prediction error per method on the held-out subject).

    python evaluation/check_report.py --eval-fit output/eval_A --train output/train_B --goal output/goal_inference_C
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# reference palette (dataviz skill): categorical slots in fixed order, grays for baselines, text in text tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAYS = ["#5f5e5a", "#8a8984", "#a9a8a2", "#c4c3bd"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
METHOD_LABELS = {"kin": "Kinematic model", "cart": "Cartesian multi-point", "minjerk": "Min-jerk",
                 "gcv": "Goal-directed CV", "cv": "Savitzky-Golay CV"}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2)


def read_eval(run):
    """{method: {metric: (ratios, means)}} from eval.py summary.csv."""
    out = {}
    with open(ROOT / run / "summary.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["method"], []).append(r)
    res = {}
    for m, rows in out.items():
        rows.sort(key=lambda r: float(r["obs_ratio"]))
        num = lambda v: float(v) if v not in ("", None) else np.nan
        res[m] = {k[:-5]: (np.array([float(r["obs_ratio"]) for r in rows]), np.array([num(r[k]) for r in rows]))
                  for k in rows[0] if k.endswith("_mean")}
    return res


def spread(values, gap):
    """Label positions near values (sorted), at least gap apart."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        pos[b] = max(pos[b], pos[a] + gap)
    return pos


def panel_eval(ax, hand, fit, metric, title):
    ends = []
    for i, m in enumerate(["cart", "minjerk", "gcv", "cv"]):  # baselines
        if m in hand:
            x, y = hand[m][metric]
            ax.plot(100 * x, y, color=GRAYS[i], linewidth=1.5, marker="o", markersize=4)
            ends.append((METHOD_LABELS[m], 100 * x[-1], y[-1]))
    if ends:
        lo, hi = ax.get_ylim()
        for (name, x_end, _), y_lab in zip(ends, spread([e[2] for e in ends], 0.045 * (hi - lo))):
            ax.annotate(name, (x_end, y_lab), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=TEXT2)
    for color, res, label in ((SERIES[1], hand, "Kinematic, initial weights"), (SERIES[0], fit, "Kinematic, IOC-fitted")):
        if res is not None and "kin" in res:
            x, y = res["kin"][metric]
            ax.plot(100 * x, y, color=color, linewidth=2.2, marker="o", markersize=6, label=label, zorder=3)
    style(ax, title, "observed fraction of the reach (%)", "cm (mean over the held-out reaches)")
    ax.set_xlim(5, 95)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def panel_train(ax, train):
    """IOC-fitted / initial cost weight (log scale)."""
    data = json.loads((ROOT / train / "params.json").read_text())
    init, fit = data.get("initial_params", {}), data["params"]
    names = [k for k in data.get("infer", []) if init.get(k, 0) > 0]
    if not names:
        ax.text(0.5, 0.5, "no fitted parameters", ha="center", transform=ax.transAxes, color=TEXT2)
        return
    y = np.arange(len(names))[::-1]
    r = [fit[k] / init[k] for k in names]
    ax.barh(y, np.log10(r), color=[SERIES[0] if v >= 1 else SERIES[1] for v in r], height=0.6, edgecolor=SURFACE)
    ax.axvline(0.0, color=TEXT2, linewidth=1)
    ax.set_yticks(y, names, fontsize=8)
    style(ax, f"train.py: IOC-fitted / initial weight ({len(data.get('train_subjects', []))} training subjects)",
          "log10 ratio", "")


def panel_goal_accuracy(ax, goal):
    res = json.loads((ROOT / goal / "results.json").read_text())
    x = np.arange(5, 100, 10)
    for color, name in ((GRAYS[1], "none"), (SERIES[0], "heading + gaze")):
        acc = res["cues"][name]["accuracy_bins"]
        ax.plot(x, acc, color=color, linewidth=2, marker="o", markersize=5, label=f"cue prior: {name}")
    style(ax, "goal_inference.py: correct goal during the reach (held-out subject)", "movement progress (%)",
          "most probable goal correct (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def panel_goal_errors(ax, goal):
    res = json.loads((ROOT / goal / "results.json").read_text())
    table = list(res["tables"].values())[0]["moving"]
    methods = ["goal inferred", "goal known", "oracle", "constant velocity", "frozen"]
    vals = [table[m][0] for m in methods]
    colors = [SERIES[0], SERIES[1], SERIES[2], GRAYS[1], GRAYS[2]]
    bars = ax.bar(range(len(methods)), vals, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(methods)), ["goal\ninferred", "goal\nknown", "goal and\nonset known", "constant\nvelocity",
                                        "frozen"], fontsize=8)
    style(ax, "goal_inference.py: MPJPE over 1 s while moving (held-out subject)", "", "cm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-fit", required=True, help="eval.py run (IOC-fitted weights, with kin_init)")
    ap.add_argument("--eval-hand", default=None, help="optional eval.py run of the initial weights only")
    ap.add_argument("--train", default=None)
    ap.add_argument("--goal", default=None, help="goal_inference.py run on the real sessions")
    args = ap.parse_args()
    out = ROOT / "output" / f"check_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)

    fit = read_eval(args.eval_fit)
    hand = read_eval(args.eval_hand) if args.eval_hand else {k: v for k, v in fit.items() if k != "kin"}
    if "kin_init" in hand:
        hand["kin"] = hand.pop("kin_init")
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), facecolor=SURFACE)
    panel_eval(axes[0, 0], hand, fit, "mpjpe_cm", "eval.py: MPJPE, 9 upper-body joints")
    panel_eval(axes[0, 1], hand, fit, "wrist_ade_cm", "eval.py: reaching-wrist ADE")
    if args.train:
        panel_train(axes[1, 0], args.train)
    if args.goal:
        panel_goal_accuracy(axes[1, 1], args.goal)
        panel_goal_errors(axes[2, 0], args.goal)
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    rel = lambda p: os.path.relpath(ROOT / p, out)
    links = [("eval.py (held-out subject), IOC-fitted and initial weights", f"{args.eval_fit}/summary.html")]
    for r in (10, 30, 50, 70):
        for f in sorted((ROOT / args.eval_fit / f"obs{r}" / "html").glob("skeleton_*.html")):
            links.append((f"&nbsp;&nbsp;{r}% observed: {f.stem}", f"{args.eval_fit}/obs{r}/html/{f.name}"))
        links.append((f"&nbsp;&nbsp;{r}% observed: frames / GIF", f"{args.eval_fit}/obs{r}/frames"))
    if args.train:
        links.append(("train.py run (params.json)", f"{args.train}/params.json"))
        for f in sorted((ROOT / args.train / "figures").glob("*.png")):
            links.append((f"&nbsp;&nbsp;{f.stem}", f"{args.train}/figures/{f.name}"))
    if args.eval_fit:
        for f in sorted((ROOT / args.eval_fit / "figures").glob("*.png")):
            links.append((f"&nbsp;&nbsp;eval: {f.stem}", f"{args.eval_fit}/figures/{f.name}"))
    if args.goal:
        links.append(("goal_inference.py (summary, goal-uncertainty figures)", f"{args.goal}/summary.html"))
    items = "".join(f"<li><a href='{rel(p)}'>{name}</a></li>" for name, p in links)
    extra = ""
    (out / "index.html").write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Pipeline check</title><style>body{font-family:"
        "sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}img{max-width:"
        "100%}a{color:#2a78d6}</style></head><body><h1>Pipeline check</h1>"
        f"<p>{datetime.now():%Y-%m-%d %H:%M}</p><ul>{items}</ul><img src='overview.png'>{extra}</body></html>")
    print(f"-> {out / 'index.html'}")


if __name__ == "__main__":
    main()
