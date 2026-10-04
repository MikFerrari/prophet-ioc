#!/usr/bin/env python3
"""Aggregate results of eval.py: per-trial CSV, summary CSV, JSON and an HTML table (best method per metric in bold).

    python save_results.py ../output/latest/results.json     # regenerate the tables of a saved evaluation
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# (key, label, lower is better: True / False, None = not ranked); errors in cm only (the relative errors *_pct
# of eval.py stay in results.json)
METRICS = [
    ("mpjpe_cm", "MPJPE (cm)", True),
    ("wrist_ade_cm", "Wrist ADE (cm)", True),
    ("wrist_fde_cm", "Wrist FDE (cm)", True),
    ("elbow_ade_cm", "Elbow ADE (cm)", True),
    ("elbow_fde_cm", "Elbow FDE (cm)", True),
    ("bone_distortion_pct", "Max bone distortion (%)", True),
    ("coverage_wrist_pct", "95% coverage wrist (%)", None),
    ("coverage_elbow_pct", "95% coverage elbow (%)", None),
    ("latency_ms", "Inference latency (ms)", True),
    ("rate_hz", "Inference rate (Hz)", False),
]
METHODS = [("kin", "Kinematic model, IOC-fitted"), ("kin_init", "Kinematic model, initial weights"),
           ("cart", "Cartesian Multi-Point"), ("minjerk", "Flash & Hogan Min-Jerk"),
           ("gcv", "Goal-Directed Const. Vel."), ("cv", "Savitzky-Golay Const. Vel."),
           ("promp", "ProMP (learned)"), ("dmp", "DMP (learned)")]   # promp, dmp: eval.baselines


def summarize(rows: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{method: {metric: {mean, std, median}}} over the trials."""
    out = {}
    for m, _ in METHODS:
        vals = [r for r in rows if r["method"] == m]
        out[m] = {}
        for key, _, _ in METRICS:
            v = np.array([r[key] for r in vals if r.get(key) is not None], dtype=float)
            if len(v):
                out[m][key] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v))}
    return out


def best_methods(summary: Dict) -> Dict[str, List[str]]:
    """Methods with the best (lowest) mean for every ranked metric."""
    best = {}
    for key, _, lower in METRICS:
        means = {m: s[key]["mean"] for m, s in summary.items() if key in s}
        if lower is None or len(means) < 2:
            continue
        b = min(means.values()) if lower else max(means.values())
        best[key] = [m for m, v in means.items() if round(v, 2) == round(b, 2)]  # ties at the displayed precision
    return best


def table_html(summary: Dict) -> str:
    """HTML table of one summary (methods x metrics, mean ± std, best method per metric in bold)."""
    best = best_methods(summary)
    head = "".join(f"<th>{label}</th>" for _, label, _ in METRICS)
    body = []
    for m, label in METHODS:
        if not summary.get(m):   # method not evaluated (e.g. eval.baselines: [])
            continue
        cells = []
        for key, _, _ in METRICS:
            s = summary[m].get(key)
            if s is None:
                cells.append('<td class="na">—</td>')
                continue
            txt = f"{s['mean']:.2f} <span class=\"std\">± {s['std']:.2f}</span>"
            cells.append(f"<td><b>{txt}</b></td>" if m in best.get(key, []) else f"<td>{txt}</td>")
        cls = ' class="ours"' if m in ("kin", "kin_init") else ""
        body.append(f"<tr{cls}><th class=\"model\">{label}</th>{''.join(cells)}</tr>")
    rows_html = "\n".join(body)
    return f"<table><thead><tr><th>Method</th>{head}</tr></thead><tbody>\n{rows_html}\n</tbody></table>"


_STYLE = """<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #F8FAFC; color: #0F172A; padding: 24px; }
  h1 { font-size: 20px; margin-bottom: 4px; } h2 { font-size: 16px; margin: 28px 0 0 0; }
  p { color: #475569; font-size: 13px; margin: 2px 0; }
  table { border-collapse: collapse; font-size: 13px; background: white; margin-top: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 9px 13px; border-bottom: 1px solid #E2E8F0; text-align: right; font-variant-numeric: tabular-nums; }
  thead th { background: #0F172A; color: white; font-weight: 600; font-size: 12px; text-align: center; }
  th.model { text-align: left; font-weight: 600; }
  tr.ours { background: #EFF6FF; }
  .std { color: #64748B; font-size: 11px; font-weight: normal; }
  td.na { color: #94A3B8; text-align: center; }
  b { color: #1D4ED8; }
</style>"""

NOTES = """<p>Mean ± std over the trials; best method per metric in <b>bold</b>. Only the reaching wrist has a goal (the
target); every method predicts all 9 joints. Coverage: fraction of the prediction steps whose ground truth lies in the 95 %
ellipsoid of the kinematic model (95 % when calibrated) or of the ProMP (its own predictive covariance, not
calibrated). ProMP and DMP are learned from the training subjects' reaches. Latency: one complete prediction from the observation window
(kinematic model: arrival time, Kalman-filtered handover, gILQR solve, keypoints and covariances), after compilation.</p>"""


def summary_html(summary: Dict, meta: Dict) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CARI v2 motion prediction</title>
{_STYLE}</head><body>
<h1>CARI v2 upper-body motion prediction: kinematic model vs baselines</h1>
<p>{meta['n_trials']} trials ({meta['velocity']}, subjects {', '.join(meta['subjects'])}, instructions {', '.join(map(str, meta['instructions']))}),
first {meta['obs_ratio']:.0%} of each reach observed, the rest predicted. Kinematic model parameters: {meta['params_source']}.</p>
{NOTES}
{table_html(summary)}
</body></html>
"""


def save_results(rows: List[Dict], meta: Dict, out_dir: Path) -> Dict[str, Path]:
    """Writes per_trial.csv, summary.csv, summary.html and results.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    paths = {k: out_dir / f for k, f in (("per_trial", "per_trial.csv"), ("summary", "summary.csv"),
                                         ("html", "summary.html"), ("json", "results.json"))}
    keys = ["subject", "instruction", "method"] + [k for k, _, _ in METRICS]
    with open(paths["per_trial"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(paths["summary"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{k}_{stat}" for k, _, _ in METRICS for stat in ("mean", "std", "median")])
        for m, _ in METHODS:
            w.writerow([m] + [summary[m].get(k, {}).get(stat, "") for k, _, _ in METRICS
                              for stat in ("mean", "std", "median")])
    paths["html"].write_text(summary_html(summary, meta))
    paths["json"].write_text(json.dumps({"meta": meta, "summary": summary, "best": best_methods(summary),
                                         "per_trial": rows}, indent=2))
    return paths


CONSOLE_METRICS = ["mpjpe_cm", "wrist_ade_cm", "wrist_fde_cm", "elbow_ade_cm",
                   "latency_ms", "rate_hz"]


def save_overview(overview: Dict[float, Dict], out_dir: Path) -> None:
    """summary.csv / summary.html over the observed fractions ({obs_ratio: summarize(rows)})."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_ratio", "method"] + [f"{k}_{stat}" for k, _, _ in METRICS for stat in ("mean", "std")])
        for r, summary in overview.items():
            for m, _ in METHODS:
                w.writerow([r, m] + [summary[m].get(k, {}).get(stat, "") for k, _, _ in METRICS
                                     for stat in ("mean", "std")])
    sections = "\n".join(f"<h2>{r:.0%} observed</h2>\n{table_html(summary)}" for r, summary in overview.items())
    (out_dir / "summary.html").write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CARI v2 motion prediction</title>
{_STYLE}</head><body>
<h1>CARI v2 upper-body motion prediction: kinematic model vs baselines</h1>
{NOTES}
{sections}
</body></html>
""")


def print_summary(summary: Dict, keys=CONSOLE_METRICS) -> None:
    """Console table of the main metrics (all of them are in summary.html / summary.csv)."""
    best = best_methods(summary)
    print(f"\n{'method':28s}" + "".join(f"{k:>21s}" for k in keys))
    for m, label in METHODS:
        if not summary.get(m):
            continue
        cells = []
        for k in keys:
            s = summary[m].get(k)
            txt = "—" if s is None else ("*" if m in best.get(k, []) else "") + f"{s['mean']:.2f} ± {s['std']:.2f}"
            cells.append(f"{txt:>21s}")
        print(f"{label:28s}" + "".join(cells))
    print("(* = best mean)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_results.py <results.json> [output_dir]")
        sys.exit(1)
    src = Path(sys.argv[1])
    data = json.loads(src.read_text())
    out = save_results(data["per_trial"], data["meta"], Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent)
    print_summary(summarize(data["per_trial"]))
    print("\n".join(f"  {p}" for p in out.values()))
