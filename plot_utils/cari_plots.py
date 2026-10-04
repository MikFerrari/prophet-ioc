"""3D figures of a CARI v2 prediction (evaluation/eval.py).

All functions take the prediction record of one trial built by eval.py:
    pred = {"subject", "velocity", "instruction", "task", "obs_ratio", "pred_dur", "H", "hand", "dt" (frame period),
            "target": (3,), "obs": {joint: (n_obs, 3)}, "gt": {joint: (H+1, 3)},
            "methods": {method: {joint: (H+1, 3)}}, "metrics": {method: {metric: value}},
            "cov": {"wrist": (H+1, 3, 3), "elbow": (H+1, 3, 3)} (kinematic model), "nom_bone_lens": {(a, b): length}}
Every 3D scene uses the same metric scale on x, y and z (ranges from the plotted data, equal aspect), so limbs keep
their real proportions.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_utils.skeleton import UPPER_BODY_BONES, add_bone_strain_skeleton, add_upper_body_skeleton
from plot_utils.surfaces import create_3d_tube_surface

METHOD_LABELS = {
    "kin": "Kinematic Model (19-DOF)",
    "cart": "Cartesian Multi-Point",
    "minjerk": "Flash & Hogan Min-Jerk",
    "gcv": "Goal-Directed Const. Vel.",
    "cv": "Savitzky-Golay Const. Vel.",
    "promp": "ProMP (learned)",
    "dmp": "DMP (learned)",
}
METHOD_STYLE = {   # baselines drawn when present in the record (promp, dmp: only with eval.baselines)
    "minjerk": dict(color="#8B5CF6", dash="dot"),
    "cart": dict(color="#06B6D4", dash="dash"),
    "cv": dict(color="#64748B", dash="dashdot"),
    "gcv": dict(color="#D97706", dash="longdash"),
    "promp": dict(color="#16A34A", dash="dash"),
    "dmp": dict(color="#DB2777", dash="dot"),
}
CHI2_3_95 = 7.815


def tube_radii(pred: Dict, part: str, method: str = "kin") -> np.ndarray:
    """Radius of the sphere with the mean variance of the 95 % ellipsoid of the model's `part` (wrist or
    elbow) position: sqrt(chi2_3(95%) * trace / 3), at least 3 mm."""
    cov_dict = None
    if "covs" in pred and method in pred["covs"]:
        cov_dict = pred["covs"][method]
    elif method == "kin" and "cov" in pred:
        cov_dict = pred["cov"]
    if cov_dict is None or part not in cov_dict or cov_dict[part] is None:
        return np.full(pred["H"] + 1, 0.003)
    return np.maximum(np.sqrt(CHI2_3_95 * np.trace(cov_dict[part], axis1=1, axis2=2) / 3.0), 0.003)


_CAMERA = dict(eye=dict(x=-1.85, y=-1.65, z=1.25), center=dict(x=0.0, y=0.0, z=-0.05), up=dict(x=0, y=0, z=1))
_TUBE_LIGHT = dict(ambient=0.95, diffuse=0.08, specular=0.0, roughness=1.0, fresnel=0.0)


def metric_scene(points: Iterable[np.ndarray], margin: float = 0.08) -> Dict:
    """Scene settings with axis ranges covering `points` (+ margin, in m) and equal scale on the three axes."""
    pts = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 3) for p in points], axis=0)
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    span = hi - lo
    ratio = span / span.max()
    axes = {f"{a}axis": dict(title=f"{a.upper()} (m)", range=[float(lo[i]), float(hi[i])], gridcolor="#E2E8F0")
            for i, a in enumerate("xyz")}
    return dict(**axes, aspectmode="manual", aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                camera=_CAMERA, bgcolor="white")


def _tube(path: np.ndarray, radii: np.ndarray, color: str, name: str) -> go.Surface:
    X, Y, Z = create_3d_tube_surface(path, radii, n_theta=32, n_fine=70)
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=np.zeros_like(X), colorscale=[[0, color], [1, color]], cmin=0,
                      cmax=1, showscale=False, opacity=0.12, name=name, lighting=_TUBE_LIGHT, hoverinfo="skip",
                      contours={a: dict(show=False, highlight=False) for a in "xyz"}, showlegend=True)


def _line(p: np.ndarray, name: str, color: str, width: float = 5, dash: str = None, markers: bool = False,
          showlegend: bool = True) -> go.Scatter3d:
    return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers" if markers else "lines",
                        line=dict(color=color, width=width, dash=dash), marker=dict(size=3.5, color=color),
                        name=name, showlegend=showlegend)


def _marker(p: np.ndarray, name: str, color: str, size: int = 9, showlegend: bool = True) -> go.Scatter3d:
    return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=name, showlegend=showlegend,
                        marker=dict(size=size, color=color, symbol="diamond"))


def _subtitle(pred: Dict) -> str:
    return (f"Subject: {pred['subject']} | {pred['velocity']} | Instruction {pred['instruction']} ({pred['task']}) | "
            f"Obs: {pred['obs_ratio']:.0%} | Horizon: {pred['pred_dur']:.2f} s ({pred['H']} steps)")


def plot_trial_skeleton(pred: Dict, path: Path, method: str = "kin") -> None:
    """Upper-body skeleton at t_obs and at the end of the prediction (kin or baseline), wrist trajectories (observed,
    predicted with its 95 % tube, ground truth) and target."""
    hand, model_pred = pred["hand"], pred["methods"][method]
    w = f"{hand}_wrist"
    fig = go.Figure()
    add_upper_body_skeleton(fig, {j: v[0] for j, v in model_pred.items()}, bone_color="#64748B", joint_color="#475569",
                            width=6, marker_size=6, name="Handover posture (t_obs)", opacity=0.7)
    add_upper_body_skeleton(fig, {j: v[-1] for j, v in model_pred.items()},
                            bone_color="#0284C7" if method == "kin" else "#16A34A",
                            joint_color="#0369A1" if method == "kin" else "#15803D",
                            width=6, marker_size=6,
                            name=f"Predicted final posture ({METHOD_LABELS.get(method, method)})", opacity=0.9)
    has_cov = (method == "kin" and "cov" in pred) or ("covs" in pred and method in pred["covs"])
    if has_cov:
        fig.add_trace(_tube(model_pred[w], tube_radii(pred, "wrist", method=method), "#EF4444",
                            f"95% confidence tube ({METHOD_LABELS.get(method, method)} wrist)"))
    fig.add_trace(_line(pred["obs"][w], f"Observed (0-{pred['obs_ratio']:.0%})", "#0284C7", markers=True))
    fig.add_trace(_line(model_pred[w], f"{METHOD_LABELS.get(method, method)} prediction", "#DC2626", width=6))
    fig.add_trace(_line(pred["gt"][w], "Ground truth", "#047857", width=6, markers=True))
    fig.add_trace(_marker(pred["obs"][w][-1], "Prediction start", "#F59E0B"))
    fig.add_trace(_marker(pred["target"], "Target", "#DC2626", size=10))
    m = pred["metrics"][method]
    label = METHOD_LABELS.get(method, method)
    fig.update_layout(
        title=dict(text=f"<b>CARI v2 upper-body motion prediction - {label}</b><br><sup>{_subtitle(pred)} | "
                        f"Wrist ADE {m['wrist_ade_cm']:.2f} cm | MPJPE {m['mpjpe_cm']:.2f} cm</sup>", font=dict(size=15)),
        scene=metric_scene([pred["obs"][j] for j in pred["obs"]] + [model_pred[j] for j in model_pred] + [pred["gt"][w], pred["target"]]),
        paper_bgcolor="white", width=1000, height=720, margin=dict(l=20, r=20, b=20, t=75),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.88)"))
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_trial_keypoint_trajectories(pred: Dict, path: Path, method: str = "kin") -> None:
    """As plot_trial_skeleton, without uncertainty tubes, with the observed, predicted and ground-truth trajectories of
    every upper-body keypoint."""
    model_pred = pred["methods"][method]
    fig = go.Figure()
    add_upper_body_skeleton(fig, {j: v[0] for j, v in model_pred.items()}, bone_color="#64748B", joint_color="#475569",
                            width=6, marker_size=6, name="Handover posture (t_obs)", opacity=0.7)
    add_upper_body_skeleton(fig, {j: v[-1] for j, v in model_pred.items()}, bone_color="#DC2626", joint_color="#991B1B",
                            width=5, marker_size=5, name=f"Predicted final posture ({METHOD_LABELS.get(method, method)})", opacity=0.8)
    add_upper_body_skeleton(fig, {j: v[-1] for j, v in pred["gt"].items()}, bone_color="#2563EB",
                            joint_color="#1E40AF", width=5, marker_size=5, name="Ground-truth final posture", opacity=0.8)
    label = METHOD_LABELS.get(method, method)
    for kind, series, color, width, markers in (("Observed", pred["obs"], "#16A34A", 4, False),
                                               (f"Predicted ({label})", model_pred, "#DC2626", 5, False),
                                               ("Ground truth", pred["gt"], "#2563EB", 4, True)):
        for i, (j, v) in enumerate(series.items()):
            tr = _line(v, kind, color, width=width, markers=markers, showlegend=i == 0)
            tr.update(legendgroup=kind, hovertext=j, hoverinfo="text+name")
            fig.add_trace(tr)
    fig.add_trace(_marker(pred["target"], "Target", "#111827", size=8))
    m = pred["metrics"][method]
    fig.update_layout(
        title=dict(text=f"<b>CARI v2 upper-body keypoint trajectories - {label}</b><br><sup>{_subtitle(pred)} | "
                        f"MPJPE {m['mpjpe_cm']:.2f} cm ({m['mpjpe_pct']:.0f}%)</sup>", font=dict(size=15)),
        scene=metric_scene([pred["obs"][j] for j in pred["obs"]] + [model_pred[j] for j in model_pred]
                           + [pred["gt"][j] for j in pred["gt"]] + [pred["target"]]),
        paper_bgcolor="white", width=1000, height=720, margin=dict(l=20, r=20, b=20, t=75),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.88)"))
    fig.write_html(str(path), include_plotlyjs="cdn")


def _tube_mesh(path: np.ndarray, radii: np.ndarray, n_theta: int = 32, per_step: int = 6):
    """Tube surface (X, Y, Z of shape (n, n_theta + 1)) of circles of the given radii perpendicular to the path, for
    matplotlib. Stationary parts of the path (repeated points: the prediction holds its final pose after the
    expected arrival) keep the last valid direction."""
    t = np.linspace(0.0, 1.0, len(path))
    tf = np.linspace(0.0, 1.0, max(per_step * (len(path) - 1), 2))
    P = np.stack([np.interp(tf, t, path[:, d]) for d in range(3)], axis=1)
    R = np.interp(tf, t, radii)
    T = np.gradient(P, axis=0)
    norm = np.linalg.norm(T, axis=1)
    last = np.array([1.0, 0.0, 0.0])
    for k in range(len(T)):
        if norm[k] > 1e-9:
            last = T[k] / norm[k]
        T[k] = last
    ref = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    N = np.cross(T, ref)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)
    th = np.linspace(0.0, 2 * np.pi, n_theta + 1)
    X = P[:, None, :] + R[:, None, None] * (np.cos(th)[None, :, None] * N[:, None, :]
                                            + np.sin(th)[None, :, None] * B[:, None, :])
    return X[..., 0], X[..., 1], X[..., 2]


def render_skeleton_frames(pred: Dict, out_dir: Path, method: str = "kin", tube: bool = False,
                           slowdown: float = 4.0, dpi: int = 200) -> Path:
    """One PDF per frame and an animated GIF: the upper-body skeleton at every observed frame (green), then at every
    prediction step the predicted (red) and ground-truth (blue) skeletons, with the reaching-wrist trails; with
    tube=True also the 95 % tube of the predicted wrist up to the current step. The GIF plays `slowdown` times
    slower than real time (observed frames every dt, prediction steps every horizon / H). Returns the GIF path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    obs, gt, model_pred, w = pred["obs"], pred["gt"], pred["methods"][method], f"{pred['hand']}_wrist"
    radii = tube_radii(pred, "wrist", method=method)
    pts = [np.concatenate(list(d.values())) for d in (obs, gt, model_pred)] + [pred["target"][None]]
    if tube:
        pts += [model_pred[w] + radii[:, None], model_pred[w] - radii[:, None]]
    pts = np.concatenate(pts)
    lo, hi = pts.min(axis=0) - 0.05, pts.max(axis=0) + 0.05
    n_obs, H = len(obs[w]), pred["H"]
    dt_pred = pred["pred_dur"] / H
    colors = {"obs": "#16A34A", "pred": "#DC2626", "gt": "#2563EB"}

    def draw(ax, pose, color, label):
        for a, b in UPPER_BODY_BONES:
            ax.plot(*np.stack([pose[a], pose[b]]).T, color=color, lw=3.5, solid_capstyle="round")
        ax.scatter(*np.stack(list(pose.values())).T, color=color, s=22, depthshade=False, label=label)

    label_m = METHOD_LABELS.get(method, method)
    frames, durations = [], []
    for k in range(n_obs + H):
        fig = plt.figure(figsize=(9, 7.5), dpi=dpi)
        ax = fig.add_subplot(projection="3d")
        if k < n_obs:
            draw(ax, {j: v[k] for j, v in obs.items()}, colors["obs"], "observed")
            ax.plot(*obs[w][:k + 1].T, color=colors["obs"], lw=1.8, alpha=0.7)
            t, phase, duration = k * pred["dt"], "observation", pred["dt"]
        else:
            i = k - n_obs + 1  # prediction step 1..H (step 0 is the last observed frame)
            ax.plot(*obs[w].T, color=colors["obs"], lw=1.8, alpha=0.5)
            if tube:
                X, Y, Z = _tube_mesh(model_pred[w][:i + 1], radii[:i + 1])
                ax.plot_surface(X, Y, Z, color=colors["pred"], alpha=0.15, linewidth=0, edgecolor="none",
                                shade=False)
            draw(ax, {j: v[i] for j, v in gt.items()}, colors["gt"], "ground truth")
            draw(ax, {j: v[i] for j, v in model_pred.items()}, colors["pred"], f"{label_m} prediction")
            ax.plot(*gt[w][:i + 1].T, color=colors["gt"], lw=1.8, alpha=0.7)
            ax.plot(*model_pred[w][:i + 1].T, color=colors["pred"], lw=1.8, alpha=0.7)
            t, phase, duration = (n_obs - 1) * pred["dt"] + i * dt_pred, "prediction", dt_pred
        ax.scatter(*pred["target"], color="#111827", marker="x", s=80, label="target")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=25, azim=-135)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(loc="upper left", fontsize=9)
        method_str = f" | {label_m}" if method != "kin" else ""
        title = f"{pred['subject']} inst{pred['instruction']} ({pred['velocity']}){method_str} | {pred['obs_ratio']:.0%} observed"
        ax.set_title(f"{title} | {phase} | t = {t:.2f} s" + (" | 95% wrist tube" if tube else ""), fontsize=11)
        path = out_dir / f"frame_{k:04d}.pdf"
        fig.savefig(path)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
        durations.append(max(int(round(duration * 1000 * slowdown)), 20))
        plt.close(fig)
    gif = out_dir / ("skeleton_tube.gif" if tube else "skeleton.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return gif


def plot_trial_arm_tubes(pred: Dict, path: Path) -> None:
    """Wrist and elbow trajectories of every method, with the 95 % tubes of the kinematic model."""
    hand = pred["hand"]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.03,
                        subplot_titles=[f"<b>{hand.capitalize()} wrist</b>", f"<b>{hand.capitalize()} elbow</b>"])
    pts = []
    for col, part, color in ((1, "wrist", "#DC2626"), (2, "elbow", "#EA580C")):
        j = f"{hand}_{part}"
        first = col == 1
        fig.add_trace(_tube(pred["methods"]["kin"][j], tube_radii(pred, part), color, f"95% confidence tube ({part})"),
                      row=1, col=col)
        fig.add_trace(_line(pred["obs"][j], f"Observed (0-{pred['obs_ratio']:.0%})", "#0284C7", markers=True,
                            showlegend=first), row=1, col=col)
        fig.add_trace(_line(pred["methods"]["kin"][j], METHOD_LABELS["kin"], color, showlegend=first), row=1, col=col)
        fig.add_trace(_line(pred["gt"][j], "Ground truth", "#047857", width=5.5, markers=True, showlegend=first),
                      row=1, col=col)
        for method, style in METHOD_STYLE.items():
            if method not in pred["methods"]:
                continue
            fig.add_trace(_line(pred["methods"][method][j], METHOD_LABELS[method], style["color"], width=3,
                                dash=style["dash"], showlegend=first), row=1, col=col)
        fig.add_trace(_marker(pred["obs"][j][-1], "Prediction start", "#F59E0B", 8, showlegend=False), row=1, col=col)
        pts += [pred["obs"][j], pred["gt"][j]] + [pred["methods"][m][j] for m in pred["methods"]]
    fig.add_trace(_marker(pred["target"], "Target", "#DC2626"), row=1, col=1)
    fig.update_scenes(**metric_scene(pts + [pred["target"]]))
    fig.update_layout(title=dict(text=f"<b>CARI v2 wrist and elbow predictions</b><br><sup>{_subtitle(pred)}</sup>",
                                 font=dict(size=15)),
                      paper_bgcolor="white", width=1280, height=680, margin=dict(l=20, r=20, b=20, t=75),
                      legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.88)"))
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_bone_strain_comparison(pred: Dict, path: Path) -> None:
    """Ground truth and every method side by side, bones colored by length strain, at mid prediction (default) or
    at the end (button)."""
    method_order = ("kin", "promp", "dmp", "minjerk", "cart", "gcv", "cv")
    active_methods = [m for m in method_order if m in pred["methods"]]
    panels = [("gt", "Ground truth (IK + FK)", pred["gt"])] + [
        (m, METHOD_LABELS.get(m, m), pred["methods"][m]) for m in active_methods]
    subtitles = []
    for key, _, _ in panels:
        d = 0.0 if key == "gt" else pred["metrics"][key]["bone_distortion_pct"]
        subtitles.append(f"max bone distortion {d:.1f}%")
    n_panels = len(panels)
    n_cols = 3 if n_panels <= 6 else 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=[[{"type": "scene"}] * n_cols] * n_rows,
                        horizontal_spacing=0.02, vertical_spacing=0.08,
                        subplot_titles=[f"<b>{t}</b><br><sup>{s}</sup>" for (_, t, _), s in zip(panels, subtitles)])
    w = f"{pred['hand']}_wrist"
    mid = len(pred["gt"][w]) // 2
    mid_idx, final_idx, pts = [], [], []
    for idx, (key, title, series) in enumerate(panels):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        start = len(fig.data)
        add_bone_strain_skeleton(fig, {j: v[mid] for j, v in series.items()}, pred["nom_bone_lens"], row=r, col=c,
                                 width=6, marker_size=5, visible=True)
        mid_idx += range(start, len(fig.data))
        start = len(fig.data)
        add_bone_strain_skeleton(fig, {j: v[-1] for j, v in series.items()}, pred["nom_bone_lens"], row=r, col=c,
                                 width=6, marker_size=5, visible=False)
        final_idx += range(start, len(fig.data))
        fig.add_trace(_line(series[w], f"Wrist ({title})", "#2563EB" if key in ("gt", "kin") else "#DC2626",
                            width=4, showlegend=False), row=r, col=c)
        fig.add_trace(_marker(pred["target"], "Target", "#DC2626", 7, showlegend=False), row=r, col=c)
        pts += [v for v in series.values()]
    scene = metric_scene(pts + [pred["target"]])
    fig.update_layout({("scene" if i == 1 else f"scene{i}"): scene for i in range(1, n_panels + 1)})
    n = len(fig.data)
    vis_mid = [i not in final_idx for i in range(n)]
    vis_final = [i not in mid_idx for i in range(n)]
    title = lambda when: (f"<b>Upper-body skeletons {when}</b><br><sup>{_subtitle(pred)} | green: rigid (≤2% strain), "
                          f"amber: 2-10%, crimson: >10%</sup>")
    fig.update_layout(
        title=dict(text=title("at mid prediction"), font=dict(size=15)),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center", y=1.08, buttons=[
            dict(label="Mid prediction", method="update", args=[{"visible": vis_mid}, {"title.text": title("at mid prediction")}]),
            dict(label="End of prediction", method="update", args=[{"visible": vis_final}, {"title.text": title("at the end of the prediction")}]),
        ])],
        paper_bgcolor="white", width=500 * n_cols, height=500 * n_rows, margin=dict(l=10, r=10, b=10, t=110), showlegend=False)
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_trial(pred: Dict, out_dir: Path) -> List[Path]:
    """All figures of one trial: out_dir/html/*.html and out_dir/frames/<trial>/{skeleton,skeleton_tube,...}/ (PDF
    frames + GIF). Returns their paths."""
    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{pred['subject']}_{pred['velocity']}_inst{pred['instruction']}"
    paths = [html_dir / f"skeleton_{tag}.html", html_dir / f"keypoint_trajectories_{tag}.html",
             html_dir / f"arm_tubes_{tag}.html", html_dir / f"bone_strain_{tag}.html"]
    plot_trial_skeleton(pred, paths[0], method="kin")
    plot_trial_keypoint_trajectories(pred, paths[1], method="kin")
    plot_trial_arm_tubes(pred, paths[2])
    plot_bone_strain_comparison(pred, paths[3])
    frames = out_dir / "frames" / tag
    paths.append(render_skeleton_frames(pred, frames / "skeleton", method="kin"))
    paths.append(render_skeleton_frames(pred, frames / "skeleton_tube", method="kin", tube=True))

    if "promp" in pred["methods"]:
        p_skel = html_dir / f"skeleton_{tag}_promp.html"
        p_traj = html_dir / f"keypoint_trajectories_{tag}_promp.html"
        plot_trial_skeleton(pred, p_skel, method="promp")
        plot_trial_keypoint_trajectories(pred, p_traj, method="promp")
        paths.extend([p_skel, p_traj])

        paths.append(render_skeleton_frames(pred, frames / "skeleton_promp", method="promp"))
        has_promp_cov = ("covs" in pred and "promp" in pred["covs"]) or ("cov" in pred and "promp" in pred.get("covs", {}))
        if has_promp_cov:
            paths.append(render_skeleton_frames(pred, frames / "skeleton_tube_promp", method="promp", tube=True))

    return paths
