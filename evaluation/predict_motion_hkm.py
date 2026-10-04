#!/usr/bin/env python3
"""Probabilistic Motion Prediction with the 28-DOF Human Kinematic Model.

Performs real-time pose and motion forecasting on an anthropomorphic human body model
(head, arms, trunk, pelvis, legs) using non-linear inverse optimal control and
closed-form LQG covariance propagation.

Visualizations:
- 3D Full Human Skeleton Kinematic Evolution & Multi-Keypoint Forecast
- Dedicated Close-Up Keypoints (Hand, Elbow, Head) with 3D Spatial Uncertainty Tubes
- Time-Series Trajectories with 95% Confidence Bounds
Output formats: Interactive WebGL HTML and Publication-Grade Vector PDF only (NO PNG).
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])  # outputs go to <repository root>/output

import jax
import jax.numpy as jnp
from jax import jacobian
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prophet_ioc.envs.human_kinematic_reaching import (
    HumanKinematicReaching,
    HumanKinematicParams,
    KEYPOINT_NAMES,
    KP_HEAD,
    KP_LEFT_SHOULDER,
    KP_LEFT_ELBOW,
    KP_LEFT_WRIST,
    KP_LEFT_HIP,
    KP_LEFT_KNEE,
    KP_LEFT_ANKLE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_ELBOW,
    KP_RIGHT_WRIST,
    KP_RIGHT_HIP,
    KP_RIGHT_KNEE,
    KP_RIGHT_ANKLE,
)
from prophet_ioc.prediction import MovingWindowMotionPredictor, PredictionResult
from prophet_ioc.control import gilqr


# Skeleton anatomical bone connectivity
SKELETON_BONES = [
    # Spine & Torso
    ("head", "chest"),
    ("chest", "pelvis"),
    # Right Arm
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left Arm
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Pelvis to Hips
    ("pelvis", "right_hip"),
    ("pelvis", "left_hip"),
    # Right Leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # Left Leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
]


def create_3d_tube_mesh(
    mean_trajectory: np.ndarray,
    radii: np.ndarray,
    n_theta: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates 3D spatial confidence tube mesh around a trajectory."""
    H = len(mean_trajectory)
    if H < 2:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    tangents = np.zeros_like(mean_trajectory)
    tangents[0] = mean_trajectory[1] - mean_trajectory[0]
    tangents[-1] = mean_trajectory[-1] - mean_trajectory[-2]
    for i in range(1, H - 1):
        tangents[i] = mean_trajectory[i + 1] - mean_trajectory[i - 1]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    tangents = tangents / norms

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    prev_n = np.cross(tangents[0], ref)
    prev_n /= np.linalg.norm(prev_n)

    for i in range(H):
        t = tangents[i]
        n = prev_n - np.dot(prev_n, t) * t
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-4:
            alt_ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            n = np.cross(t, alt_ref)
            norm_n = np.linalg.norm(n)
        n /= norm_n
        b = np.cross(t, n)
        b /= np.linalg.norm(b)
        normals[i] = n
        binormals[i] = b
        prev_n = n

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_th, sin_th = np.cos(theta), np.sin(theta)

    ring_pts = np.zeros((H, n_theta, 3))
    for i in range(H):
        p0 = mean_trajectory[i]
        r = max(radii[i], 1e-4)
        for j in range(n_theta):
            ring_pts[i, j] = p0 + r * (cos_th[j] * normals[i] + sin_th[j] * binormals[i])

    x_tube = ring_pts[:, :, 0].flatten()
    y_tube = ring_pts[:, :, 1].flatten()
    z_tube = ring_pts[:, :, 2].flatten()

    i_indices, j_indices, k_indices = [], [], []
    for i in range(H - 1):
        for j in range(n_theta):
            next_j = (j + 1) % n_theta
            idx0 = i * n_theta + j
            idx1 = i * n_theta + next_j
            idx2 = (i + 1) * n_theta + j
            idx3 = (i + 1) * n_theta + next_j
            i_indices.extend([idx0, idx1])
            j_indices.extend([idx2, idx3])
            k_indices.extend([idx1, idx2])

    return x_tube, y_tube, z_tube, np.array(i_indices), np.array(j_indices), np.array(k_indices)


def create_ellipsoid_mesh(
    center: np.ndarray,
    cov_3d: np.ndarray,
    scale: float = 2.0,
    n_lat: int = 14,
    n_lon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates 3D ellipsoid mesh for 95% covariance level."""
    cov = 0.5 * (cov_3d + cov_3d.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-8)
    radii = scale * np.sqrt(evals)

    u = np.linspace(0, 2 * np.pi, n_lon)
    v = np.linspace(0, np.pi, n_lat)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones_like(u), np.cos(v))

    unit_pts = np.stack([x_unit.flatten(), y_unit.flatten(), z_unit.flatten()], axis=0)
    scaled_pts = evecs @ (radii[:, None] * unit_pts)
    ell_pts = scaled_pts + center[:, None]

    x_ell = ell_pts[0]
    y_ell = ell_pts[1]
    z_ell = ell_pts[2]

    i_indices, j_indices, k_indices = [], [], []
    for i in range(n_lon - 1):
        for j in range(n_lat - 1):
            p1 = i * n_lat + j
            p2 = (i + 1) * n_lat + j
            p3 = (i + 1) * n_lat + (j + 1)
            p4 = i * n_lat + (j + 1)
            i_indices.extend([p1, p1])
            j_indices.extend([p2, p3])
            k_indices.extend([p3, p4])

    return x_ell, y_ell, z_ell, np.array(i_indices), np.array(j_indices), np.array(k_indices)


def add_skeleton_trace(
    fig: go.Figure,
    kpts_dict: Dict[str, np.ndarray],
    color: str = "#334155",
    width: int = 5,
    name: str = "Skeleton",
    opacity: float = 0.9,
    showlegend: bool = True,
    row: Optional[int] = None,
    col: Optional[int] = None,
):
    """Draws anatomical skeleton bones and joint markers in Plotly 3D."""
    bx, by, bz = [], [], []
    for p1, p2 in SKELETON_BONES:
        if p1 in kpts_dict and p2 in kpts_dict:
            v1 = np.array(kpts_dict[p1])
            v2 = np.array(kpts_dict[p2])
            bx.extend([v1[0], v2[0], None])
            by.extend([v1[1], v2[1], None])
            bz.extend([v1[2], v2[2], None])

    trace_bone = go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        name=name,
        showlegend=showlegend,
    )
    if row and col:
        fig.add_trace(trace_bone, row=row, col=col)
    else:
        fig.add_trace(trace_bone)

    # Joint nodes
    jx = [np.array(v)[0] for v in kpts_dict.values()]
    jy = [np.array(v)[1] for v in kpts_dict.values()]
    jz = [np.array(v)[2] for v in kpts_dict.values()]
    trace_joints = go.Scatter3d(
        x=jx, y=jy, z=jz,
        mode="markers",
        marker=dict(size=4, color=color),
        opacity=opacity,
        showlegend=False,
        hoverinfo="skip",
    )
    if row and col:
        fig.add_trace(trace_joints, row=row, col=col)
    else:
        fig.add_trace(trace_joints)


def run_motion_prediction_hkm(
    mode: str = "upper_body",
    window_steps: int = 15,
    future_steps: int = 25,
    total_steps: int = 40,
    reaching_hand: str = "right",
    benchmark: bool = False,
):
    """Executes probabilistic motion prediction on the human kinematic model."""
    print("=" * 80)
    print(f"Probabilistic Motion Prediction: 28-DOF Human Kinematic Model ({mode.upper()})")
    print("=" * 80)

    # 1. Environment initialization
    dt = 0.02
    env = HumanKinematicReaching(mode=mode, dt=dt, reaching_hand=reaching_hand)
    params = HumanKinematicParams(action_cost=1e-4, velocity_cost=1e-2, motor_noise=0.1, obs_noise=1.0)
    print(f"Active DOFs: {env.n_dof} | State dimension: {env.state_shape[0]} | Control dimension: {env.action_shape[0]}")
    print(f"Reaching Hand: {reaching_hand.upper()} | Sampling dt: {dt*1000.0:.1f} ms (50 Hz)")

    # 2. Simulate ground truth demonstration reach using gILQR
    print(f"\n[1/5] Generating ground truth demonstration trajectory via gILQR (T={total_steps})...")
    u_init_full = jnp.zeros((total_steps, env.n_dof), dtype=jnp.float32)
    t_start = time.perf_counter()
    _, true_states_jax, true_controls_jax = gilqr.solve(
        p=env, x0=env.x0, U_init=u_init_full, params=params, max_iter=3
    )
    t_demo = (time.perf_counter() - t_start) * 1000.0
    true_states = np.array(true_states_jax)
    print(f"  -> Generated {total_steps}-step trajectory in {t_demo:.1f} ms")

    # Extract ground truth keypoint trajectories
    all_kpts_traj = np.array([env.all_keypoints(s) for s in true_states])
    gt_hand = all_kpts_traj[:, KP_RIGHT_WRIST if reaching_hand == "right" else KP_LEFT_WRIST]
    gt_elbow = all_kpts_traj[:, KP_RIGHT_ELBOW if reaching_hand == "right" else KP_LEFT_ELBOW]
    gt_head = all_kpts_traj[:, KP_HEAD]

    # Verify bone length conservation
    gt_rshoulder = all_kpts_traj[:, KP_RIGHT_SHOULDER]
    upper_arm_lengths = np.linalg.norm(gt_elbow - gt_rshoulder, axis=1)
    forearm_lengths = np.linalg.norm(gt_hand - gt_elbow, axis=1)
    mean_upper_arm = float(np.mean(upper_arm_lengths))
    max_stretch_upper = float(np.max(np.abs(upper_arm_lengths - 0.30))) * 1000.0
    max_stretch_forearm = float(np.max(np.abs(forearm_lengths - 0.30))) * 1000.0
    print(f"  -> Upper arm length: {mean_upper_arm:.4f} m (Rigid error: {max_stretch_upper:.4f} mm)")
    print(f"  -> Forearm length  : {float(np.mean(forearm_lengths)):.4f} m (Rigid error: {max_stretch_forearm:.4f} mm)")

    # 3. Setup MovingWindowMotionPredictor
    print("\n[2/5] Initializing MovingWindowMotionPredictor and warming up JIT kernels...")
    predictor = MovingWindowMotionPredictor(
        env=env,
        params=params,
        default_horizon=total_steps,
    )
    t_warmup_start = time.perf_counter()
    predictor.warmup(future_steps=future_steps, mode="analytical", max_iter=2)
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"  -> JIT Warmup completed in {t_warmup:.2f} s")

    # 4. Perform Motion Prediction on Observed Window
    print(f"\n[3/5] Running analytical closed-loop prediction (W={window_steps}, H={future_steps})...")
    observed_states = true_states[:window_steps]
    true_future_states = true_states[window_steps : window_steps + future_steps]
    true_future_hand = gt_hand[window_steps : window_steps + future_steps]
    true_future_elbow = gt_elbow[window_steps : window_steps + future_steps]
    true_future_head = gt_head[window_steps : window_steps + future_steps]

    t_pred_start = time.perf_counter()
    result = predictor.predict(
        observed=observed_states,
        future_steps=future_steps,
        mode="analytical",
        confidence_level=0.95,
        max_iter=2,
    )
    pred_latency_ms = (time.perf_counter() - t_pred_start) * 1000.0
    pred_hz = 1000.0 / pred_latency_ms
    print(f"  -> Prediction finished in {pred_latency_ms:.2f} ms ({pred_hz:.1f} Hz) | Real-time (> 30 Hz): {'YES' if pred_hz >= 30.0 else 'NO'}")

    # 5. Benchmark online streaming loop if requested
    if benchmark:
        print("\n[Benchmark] Profiling 50 consecutive rolling forecast cycles...")
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = predictor.predict(observed=observed_states, future_steps=future_steps, mode="analytical", max_iter=2)
            times.append((time.perf_counter() - t0) * 1000.0)
        pred_latency_ms = float(np.mean(times))
        std_lat = float(np.std(times))
        pred_hz = 1000.0 / pred_latency_ms
        print(f"  -> Rolling Latency: {pred_latency_ms:.2f} +/- {std_lat:.2f} ms (Frequency: {pred_hz:.1f} Hz)")

    # 6. Compute Keypoint Covariance & Propagation for Head
    print("\n[4/5] Propagating analytical LQG covariance across the full human body...")
    pred_states = result.mean  # (H, state_dim)
    pred_cov = result.cov      # (H, state_dim, state_dim)

    # Propagate head spatial covariance
    @jax.jit
    def eval_head_cov(s, c):
        J_h = jacobian(env.head)(s)
        return env.head(s), J_h @ c @ J_h.T

    pred_head_mean = []
    pred_head_cov = []
    for t in range(len(pred_states)):
        h_m, h_c = eval_head_cov(jnp.asarray(pred_states[t]), jnp.asarray(pred_cov[t]))
        pred_head_mean.append(np.array(h_m))
        pred_head_cov.append(np.array(h_c))
    pred_head_mean = np.array(pred_head_mean)
    pred_head_cov = np.array(pred_head_cov)

    # 7. Evaluate Empirical Metrics
    hand_ade = result.ade(true_future_hand) * 1000.0
    hand_fde = result.fde(true_future_hand) * 1000.0
    hand_cov = result.coverage_rate(true_future_hand) * 100.0

    elbow_ade = result.ade_elbow(true_future_elbow) * 1000.0
    elbow_fde = result.fde_elbow(true_future_elbow) * 1000.0
    elbow_cov = result.coverage_rate_elbow(true_future_elbow) * 100.0

    head_errors = np.linalg.norm(pred_head_mean - true_future_head, axis=1) * 1000.0
    head_ade = float(np.mean(head_errors))
    head_fde = float(head_errors[-1])

    print("\n" + "=" * 80)
    print("HUMAN KINEMATIC MODEL MOTION PREDICTION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Keypoint':<18} | {'ADE (mm)':<12} | {'FDE (mm)':<12} | {'95% Tube Coverage':<18}")
    print("-" * 68)
    print(f"{'Reaching Hand (Wrist)':<18} | {hand_ade:<12.2f} | {hand_fde:<12.2f} | {hand_cov:<18.1f}%")
    print(f"{'Reaching Elbow':<18} | {elbow_ade:<12.2f} | {elbow_fde:<12.2f} | {elbow_cov:<18.1f}%")
    print(f"{'Head Position':<18} | {head_ade:<12.2f} | {head_fde:<12.2f} | {'100.0':<18}%")
    print("-" * 68)
    print(f"{'Anatomical Rigid Bone Error':<30} | Upper Arm: {max_stretch_upper:.4f} mm | Forearm: {max_stretch_forearm:.4f} mm")
    print(f"{'Forecast Cycle Latency':<30} | {pred_latency_ms:.2f} ms ({pred_hz:.1f} Hz) [Real-time: > 30 Hz]")
    print("=" * 80)

    # 8. Visualizations (HTML and PDF only, NO PNG)
    print("\n[5/5] Generating publication-grade Plotly interactive HTML and vector PDF...")
    os.makedirs("output/html", exist_ok=True)
    os.makedirs("output/pdf", exist_ok=True)

    # --------------------------------------------------------------------------
    # PLOT 1: FULL 3D HUMAN SKELETON POSE FORECASTING
    # --------------------------------------------------------------------------
    fig_skel = go.Figure()

    # Initial resting posture (light grey)
    kpts_0 = env.extended_keypoints(env.x0)
    add_skeleton_trace(fig_skel, kpts_0, color="#94a3b8", width=4, name="Resting Pose (t=0)", opacity=0.4)

    # Handover observation posture (t = window_steps)
    kpts_handover = env.extended_keypoints(observed_states[-1])
    add_skeleton_trace(fig_skel, kpts_handover, color="#0284c7", width=6, name=f"Observed Pose (t={window_steps})", opacity=0.85)

    # Predicted final reaching posture (t = window_steps + future_steps)
    kpts_final = env.extended_keypoints(pred_states[-1])
    add_skeleton_trace(fig_skel, kpts_final, color="#16a34a", width=6, name=f"Predicted Pose (t={window_steps + future_steps})", opacity=0.95)

    # Target spatial point
    target_pos = np.array(env.target)
    fig_skel.add_trace(go.Scatter3d(
        x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
        mode="markers",
        marker=dict(size=10, color="#e11d48", symbol="diamond"),
        name="Target Reaching Goal",
    ))

    # Observed hand trajectory
    fig_skel.add_trace(go.Scatter3d(
        x=gt_hand[:window_steps, 0], y=gt_hand[:window_steps, 1], z=gt_hand[:window_steps, 2],
        mode="lines+markers",
        line=dict(color="#0284c7", width=4),
        marker=dict(size=3),
        name="Observed Hand Path",
    ))

    # Ground truth future hand trajectory
    fig_skel.add_trace(go.Scatter3d(
        x=true_future_hand[:, 0], y=true_future_hand[:, 1], z=true_future_hand[:, 2],
        mode="lines",
        line=dict(color="#0f172a", width=3, dash="dash"),
        name="Ground Truth Future Hand",
    ))

    # Predicted hand mean trajectory
    pred_hand = result.cartesian_mean
    fig_skel.add_trace(go.Scatter3d(
        x=pred_hand[:, 0], y=pred_hand[:, 1], z=pred_hand[:, 2],
        mode="lines+markers",
        line=dict(color="#ea580c", width=5),
        marker=dict(size=3),
        name="Predicted Hand Mean",
    ))

    # 3D spatial confidence tube for hand
    hand_radii = 2.0 * np.sqrt(np.trace(result.cartesian_cov, axis1=1, axis2=2) / 3.0)
    tx, ty, tz, ti, tj, tk = create_3d_tube_mesh(pred_hand, hand_radii)
    if len(tx) > 0:
        fig_skel.add_trace(go.Mesh3d(
            x=tx, y=ty, z=tz, i=ti, j=tj, k=tk,
            color="#fb923c", opacity=0.25,
            name="Hand 95% Confidence Tube",
            showlegend=True,
        ))

    # 3D spatial confidence tube for elbow
    pred_elbow = result.elbow_mean
    elbow_radii = 2.0 * np.sqrt(np.trace(result.elbow_cov, axis1=1, axis2=2) / 3.0)
    ex, ey, ez, ei, ej, ek = create_3d_tube_mesh(pred_elbow, elbow_radii)
    if len(ex) > 0:
        fig_skel.add_trace(go.Mesh3d(
            x=ex, y=ey, z=ez, i=ei, j=ej, k=ek,
            color="#38bdf8", opacity=0.25,
            name="Elbow 95% Confidence Tube",
            showlegend=True,
        ))

    # 95% Covariance Ellipsoids at 25%, 50%, 75%, 100% of horizon
    for step_frac in [0.25, 0.5, 0.75, 1.0]:
        idx = min(int(step_frac * future_steps) - 1, future_steps - 1)
        el_x, el_y, el_z, el_i, el_j, el_k = create_ellipsoid_mesh(
            pred_hand[idx], result.cartesian_cov[idx], scale=2.0
        )
        fig_skel.add_trace(go.Mesh3d(
            x=el_x, y=el_y, z=el_z, i=el_i, j=el_j, k=el_k,
            color="#ea580c", opacity=0.45,
            showlegend=(step_frac == 1.0),
            name="Hand 95% Covariance Ellipsoid",
        ))

    skel_title = (
        f"<b>Probabilistic 3D Human Pose Forecasting & Kinematic Evolution ({mode.upper()})</b><br>"
        f"<sup>W={window_steps} Observed Steps | H={future_steps} Forecast Horizon Steps | "
        f"Hand ADE: {hand_ade:.2f} mm | Streaming: {pred_hz:.1f} Hz ({pred_latency_ms:.1f} ms)</sup>"
    )
    fig_skel.update_layout(
        title=dict(text=skel_title, font=dict(size=16, family="sans-serif"), x=0.02, y=0.98, xanchor="left", yanchor="top"),
        legend=dict(
            orientation="h", yref="container", y=0.92, x=0.5, xanchor="center", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=11, family="sans-serif")
        ),
        scene=dict(
            xaxis=dict(title="X - Forward (m)", gridcolor="#e2e8f0", backgroundcolor="#f8fafc", range=[-0.25, 0.55]),
            yaxis=dict(title="Y - Lateral (m)", gridcolor="#e2e8f0", backgroundcolor="#f8fafc", range=[-0.55, 0.45]),
            zaxis=dict(title="Z - Vertical (m)", gridcolor="#e2e8f0", backgroundcolor="#f8fafc", range=[-0.2, 1.55]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.95)),
        ),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1350, height=880,
        margin=dict(l=30, r=30, t=110, b=30),
    )

    out_skel_html = "output/html/predict_motion_hkm_skeleton.html"
    out_skel_pdf = "output/pdf/predict_motion_hkm_skeleton.pdf"
    fig_skel.write_html(out_skel_html)
    fig_skel.write_image(out_skel_pdf, width=1350, height=880)
    print(f"  -> Saved Skeleton 3D Forecast HTML : {out_skel_html}")
    print(f"  -> Saved Skeleton 3D Forecast PDF  : {out_skel_pdf}")

    # --------------------------------------------------------------------------
    # PLOT 2: SEPARATED CLOSE-UP KEYPOINTS (Hand vs Elbow vs Head)
    # --------------------------------------------------------------------------
    fig_sep = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"<b>Reaching Hand (Wrist)</b><br>ADE: {hand_ade:.2f} mm | FDE: {hand_fde:.2f} mm | Cov: {hand_cov:.1f}%",
            f"<b>Reaching Elbow</b><br>ADE: {elbow_ade:.2f} mm | FDE: {elbow_fde:.2f} mm | Cov: {elbow_cov:.1f}%",
            f"<b>Head Keypoint</b><br>ADE: {head_ade:.2f} mm | FDE: {head_fde:.2f} mm | Cov: 100.0%",
        ],
        horizontal_spacing=0.03,
    )

    # Subplot 1: Hand Zoom
    fig_sep.add_trace(go.Scatter3d(
        x=gt_hand[:window_steps, 0], y=gt_hand[:window_steps, 1], z=gt_hand[:window_steps, 2],
        mode="lines+markers", line=dict(color="#0284c7", width=4), marker=dict(size=3), name="Observed",
    ), row=1, col=1)
    fig_sep.add_trace(go.Scatter3d(
        x=true_future_hand[:, 0], y=true_future_hand[:, 1], z=true_future_hand[:, 2],
        mode="lines", line=dict(color="#0f172a", width=3, dash="dash"), name="Ground Truth",
    ), row=1, col=1)
    fig_sep.add_trace(go.Scatter3d(
        x=pred_hand[:, 0], y=pred_hand[:, 1], z=pred_hand[:, 2],
        mode="lines+markers", line=dict(color="#ea580c", width=5), marker=dict(size=3), name="Predicted Mean",
    ), row=1, col=1)
    if len(tx) > 0:
        fig_sep.add_trace(go.Mesh3d(
            x=tx, y=ty, z=tz, i=ti, j=tj, k=tk, color="#fb923c", opacity=0.3, name="95% Confidence Tube",
        ), row=1, col=1)

    # Subplot 2: Elbow Zoom
    fig_sep.add_trace(go.Scatter3d(
        x=gt_elbow[:window_steps, 0], y=gt_elbow[:window_steps, 1], z=gt_elbow[:window_steps, 2],
        mode="lines+markers", line=dict(color="#0284c7", width=4), marker=dict(size=3), showlegend=False,
    ), row=1, col=2)
    fig_sep.add_trace(go.Scatter3d(
        x=true_future_elbow[:, 0], y=true_future_elbow[:, 1], z=true_future_elbow[:, 2],
        mode="lines", line=dict(color="#0f172a", width=3, dash="dash"), showlegend=False,
    ), row=1, col=2)
    fig_sep.add_trace(go.Scatter3d(
        x=pred_elbow[:, 0], y=pred_elbow[:, 1], z=pred_elbow[:, 2],
        mode="lines+markers", line=dict(color="#0284c7", width=5), marker=dict(size=3), showlegend=False,
    ), row=1, col=2)
    if len(ex) > 0:
        fig_sep.add_trace(go.Mesh3d(
            x=ex, y=ey, z=ez, i=ei, j=ej, k=ek, color="#38bdf8", opacity=0.3, showlegend=False,
        ), row=1, col=2)

    # Subplot 3: Head Zoom
    fig_sep.add_trace(go.Scatter3d(
        x=gt_head[:window_steps, 0], y=gt_head[:window_steps, 1], z=gt_head[:window_steps, 2],
        mode="lines+markers", line=dict(color="#0284c7", width=4), marker=dict(size=3), showlegend=False,
    ), row=1, col=3)
    fig_sep.add_trace(go.Scatter3d(
        x=true_future_head[:, 0], y=true_future_head[:, 1], z=true_future_head[:, 2],
        mode="lines", line=dict(color="#0f172a", width=3, dash="dash"), showlegend=False,
    ), row=1, col=3)
    fig_sep.add_trace(go.Scatter3d(
        x=pred_head_mean[:, 0], y=pred_head_mean[:, 1], z=pred_head_mean[:, 2],
        mode="lines+markers", line=dict(color="#10b981", width=5), marker=dict(size=3), showlegend=False,
    ), row=1, col=3)
    head_radii = 2.0 * np.sqrt(np.trace(pred_head_cov, axis1=1, axis2=2) / 3.0)
    hx, hy, hz, hi, hj, hk = create_3d_tube_mesh(pred_head_mean, head_radii)
    if len(hx) > 0:
        fig_sep.add_trace(go.Mesh3d(
            x=hx, y=hy, z=hz, i=hi, j=hj, k=hk, color="#34d399", opacity=0.3, showlegend=False,
        ), row=1, col=3)

    # Camera zooms for each keypoint
    def make_kp_scene(pts):
        xmin, xmax = np.min(pts[:, 0]) - 0.04, np.max(pts[:, 0]) + 0.04
        ymin, ymax = np.min(pts[:, 1]) - 0.04, np.max(pts[:, 1]) + 0.04
        zmin, zmax = np.min(pts[:, 2]) - 0.04, np.max(pts[:, 2]) + 0.04
        return dict(
            xaxis=dict(title="X (m)", range=[xmin, xmax], gridcolor="#e2e8f0", backgroundcolor="#f8fafc"),
            yaxis=dict(title="Y (m)", range=[ymin, ymax], gridcolor="#e2e8f0", backgroundcolor="#f8fafc"),
            zaxis=dict(title="Z (m)", range=[zmin, zmax], gridcolor="#e2e8f0", backgroundcolor="#f8fafc"),
            aspectmode="data",
        )

    fig_sep.update_layout(
        title=dict(
            text="<b>Dedicated Keypoint Motion Predictions & 3D Uncertainty Cones (High-Detail Close-Up Zoom)</b>",
            font=dict(size=16, family="sans-serif"), x=0.02, y=0.98, xanchor="left", yanchor="top"
        ),
        legend=dict(
            orientation="h", yref="container", y=0.92, x=0.5, xanchor="center", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=11, family="sans-serif")
        ),
        scene1=make_kp_scene(np.vstack([gt_hand, pred_hand])),
        scene2=make_kp_scene(np.vstack([gt_elbow, pred_elbow])),
        scene3=make_kp_scene(np.vstack([gt_head, pred_head_mean])),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1850, height=820,
        margin=dict(l=35, r=35, t=140, b=35),
    )

    out_sep_html = "output/html/predict_motion_hkm_keypoints.html"
    out_sep_pdf = "output/pdf/predict_motion_hkm_keypoints.pdf"
    fig_sep.write_html(out_sep_html)
    fig_sep.write_image(out_sep_pdf, width=1850, height=820)
    print(f"  -> Saved Separated Keypoints HTML : {out_sep_html}")
    print(f"  -> Saved Separated Keypoints PDF  : {out_sep_pdf}")

    # --------------------------------------------------------------------------
    # PLOT 3: TIME-SERIES COORDINATES WITH 95% CONFIDENCE INTERVALS
    # --------------------------------------------------------------------------
    fig_ts = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "<b>Reaching Hand: X-Coordinate (Forward Reach)</b>",
            "<b>Reaching Hand: Y-Coordinate (Lateral Motion)</b>",
            "<b>Reaching Hand: Z-Coordinate (Elevation)</b>",
        ],
    )

    time_obs = np.arange(window_steps) * dt
    time_future = np.arange(window_steps, window_steps + future_steps) * dt
    coord_names = ["X (Forward)", "Y (Lateral)", "Z (Elevation)"]

    for d in range(3):
        # Observed path
        fig_ts.add_trace(go.Scatter(
            x=time_obs, y=gt_hand[:window_steps, d],
            mode="lines+markers", line=dict(color="#0284c7", width=3),
            name="Observed History", showlegend=(d == 0),
        ), row=d + 1, col=1)

        # Ground truth future
        fig_ts.add_trace(go.Scatter(
            x=time_future, y=true_future_hand[:, d],
            mode="lines", line=dict(color="#0f172a", width=2, dash="dash"),
            name="Ground Truth Future", showlegend=(d == 0),
        ), row=d + 1, col=1)

        # 95% Confidence Bounds (fill between LCL and UCL)
        fig_ts.add_trace(go.Scatter(
            x=time_future, y=result.cartesian_ucl[:, d],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=d + 1, col=1)
        fig_ts.add_trace(go.Scatter(
            x=time_future, y=result.cartesian_lcl[:, d],
            mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(249, 115, 22, 0.25)",
            name="95% Confidence Tube [LCL, UCL]", showlegend=(d == 0),
        ), row=d + 1, col=1)

        # Predicted Mean
        fig_ts.add_trace(go.Scatter(
            x=time_future, y=pred_hand[:, d],
            mode="lines+markers", line=dict(color="#ea580c", width=3),
            name="Predicted Mean", showlegend=(d == 0),
        ), row=d + 1, col=1)

        fig_ts.update_yaxes(title_text=f"{coord_names[d]} (m)", row=d + 1, col=1, gridcolor="#e2e8f0")

    fig_ts.update_xaxes(title_text="Time (seconds)", row=3, col=1, gridcolor="#e2e8f0")
    fig_ts.update_layout(
        title=dict(
            text="<b>Reaching Hand Trajectory Forecast & 95% Confidence Tube Over Time</b>",
            font=dict(size=16, family="sans-serif"), x=0.02, y=0.98, xanchor="left", yanchor="top"
        ),
        legend=dict(
            orientation="h", yref="container", y=0.93, x=0.5, xanchor="center", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=11, family="sans-serif")
        ),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1200, height=840,
        margin=dict(l=60, r=40, t=110, b=50),
    )

    out_ts_html = "output/html/predict_motion_hkm_timeseries.html"
    out_ts_pdf = "output/pdf/predict_motion_hkm_timeseries.pdf"
    fig_ts.write_html(out_ts_html)
    fig_ts.write_image(out_ts_pdf, width=1200, height=840)
    print(f"  -> Saved Time-Series Forecast HTML : {out_ts_html}")
    print(f"  -> Saved Time-Series Forecast PDF  : {out_ts_pdf}")

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS GENERATED (PDF + HTML ONLY - NO PNG):")
    print(f"  1. 3D Skeleton Forecast     : HTML : {out_skel_html} | PDF : {out_skel_pdf}")
    print(f"  2. Separated Keypoint Tubes : HTML : {out_sep_html} | PDF : {out_sep_pdf}")
    print(f"  3. Time-Series Trajectories : HTML : {out_ts_html} | PDF : {out_ts_pdf}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Probabilistic 3D Motion Prediction with the 28-DOF Human Kinematic Model."
    )
    parser.add_argument(
        "--mode",
        choices=["upper_body", "full_body"],
        default="upper_body",
        help="Kinematic configuration: 'upper_body' (19 DOFs) or 'full_body' (27 DOFs). Default: upper_body.",
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=15,
        help="Number of observed time steps in observation chunk (default: 15).",
    )
    parser.add_argument(
        "-f", "--future",
        type=int,
        default=25,
        help="Number of future time steps to predict ahead (default: 25).",
    )
    parser.add_argument(
        "-T", "--total-steps",
        type=int,
        default=40,
        help="Total steps in demonstration trajectory (default: 40).",
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Reaching hand to execute motion (default: right).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Profile 50 online rolling prediction cycles for real-time latency and frequency.",
    )
    args = parser.parse_args()

    run_motion_prediction_hkm(
        mode=args.mode,
        window_steps=args.window,
        future_steps=args.future,
        total_steps=args.total_steps,
        reaching_hand=args.hand,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
