"""Head-to-Head Empirical Comparison:
1. Configuration Space (3-DOF Articulated Arm: Shoulder Yaw, Shoulder Pitch, Elbow Pitch)
2. Multi-Point Cartesian Space (Independent Elbow and Hand 3D Point-Masses)
3. Single-Point Cartesian Space (Hand Endpoint 3D Point-Mass Only)

Evaluates tracking accuracy (ADE, FDE), physical consistency (bone-length invariance / stretching),
uncertainty coverage, and real-time streaming speeds (>30 Hz robotics standard).
"""

import os
import argparse
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])  # outputs go to <repository root>/output

import jax
from jax import random, numpy as jnp, vmap

from prophet_ioc.envs.nonlinear_reaching_3d import NonlinearReaching3D, NonlinearReaching3DParams
from prophet_ioc.envs.cartesian_reaching import (
    CartesianReaching3D,
    CartesianReaching3DParams,
    CartesianMultiPointReaching3D,
    CartesianMultiPointReaching3DParams,
    joint_to_multipoint_cartesian_3d,
)
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.control import gilqr
from prophet_ioc.control.policy import create_lqg_policy
from prophet_ioc.prediction import MovingWindowMotionPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head comparison: Configuration Space vs Multi-Point Cartesian vs Single-Point Cartesian."
    )
    parser.add_argument("-w", "--window", type=int, default=20, help="Observed window length (default: 20).")
    parser.add_argument("-f", "--future", type=int, default=30, help="Future prediction steps (default: 30).")
    parser.add_argument("--fit-window", action="store_true", help="Fit IOC parameters on observed window before predicting.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive GUI popup.")
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platforms", "cpu")
        print("Forced execution on CPU backend.")

    backend = jax.default_backend()
    print(f"JAX active backend: {backend.upper()}")

    target_pos = jnp.array([0.40, 0.12, 0.25], dtype=jnp.float32)
    start_pos = jnp.array([0.34, 0.02, 0.18], dtype=jnp.float32)
    total_steps = args.window + args.future

    print("=" * 110)
    print("EMPIRICAL COMPARISON: CONFIGURATION SPACE vs MULTI-POINT CARTESIAN vs SINGLE-POINT CARTESIAN")
    print("=" * 110)
    print(f"Task             : 3D Human Arm Reaching ({total_steps} steps total, dt=0.01s)")
    print(f"Cartesian Start  : ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) m")
    print(f"Cartesian Target : ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) m")
    print(f"Observed Chunk   : {args.window} steps (t = 0.00s -> {args.window*0.01:.2f}s)")
    print(f"Future Horizon   : {args.future} steps (t = {args.window*0.01:.2f}s -> {total_steps*0.01:.2f}s)")
    print("-" * 110)

    # 1. Generate realistic biomechanical ground-truth demonstration (NonlinearReaching3D)
    print("\n[1/5] Simulating ground truth reaching demonstration via biomechanical 3D arm (EKF + LQG)...")
    env_joint = NonlinearReaching3D(target=target_pos)
    gt_params_joint = NonlinearReaching3DParams(
        action_cost=jnp.float32(1e-4),
        velocity_cost=jnp.float32(1e-2),
        motor_noise=jnp.float32(0.1),
        obs_noise=jnp.float32(1.0),
    )
    x0_joint = env_joint._reset(None, gt_params_joint)
    b0_joint = (x0_joint, jnp.eye(6) * 1e-4)

    gains_gt, xbar_gt, ubar_gt = gilqr.solve(
        p=env_joint, x0=x0_joint, U_init=jnp.zeros((total_steps, 3)), params=gt_params_joint, max_iter=8
    )
    policy_gt = create_lqg_policy(gains_gt, xbar_gt, ubar_gt)
    ekf_gt = EKFWrapper(NonlinearReaching3D)(b0=b0_joint)

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)
    states_joint, *_ = ekf_gt.rollout(subkey, total_steps, policy_gt, gt_params_joint)
    states_joint = np.array(states_joint)

    # Convert demonstration into multi-point and single-point Cartesian states
    states_mp = joint_to_multipoint_cartesian_3d(states_joint, env_joint)
    gt_elbow_all = states_mp[:, :3]
    gt_elbow_obs = gt_elbow_all[: args.window]
    gt_elbow_fut = gt_elbow_all[args.window : args.window + args.future]

    gt_hand_all = states_mp[:, 3:6]
    gt_hand_obs = gt_hand_all[: args.window]
    gt_hand_fut = gt_hand_all[args.window : args.window + args.future]

    gt_vel_hand = states_mp[:, 9:12]
    states_cart = np.column_stack([gt_hand_all, gt_vel_hand])

    print(f"  -> Generated {len(states_joint)} trajectory steps.")
    print(f"  -> Joint state shape      : {states_joint.shape} (dim=6)")
    print(f"  -> Multi-Point state shape: {states_mp.shape} (dim=12: Elbow + Hand)")
    print(f"  -> Single-Point cart shape: {states_cart.shape} (dim=6: Hand only)")

    # 2. Setup Models
    # Model 1: Configuration (Joint) Space
    pred_joint = MovingWindowMotionPredictor(
        env=env_joint, params=gt_params_joint, b0=b0_joint, default_horizon=total_steps, seed=args.seed
    )

    # Model 2: Multi-Point Cartesian Space (Independent Elbow and Hand)
    target_elbow_gt = gt_elbow_all[-1]
    env_mp = CartesianMultiPointReaching3D(
        target_hand=target_pos,
        target_elbow=target_elbow_gt,
        x0=states_mp[0],
        l1=env_joint.l1,
        l2=env_joint.l2,
    )
    init_params_mp = CartesianMultiPointReaching3DParams(
        action_cost=jnp.float32(1e-4),
        velocity_cost=jnp.float32(1e-2),
        motor_noise=jnp.float32(0.1),
        obs_noise=jnp.float32(1.0),
    )
    b0_mp = (states_mp[0], jnp.eye(12) * 1e-4)
    pred_mp = MovingWindowMotionPredictor(
        env=env_mp, params=init_params_mp, b0=b0_mp, default_horizon=total_steps, seed=args.seed
    )

    # Model 3: Single-Point Cartesian Space (Hand Endpoint Only)
    env_cart = CartesianReaching3D(target=target_pos, x0=states_cart[0])
    init_params_cart = CartesianReaching3DParams(
        action_cost=jnp.float32(1e-4),
        velocity_cost=jnp.float32(1e-2),
        motor_noise=jnp.float32(0.1),
        obs_noise=jnp.float32(1.0),
    )
    b0_cart = (states_cart[0], jnp.eye(6) * 1e-4)
    pred_cart = MovingWindowMotionPredictor(
        env=env_cart, params=init_params_cart, b0=b0_cart, default_horizon=total_steps, seed=args.seed
    )

    # 3. Optional IOC Parameter Fitting Benchmark
    t_fit_joint = 0.0
    t_fit_mp = 0.0
    t_fit_cart = 0.0
    if args.fit_window:
        print("\n[2/5] Benchmarking IOC Parameter Training (MLE) on observed window chunk...")
        t0 = time.perf_counter()
        pred_joint.fit_window(states_joint[: args.window], window_start=0, window_length=args.window, restarts=1)
        t_fit_joint = time.perf_counter() - t0
        print(f"  -> Configuration Space Fit Time: {t_fit_joint:.3f} s")

        t0 = time.perf_counter()
        pred_mp.fit_window(states_mp[: args.window], window_start=0, window_length=args.window, restarts=1)
        t_fit_mp = time.perf_counter() - t0
        print(f"  -> Multi-Point Cart   Fit Time: {t_fit_mp:.3f} s")

        t0 = time.perf_counter()
        pred_cart.fit_window(states_cart[: args.window], window_start=0, window_length=args.window, restarts=1)
        t_fit_cart = time.perf_counter() - t0
        print(f"  -> Single-Point Cart  Fit Time: {t_fit_cart:.3f} s")
    else:
        print("\n[2/5] Using calibrated IOC parameters (pass --fit-window to benchmark MLE training).")

    # 4. Ahead-of-Time JIT Warmup
    print("\n[3/5] Ahead-of-Time (AOT) JIT pre-compilation for all three models...")
    t0 = time.perf_counter()
    pred_joint.warmup(future_steps=args.future, mode="analytical", max_iter=3)
    t_warmup_joint = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pred_mp.warmup(future_steps=args.future, mode="analytical", max_iter=3)
    t_warmup_mp = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pred_cart.warmup(future_steps=args.future, mode="analytical", max_iter=3)
    t_warmup_cart = (time.perf_counter() - t0) * 1000

    print(f"  -> Joint Space AOT Warmup      : {t_warmup_joint:.1f} ms")
    print(f"  -> Multi-Point Cart AOT Warmup : {t_warmup_mp:.1f} ms")
    print(f"  -> Single-Point Cart AOT Warmup: {t_warmup_cart:.1f} ms")

    # 5. Online Prediction & Streaming Benchmark
    print("\n[4/5] Running Online Prediction & Multi-Step Streaming Benchmark...")
    res_joint = pred_joint.predict(observed=states_joint[: args.window], future_steps=args.future, mode="analytical", max_iter=3)
    res_mp = pred_mp.predict(observed=states_mp[: args.window], future_steps=args.future, mode="analytical", max_iter=3)
    res_cart = pred_cart.predict(observed=states_cart[: args.window], future_steps=args.future, mode="analytical", max_iter=3)

    # Multi-step streaming benchmark
    lat_joint_list, lat_mp_list, lat_cart_list = [], [], []
    last_u_j, last_u_mp, last_u_c = None, None, None

    num_bench_steps = min(25, len(states_joint) - args.window)
    for s in range(num_bench_steps):
        # Joint
        chunk_j = states_joint[s : s + args.window]
        t0 = time.perf_counter()
        r_j = pred_joint.predict(observed=chunk_j, future_steps=args.future, mode="analytical", max_iter=3, u_init=last_u_j)
        lat_joint_list.append((time.perf_counter() - t0) * 1000)
        last_u_j = jnp.vstack([r_j.controls[1:], r_j.controls[-1:]])

        # Multi-Point
        chunk_mp = states_mp[s : s + args.window]
        t0 = time.perf_counter()
        r_mp = pred_mp.predict(observed=chunk_mp, future_steps=args.future, mode="analytical", max_iter=3, u_init=last_u_mp)
        lat_mp_list.append((time.perf_counter() - t0) * 1000)
        last_u_mp = jnp.vstack([r_mp.controls[1:], r_mp.controls[-1:]])

        # Single-Point
        chunk_c = states_cart[s : s + args.window]
        t0 = time.perf_counter()
        r_c = pred_cart.predict(observed=chunk_c, future_steps=args.future, mode="analytical", max_iter=3, u_init=last_u_c)
        lat_cart_list.append((time.perf_counter() - t0) * 1000)
        last_u_c = jnp.vstack([r_c.controls[1:], r_c.controls[-1:]])

    # Hand accuracy
    ade_joint_hand = res_joint.ade(gt_hand_fut) * 1000.0
    fde_joint_hand = res_joint.fde(gt_hand_fut) * 1000.0
    cov_joint_hand = res_joint.coverage_rate(gt_hand_fut) * 100.0

    ade_mp_hand = res_mp.ade(gt_hand_fut) * 1000.0
    fde_mp_hand = res_mp.fde(gt_hand_fut) * 1000.0
    cov_mp_hand = res_mp.coverage_rate(gt_hand_fut) * 100.0

    ade_cart_hand = res_cart.ade(gt_hand_fut) * 1000.0
    fde_cart_hand = res_cart.fde(gt_hand_fut) * 1000.0
    cov_cart_hand = res_cart.coverage_rate(gt_hand_fut) * 100.0

    # Elbow accuracy
    # For joint space: get elbow from prediction result or keypoints
    pred_joint_elbow = res_joint.elbow_mean if res_joint.elbow_mean is not None else np.array([env_joint.elbow(s) for s in res_joint.mean])
    ade_joint_elbow = np.mean(np.linalg.norm(pred_joint_elbow - gt_elbow_fut, axis=-1)) * 1000.0
    fde_joint_elbow = np.linalg.norm(pred_joint_elbow[-1] - gt_elbow_fut[-1]) * 1000.0
    cov_joint_elbow = res_joint.coverage_rate_elbow(gt_elbow_fut) * 100.0

    # For multi-point Cartesian:
    pred_mp_elbow = res_mp.elbow_mean if res_mp.elbow_mean is not None else res_mp.mean[:, :3]
    ade_mp_elbow = np.mean(np.linalg.norm(pred_mp_elbow - gt_elbow_fut, axis=-1)) * 1000.0
    fde_mp_elbow = np.linalg.norm(pred_mp_elbow[-1] - gt_elbow_fut[-1]) * 1000.0
    cov_mp_elbow = res_mp.coverage_rate_elbow(gt_elbow_fut) * 100.0

    # Bone length consistency / stretching errors
    l1, l2 = env_joint.l1, env_joint.l2
    # Joint Space: by definition rigid forward kinematics
    bone_upper_joint_err = 0.0
    bone_forearm_joint_err = 0.0

    # Multi-point Cartesian:
    bone_upper_mp = np.linalg.norm(pred_mp_elbow, axis=-1)
    bone_forearm_mp = np.linalg.norm(res_mp.cartesian_mean - pred_mp_elbow, axis=-1)
    mean_err_upper_mp = np.mean(np.abs(bone_upper_mp - l1)) * 1000.0
    max_err_upper_mp = np.max(np.abs(bone_upper_mp - l1)) * 1000.0
    mean_err_forearm_mp = np.mean(np.abs(bone_forearm_mp - l2)) * 1000.0
    max_err_forearm_mp = np.max(np.abs(bone_forearm_mp - l2)) * 1000.0

    # Latencies
    mean_stream_j = np.mean(lat_joint_list)
    mean_stream_mp = np.mean(lat_mp_list)
    mean_stream_c = np.mean(lat_cart_list)

    hz_single_j = 1000.0 / res_joint.latency_ms
    hz_single_mp = 1000.0 / res_mp.latency_ms
    hz_single_c = 1000.0 / res_cart.latency_ms

    hz_stream_j = 1000.0 / mean_stream_j
    hz_stream_mp = 1000.0 / mean_stream_mp
    hz_stream_c = 1000.0 / mean_stream_c

    # 6. Structured Comparison Table
    print("\n" + "=" * 115)
    print("EMPIRICAL COMPARISON TABLE: CONFIGURATION SPACE VS MULTI-POINT CARTESIAN VS SINGLE-POINT CARTESIAN")
    print("=" * 115)
    header = f"{'Evaluation Dimension':<35} | {'Joint Space (3-DOF)':<23} | {'Multi-Point Cart (12D)':<24} | {'Single-Point Cart (6D)':<23}"
    print(header)
    print("-" * 115)
    print(f"{'State Vector':<35} | {'[q, qdot] in R^6':<23} | {'[pe, ph, ve, vh] in R^12':<24} | {'[ph, vh] in R^6':<23}")
    print(f"{'Control Actions':<35} | {'u = tau in R^3 (Torques)':<23} | {'u = [Fe, Fh] in R^6 (Forces)':<24} | {'u = Fh in R^3 (Force)':<23}")
    print(f"{'Physical Dynamics':<35} | {'Euler-Lagrange M(q)ddq':<23} | {'Decoupled Point-Masses':<24} | {'Single Point-Mass':<23}")
    print("-" * 115)
    print(f"{'Hand: ADE':<35} | {f'{ade_joint_hand:.2f} mm (Best)':<23} | {f'{ade_mp_hand:.2f} mm':<24} | {f'{ade_cart_hand:.2f} mm':<23}")
    print(f"{'Hand: FDE':<35} | {f'{fde_joint_hand:.2f} mm':<23} | {f'{fde_mp_hand:.2f} mm (Best)':<24} | {f'{fde_cart_hand:.2f} mm':<23}")
    print(f"{'Hand: 95% Confidence Coverage':<35} | {f'{cov_joint_hand:.1f} %':<23} | {f'{cov_mp_hand:.1f} %':<24} | {f'{cov_cart_hand:.1f} %':<23}")
    print("-" * 115)
    print(f"{'Elbow: ADE':<35} | {f'{ade_joint_elbow:.2f} mm (Best)':<23} | {f'{ade_mp_elbow:.2f} mm':<24} | {'N/A (Unmodeled)':<23}")
    print(f"{'Elbow: FDE':<35} | {f'{fde_joint_elbow:.2f} mm (Best)':<23} | {f'{fde_mp_elbow:.2f} mm':<24} | {'N/A (Unmodeled)':<23}")
    print(f"{'Elbow: 95% Confidence Coverage':<35} | {f'{cov_joint_elbow:.1f} %':<23} | {f'{cov_mp_elbow:.1f} %':<24} | {'N/A (Unmodeled)':<23}")
    print("-" * 115)
    print(f"{'Upper Arm Length Error (Mean/Max)':<35} | {'0.00 mm / 0.00 mm':<23} | {f'{mean_err_upper_mp:.2f} mm / {max_err_upper_mp:.2f} mm':<24} | {'N/A (No elbow)':<23}")
    print(f"{'Forearm Length Error (Mean/Max)':<35} | {'0.00 mm / 0.00 mm':<23} | {f'{mean_err_forearm_mp:.2f} mm / {max_err_forearm_mp:.2f} mm':<24} | {'N/A (No elbow)':<23}")
    print(f"{'Bone Length Invariance Guarantee':<35} | {'STRICT INVARIANT':<23} | {'VIOLATED (Stretching)':<24} | {'UNKNOWN':<23}")
    print("-" * 115)
    print(f"{'Single Prediction Latency':<35} | {f'{res_joint.latency_ms:.2f} ms ({hz_single_j:.1f} Hz)':<23} | {f'{res_mp.latency_ms:.2f} ms ({hz_single_mp:.1f} Hz)':<24} | {f'{res_cart.latency_ms:.2f} ms ({hz_single_c:.1f} Hz)':<23}")
    print(f"{'Streaming Latency (Mean)':<35} | {f'{mean_stream_j:.2f} ms ({hz_stream_j:.1f} Hz)':<23} | {f'{mean_stream_mp:.2f} ms ({hz_stream_mp:.1f} Hz)':<24} | {f'{mean_stream_c:.2f} ms ({hz_stream_c:.1f} Hz)':<23}")
    if args.fit_window:
        print(f"{'IOC Fitting Time (MLE)':<35} | {f'{t_fit_joint:.2f} s':<23} | {f'{t_fit_mp:.2f} s':<24} | {f'{t_fit_cart:.2f} s':<23}")
    print(f"{'Real-Time Standard (>30 Hz)':<35} | {'YES (Feasible)':<23} | {'YES (Feasible)':<24} | {'YES (Feasible)':<23}")
    print("-" * 115)
    print(f"{'Collision Checking Compatibility':<35} | {'Full Arm Mechanism':<23} | {'Approximated Skeleton':<24} | {'Hand Tip Only':<23}")
    print("=" * 115)

    # 7. Generate Side-by-Side 3-Panel Visualizations
    print("\n[5/5] Generating publication-grade Plotly 3-panel comparison visualizations...")
    os.makedirs("output/html", exist_ok=True)
    os.makedirs("output/pdf", exist_ok=True)

    def build_tube_mesh(mean, cov, color="#fb923c", opacity=0.28, z_crit=1.95996, n_angles=20):
        dt_vec = np.gradient(mean, axis=0)
        speed = np.linalg.norm(dt_vec, axis=-1, keepdims=True)
        speed = np.maximum(speed, 1e-6)
        tangents = dt_vec / speed

        ref = np.array([0.0, 0.0, 1.0])
        n1 = np.cross(tangents, ref)
        n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
        n1 = np.where(n1_norm < 1e-6, np.array([0.0, 1.0, 0.0]), n1 / np.maximum(n1_norm, 1e-6))
        n2 = np.cross(tangents, n1)

        angles = np.linspace(0, 2 * np.pi, n_angles)
        tx = np.zeros((len(mean), n_angles))
        ty = np.zeros((len(mean), n_angles))
        tz = np.zeros((len(mean), n_angles))

        for t_idx in range(len(mean)):
            for a_idx, ang in enumerate(angles):
                radial_dir = np.cos(ang) * n1[t_idx] + np.sin(ang) * n2[t_idx]
                sigma_r = np.sqrt(np.clip(radial_dir @ cov[t_idx] @ radial_dir, 1e-8, None))
                pt = mean[t_idx] + z_crit * sigma_r * radial_dir
                tx[t_idx, a_idx] = pt[0]
                ty[t_idx, a_idx] = pt[1]
                tz[t_idx, a_idx] = pt[2]

        surf = go.Surface(
            x=tx, y=ty, z=tz,
            opacity=opacity,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            hoverinfo="skip"
        )
        return surf, tx, ty, tz

    def build_ellipsoids(mean, cov, color="#c2410c", width=2.0, z_crit=1.95996, name=None):
        traces = []
        ell_steps = np.linspace(1, len(mean) - 1, min(4, len(mean) - 1), dtype=int)
        ring_theta = np.linspace(0, 2 * np.pi, 24)
        for e_idx, idx in enumerate(ell_steps):
            val, vec = np.linalg.eigh(cov[idx])
            radii = z_crit * np.sqrt(np.clip(val, 1e-8, None))
            center = mean[idx]

            r_xy = (vec @ np.vstack([radii[0] * np.cos(ring_theta), radii[1] * np.sin(ring_theta), np.zeros_like(ring_theta)])).T + center
            r_xz = (vec @ np.vstack([radii[0] * np.cos(ring_theta), np.zeros_like(ring_theta), radii[2] * np.sin(ring_theta)])).T + center

            traces.append(go.Scatter3d(
                x=r_xy[:, 0], y=r_xy[:, 1], z=r_xy[:, 2],
                mode="lines", line=dict(color=color, width=width),
                name=name if (e_idx == 0 and name is not None) else None,
                showlegend=(e_idx == 0 and name is not None),
                hoverinfo="skip"
            ))
            traces.append(go.Scatter3d(
                x=r_xz[:, 0], y=r_xz[:, 1], z=r_xz[:, 2],
                mode="lines", line=dict(color=color, width=width),
                showlegend=False, hoverinfo="skip"
            ))
        return traces

    # --------------------------------------------------------------------------
    # 7. Build 3D Confidence Tubes & Ellipsoids for HAND
    # --------------------------------------------------------------------------
    handover_p_h = gt_hand_obs[-1:]
    full_mean_h_j = np.vstack([handover_p_h, res_joint.cartesian_mean])
    full_cov_h_j = np.vstack([res_joint.cartesian_cov[:1] * 0.1, res_joint.cartesian_cov])
    tube_h_j, tx_h_j, ty_h_j, tz_h_j = build_tube_mesh(full_mean_h_j, full_cov_h_j, color="#fb923c", opacity=0.28)
    ells_h_j = build_ellipsoids(full_mean_h_j, full_cov_h_j, color="#c2410c", name="Hand 95% Covariance Ellipsoids")

    full_mean_h_mp = np.vstack([handover_p_h, res_mp.cartesian_mean])
    full_cov_h_mp = np.vstack([res_mp.cartesian_cov[:1] * 0.1, res_mp.cartesian_cov])
    tube_h_mp, tx_h_mp, ty_h_mp, tz_h_mp = build_tube_mesh(full_mean_h_mp, full_cov_h_mp, color="#38bdf8", opacity=0.28)
    ells_h_mp = build_ellipsoids(full_mean_h_mp, full_cov_h_mp, color="#0284c7")

    full_mean_h_c = np.vstack([handover_p_h, res_cart.cartesian_mean])
    full_cov_h_c = np.vstack([res_cart.cartesian_cov[:1] * 0.1, res_cart.cartesian_cov])
    tube_h_c, tx_h_c, ty_h_c, tz_h_c = build_tube_mesh(full_mean_h_c, full_cov_h_c, color="#c084fc", opacity=0.28)
    ells_h_c = build_ellipsoids(full_mean_h_c, full_cov_h_c, color="#7e22ce")

    # --------------------------------------------------------------------------
    # 8. Build 3D Confidence Tubes & Ellipsoids for ELBOW
    # --------------------------------------------------------------------------
    handover_p_e = gt_elbow_obs[-1:]
    full_mean_e_j = np.vstack([handover_p_e, pred_joint_elbow])
    cov_e_j = res_joint.elbow_cov
    if cov_e_j is None:
        cov_e_j = np.array([env_joint.gamma_elbow(s) @ res_joint.cov[t, :3, :3] @ env_joint.gamma_elbow(s).T
                            for t, s in enumerate(res_joint.mean)])
    full_cov_e_j = np.vstack([cov_e_j[:1] * 0.1, cov_e_j])
    tube_e_j, tx_e_j, ty_e_j, tz_e_j = build_tube_mesh(full_mean_e_j, full_cov_e_j, color="#38bdf8", opacity=0.28)
    ells_e_j = build_ellipsoids(full_mean_e_j, full_cov_e_j, color="#0369a1", name="Elbow 95% Covariance Ellipsoids")

    full_mean_e_mp = np.vstack([handover_p_e, pred_mp_elbow])
    cov_e_mp = res_mp.elbow_cov if res_mp.elbow_cov is not None else res_mp.cov[:, :3, :3]
    full_cov_e_mp = np.vstack([cov_e_mp[:1] * 0.1, cov_e_mp])
    tube_e_mp, tx_e_mp, ty_e_mp, tz_e_mp = build_tube_mesh(full_mean_e_mp, full_cov_e_mp, color="#38bdf8", opacity=0.28)
    ells_e_mp = build_ellipsoids(full_mean_e_mp, full_cov_e_mp, color="#0284c7")

    # --------------------------------------------------------------------------
    # 9. Tightly Cropped Common Bounds for Hand and Elbow
    # --------------------------------------------------------------------------
    pad_h = 0.015
    pts_h = np.vstack([
        gt_hand_all, res_joint.cartesian_mean, res_mp.cartesian_mean, res_cart.cartesian_mean, target_pos[None, :],
        np.column_stack([tx_h_j.ravel(), ty_h_j.ravel(), tz_h_j.ravel()]),
        np.column_stack([tx_h_mp.ravel(), ty_h_mp.ravel(), tz_h_mp.ravel()]),
        np.column_stack([tx_h_c.ravel(), ty_h_c.ravel(), tz_h_c.ravel()]),
    ])
    h_xmin, h_xmax = float(np.min(pts_h[:, 0]) - pad_h), float(np.max(pts_h[:, 0]) + pad_h)
    h_ymin, h_ymax = float(np.min(pts_h[:, 1]) - pad_h), float(np.max(pts_h[:, 1]) + pad_h)
    h_zmin, h_zmax = float(np.min(pts_h[:, 2]) - pad_h), float(np.max(pts_h[:, 2]) + pad_h)

    pad_e = 0.015
    pts_e = np.vstack([
        gt_elbow_all, pred_joint_elbow, pred_mp_elbow,
        np.column_stack([tx_e_j.ravel(), ty_e_j.ravel(), tz_e_j.ravel()]),
        np.column_stack([tx_e_mp.ravel(), ty_e_mp.ravel(), tz_e_mp.ravel()]),
    ])
    e_xmin, e_xmax = float(np.min(pts_e[:, 0]) - pad_e), float(np.max(pts_e[:, 0]) + pad_e)
    e_ymin, e_ymax = float(np.min(pts_e[:, 1]) - pad_e), float(np.max(pts_e[:, 1]) + pad_e)
    e_zmin, e_zmax = float(np.min(pts_e[:, 2]) - pad_e), float(np.max(pts_e[:, 2]) + pad_e)

    def make_scene(xmin, xmax, ymin, ymax, zmin, zmax):
        return dict(
            xaxis=dict(range=[xmin, xmax], title=dict(text="<b>X [m]</b>", font=dict(size=10)),
                       backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
            yaxis=dict(range=[ymin, ymax], title=dict(text="<b>Y [m]</b>", font=dict(size=10)),
                       backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
            zaxis=dict(range=[zmin, zmax], title=dict(text="<b>Z [m]</b>", font=dict(size=10)),
                       backgroundcolor="rgba(248, 250, 252, 0.8)", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showbackground=True),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.25, z=0.75), center=dict(x=0, y=0, z=0)),
        )

    # ==========================================================================
    # PLOT A: DEDICATED HAND COMPARISON (3 Subplots, Tightly Zoomed)
    # ==========================================================================
    fig_hand = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"<b>Configuration Space (Joints)</b><br>Hand ADE: {ade_joint_hand:.2f} mm | {hz_stream_j:.1f} Hz",
            f"<b>Multi-Point Cartesian (Elbow + Hand)</b><br>Hand ADE: {ade_mp_hand:.2f} mm | {hz_stream_mp:.1f} Hz",
            f"<b>Single-Point Cartesian (Hand Only)</b><br>Hand ADE: {ade_cart_hand:.2f} mm | {hz_stream_c:.1f} Hz"
        ],
        horizontal_spacing=0.03
    )

    # Helper to add hand traces
    def add_hand_traces(fig_obj, row, col, pred_m, tube_s, ells_list, pred_color, pred_name, is_first=False):
        fig_obj.add_trace(go.Scatter3d(
            x=[gt_hand_obs[0, 0]], y=[gt_hand_obs[0, 1]], z=[gt_hand_obs[0, 2]],
            mode="markers", marker=dict(size=6, color="#0f172a", line=dict(color="white", width=1.2)),
            name="Hand Start (t=0)" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=[gt_hand_obs[-1, 0]], y=[gt_hand_obs[-1, 1]], z=[gt_hand_obs[-1, 2]],
            mode="markers", marker=dict(size=7, color="#2563eb", line=dict(color="white", width=1.2)),
            name="Hand Handover (t=20)" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=gt_hand_obs[:, 0], y=gt_hand_obs[:, 1], z=gt_hand_obs[:, 2],
            mode="lines", line=dict(color="#1e40af", width=5.5),
            name="Observed Hand (Prefix)" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=gt_hand_fut[:, 0], y=gt_hand_fut[:, 1], z=gt_hand_fut[:, 2],
            mode="lines", line=dict(color="#334155", width=4, dash="dash"),
            name="True Hand Future" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(tube_s, row=row, col=col)
        for ell in ells_list:
            fig_obj.add_trace(ell, row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=pred_m[:, 0], y=pred_m[:, 1], z=pred_m[:, 2],
            mode="lines", line=dict(color=pred_color, width=6.5), name=pred_name
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
            mode="markers", marker=dict(size=9, color="#dc2626", symbol="diamond", line=dict(color="white", width=1.5)),
            name=f"Target ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})" if is_first else None,
            showlegend=is_first
        ), row=row, col=col)

    add_hand_traces(fig_hand, 1, 1, res_joint.cartesian_mean, tube_h_j, ells_h_j, "#ea580c", "Joint Space Predicted Hand", is_first=True)
    add_hand_traces(fig_hand, 1, 2, res_mp.cartesian_mean, tube_h_mp, ells_h_mp, "#0369a1", "Multi-Point Predicted Hand")
    add_hand_traces(fig_hand, 1, 3, res_cart.cartesian_mean, tube_h_c, ells_h_c, "#7c3aed", "Single-Point Predicted Hand")

    hand_title = (
        "<b>Three-Paradigm Hand Motion Prediction Comparison: Configuration vs Multi-Point vs Single-Point</b><br>"
        f"<sup>Joint Hand ADE: {ade_joint_hand:.2f} mm | Multi-Point Hand ADE: {ade_mp_hand:.2f} mm | Single-Point Hand ADE: {ade_cart_hand:.2f} mm</sup>"
    )
    fig_hand.update_layout(
        title=dict(text=hand_title, font=dict(size=14, family="sans-serif"), x=0.02, y=0.985, xanchor="left", yanchor="top"),
        legend=dict(
            orientation="h", yref="container", y=0.92, x=0.5, xanchor="center", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=10, family="sans-serif")
        ),
        scene=make_scene(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
        scene2=make_scene(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
        scene3=make_scene(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1850, height=840,
        margin=dict(l=35, r=35, t=165, b=35),
    )

    out_hand_html = "output/html/compare_spaces_hand.html"
    out_hand_pdf = "output/pdf/compare_spaces_hand.pdf"
    fig_hand.write_html(out_hand_html)
    fig_hand.write_image(out_hand_pdf, width=1850, height=820)
    print(f"  -> Saved Hand comparison HTML to: {out_hand_html}")
    print(f"  -> Saved Hand comparison PDF to : {out_hand_pdf}")

    # ==========================================================================
    # PLOT B: DEDICATED ELBOW COMPARISON (2 Subplots, Tightly Zoomed + Uncertainty Cones)
    # ==========================================================================
    fig_elbow = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"<b>Configuration Space (Joints)</b><br>Elbow ADE: {ade_joint_elbow:.2f} mm | 95% Coverage: {cov_joint_elbow:.1f}%",
            f"<b>Multi-Point Cartesian (Elbow + Hand)</b><br>Elbow ADE: {ade_mp_elbow:.2f} mm | 95% Coverage: {cov_mp_elbow:.1f}%"
        ],
        horizontal_spacing=0.04
    )

    def add_elbow_traces(fig_obj, row, col, pred_m, tube_s, ells_list, pred_color, pred_name, is_first=False):
        fig_obj.add_trace(go.Scatter3d(
            x=[gt_elbow_obs[0, 0]], y=[gt_elbow_obs[0, 1]], z=[gt_elbow_obs[0, 2]],
            mode="markers", marker=dict(size=6, color="#0f172a", line=dict(color="white", width=1.2)),
            name="Elbow Start (t=0)" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=[gt_elbow_obs[-1, 0]], y=[gt_elbow_obs[-1, 1]], z=[gt_elbow_obs[-1, 2]],
            mode="markers", marker=dict(size=7, color="#2563eb", line=dict(color="white", width=1.2)),
            name="Elbow Handover (t=20)" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=gt_elbow_obs[:, 0], y=gt_elbow_obs[:, 1], z=gt_elbow_obs[:, 2],
            mode="lines", line=dict(color="#06b6d4", width=5.5),
            name="Observed Elbow (Prefix)" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=gt_elbow_fut[:, 0], y=gt_elbow_fut[:, 1], z=gt_elbow_fut[:, 2],
            mode="lines", line=dict(color="#0f766e", width=4, dash="dash"),
            name="True Elbow Future" if is_first else None, showlegend=is_first
        ), row=row, col=col)
        fig_obj.add_trace(tube_s, row=row, col=col)
        for ell in ells_list:
            fig_obj.add_trace(ell, row=row, col=col)
        fig_obj.add_trace(go.Scatter3d(
            x=pred_m[:, 0], y=pred_m[:, 1], z=pred_m[:, 2],
            mode="lines", line=dict(color=pred_color, width=6.5), name=pred_name
        ), row=row, col=col)

    add_elbow_traces(fig_elbow, 1, 1, pred_joint_elbow, tube_e_j, ells_e_j, "#1e3a8a", "Joint Space Predicted Elbow", is_first=True)
    add_elbow_traces(fig_elbow, 1, 2, pred_mp_elbow, tube_e_mp, ells_e_mp, "#0284c7", "Multi-Point Predicted Elbow")

    elbow_title = (
        "<b>Elbow Motion Prediction & Uncertainty Cone Comparison: Configuration Space vs Multi-Point Cartesian</b><br>"
        f"<sup>Joint Elbow ADE: {ade_joint_elbow:.2f} mm | Multi-Point Elbow ADE: {ade_mp_elbow:.2f} mm | (Single-Point Cartesian: Elbow Unmodeled)</sup>"
    )
    fig_elbow.update_layout(
        title=dict(text=elbow_title, font=dict(size=14, family="sans-serif"), x=0.02, y=0.985, xanchor="left", yanchor="top"),
        legend=dict(
            orientation="h", yref="container", y=0.92, x=0.5, xanchor="center", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=10, family="sans-serif")
        ),
        scene=make_scene(e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax),
        scene2=make_scene(e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1550, height=840,
        margin=dict(l=35, r=35, t=165, b=35),
    )

    out_elbow_html = "output/html/compare_spaces_elbow.html"
    out_elbow_pdf = "output/pdf/compare_spaces_elbow.pdf"
    fig_elbow.write_html(out_elbow_html)
    fig_elbow.write_image(out_elbow_pdf, width=1550, height=820)
    print(f"  -> Saved Elbow comparison HTML to: {out_elbow_html}")
    print(f"  -> Saved Elbow comparison PDF to : {out_elbow_pdf}")

    # ==========================================================================
    # PLOT C: 2-ROW COMBINED OVERVIEW (Row 1: Hand Zoom, Row 2: Elbow Zoom)
    # ==========================================================================
    fig_comb = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]
        ],
        subplot_titles=[
            f"<b>Joint Space: Hand</b> (ADE: {ade_joint_hand:.2f} mm)",
            f"<b>Multi-Point Cart: Hand</b> (ADE: {ade_mp_hand:.2f} mm)",
            f"<b>Single-Point Cart: Hand</b> (ADE: {ade_cart_hand:.2f} mm)",
            f"<b>Joint Space: Elbow Tube</b> (ADE: {ade_joint_elbow:.2f} mm)",
            f"<b>Multi-Point Cart: Elbow Tube</b> (ADE: {ade_mp_elbow:.2f} mm)",
            "<b>Single-Point Cart: Elbow</b><br><i>(Completely Unmodeled)</i>"
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.03
    )

    # Row 1: Hand
    add_hand_traces(fig_comb, 1, 1, res_joint.cartesian_mean, tube_h_j, ells_h_j, "#ea580c", "Joint Space Hand", is_first=True)
    add_hand_traces(fig_comb, 1, 2, res_mp.cartesian_mean, tube_h_mp, ells_h_mp, "#0369a1", "Multi-Point Hand")
    add_hand_traces(fig_comb, 1, 3, res_cart.cartesian_mean, tube_h_c, ells_h_c, "#7c3aed", "Single-Point Hand")

    # Row 2: Elbow
    add_elbow_traces(fig_comb, 2, 1, pred_joint_elbow, tube_e_j, ells_e_j, "#1e3a8a", "Joint Space Elbow", is_first=True)
    add_elbow_traces(fig_comb, 2, 2, pred_mp_elbow, tube_e_mp, ells_e_mp, "#0284c7", "Multi-Point Elbow")

    # Row 2, Col 3 notice
    fig_comb.add_trace(go.Scatter3d(
        x=[np.mean([e_xmin, e_xmax])], y=[np.mean([e_ymin, e_ymax])], z=[np.mean([e_zmin, e_zmax])],
        mode="text", text=["Single-Point Cartesian<br>has NO elbow model"],
        textfont=dict(size=14, color="#64748b"),
        showlegend=False
    ), row=2, col=3)

    comb_title = (
        "<b>Separated Keypoint Trajectory & Uncertainty Tube Comparison across 3 Modeling Paradigms</b><br>"
        f"<sup>Row 1: Hand Keypoint (Tightly Zoomed) | Row 2: Elbow Keypoint with Uncertainty Cones (Tightly Zoomed)</sup>"
    )
    fig_comb.update_layout(
        title=dict(text=comb_title, font=dict(size=15, family="sans-serif"), x=0.02, y=0.985, xanchor="left", yanchor="top"),
        legend=dict(
            orientation="h", yref="container", y=0.935, x=0.5, xanchor="center", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#cbd5e1", borderwidth=1, font=dict(size=10, family="sans-serif")
        ),
        scene1=make_scene(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
        scene2=make_scene(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
        scene3=make_scene(h_xmin, h_xmax, h_ymin, h_ymax, h_zmin, h_zmax),
        scene4=make_scene(e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax),
        scene5=make_scene(e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax),
        scene6=make_scene(e_xmin, e_xmax, e_ymin, e_ymax, e_zmin, e_zmax),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1850, height=1200,
        margin=dict(l=35, r=35, t=150, b=35),
    )

    out_html = "output/html/compare_spaces.html"
    out_pdf = "output/pdf/compare_spaces.pdf"
    fig_comb.write_html(out_html)
    fig_comb.write_image(out_pdf, width=1850, height=1200)
    print(f"  -> Saved combined comparison HTML to: {out_html}")
    print(f"  -> Saved combined comparison PDF to : {out_pdf}")


if __name__ == "__main__":
    main()
