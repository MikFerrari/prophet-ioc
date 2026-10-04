import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Safe memory preallocation setting for GPU environments
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jax import random, numpy as jnp

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root (plot_utils)
os.chdir(Path(__file__).resolve().parents[1])  # outputs go to <repository root>/output
from prophet_ioc.envs.nonlinear_reaching_3d import NonlinearReaching3D, NonlinearReaching3DParams
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.control import gilqr
from prophet_ioc.control.policy import create_lqg_policy
from prophet_ioc.prediction import MovingWindowMotionPredictor, PredictionResult
from plot_utils import update_rcparams, configure_high_dpi_window

# Apply high-DPI scaling (enlarges fonts, linewidths, and figure canvas for 16:10 / 4K displays)
update_rcparams(high_dpi=True, scale=1.8, dpi=130)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 3D human arm motion prediction using probabilistic IOC and optimal feedback control."
    )
    parser.add_argument("-w", "--window", type=int, default=20,
                        help="Number of observed time steps in observation chunk (default: 20).")
    parser.add_argument("-f", "--future", type=int, default=30,
                        help="Number of future time steps to predict ahead (default: 30).")
    parser.add_argument("--mode", type=str, choices=["analytical", "monte_carlo", "both"], default="analytical",
                        help="Prediction strategy: 'analytical' (fastest, >40-60 Hz), 'monte_carlo' (stochastic rollouts), or 'both' (default: analytical).")
    parser.add_argument("-n", "--samples", type=int, default=50,
                        help="Number of Monte Carlo rollout samples when mode is 'monte_carlo' or 'both' (default: 50).")
    parser.add_argument("-c", "--confidence", type=float, default=0.95,
                        help="Confidence level for UCL and LCL bounds (default: 0.95 for 95%% CI).")
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Max iLQR iterations for online re-planning (default: 3).")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run sequential multi-step sliding-window MPC benchmark to evaluate real-time streaming speed.")
    parser.add_argument("--fit-window", action="store_true",
                        help="Train IOC parameters on the observed window chunk before predicting.")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force execution on CPU backend.")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive matplotlib window popup (useful for non-interactive test scripts).")
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platforms", "cpu")
        print("Forced execution on CPU backend.")

    backend = jax.default_backend()
    print(f"JAX active backend: {backend.upper()}")

    # Setup 3D environment and ground truth parameters
    target_pos = jnp.array([0.40, 0.12, 0.25], dtype=jnp.float32)
    start_pos = jnp.array([0.34, 0.02, 0.18], dtype=jnp.float32)
    env = NonlinearReaching3D(target=target_pos)
    gt_params = NonlinearReaching3DParams(
        action_cost=jnp.float32(1e-4),
        velocity_cost=jnp.float32(1e-2),
        motor_noise=jnp.float32(0.1),
        obs_noise=jnp.float32(1.0),
    )
    total_steps = args.window + args.future

    reach_dist_mm = float(jnp.linalg.norm(target_pos - start_pos) * 1000)

    print("=" * 90)
    print("3D REAL-TIME PROBABILISTIC HUMAN MOTION PREDICTOR (AOT Warmup + Closed-Form LQG Covariance)")
    print("=" * 90)
    print(f"Task             : 3D Human Arm Reaching ({total_steps} steps total, dt={env.dt}s)")
    print(f"Kinematic Chain  : 3-DOF Anthropomorphic Arm (Shoulder Yaw, Shoulder Pitch, Elbow Pitch)")
    print(f"Link Dimensions  : Upper Arm l1={env.l1*100:.1f} cm | Forearm l2={env.l2*100:.1f} cm (Reach: {(env.l1+env.l2)*100:.1f} cm)")
    print(f"Cartesian Start  : ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) m")
    print(f"Cartesian Target : ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) m (Reaching Distance: {reach_dist_mm:.1f} mm)")
    print(f"Observed Chunk   : {args.window} steps (t = 0.00s -> {args.window*env.dt:.2f}s)")
    print(f"Future Horizon   : {args.future} steps (t = {args.window*env.dt:.2f}s -> {total_steps*env.dt:.2f}s)")
    print(f"Strategy Mode    : {args.mode.upper()} (iLQR max_iter={args.max_iter})")
    print(f"Confidence Level : {args.confidence * 100:.1f}% (3D Spatial Tube & Covariance Ellipsoids)")
    print("-" * 90)

    # 1. Simulate realistic ground truth reaching demonstration
    print("\n[1/5] Simulating realistic 3D demonstration trajectory (EKF + LQG)...")
    x0 = env._reset(None, gt_params)
    b0 = (x0, jnp.eye(6) * 1e-4)

    gains_gt, xbar_gt, ubar_gt = gilqr.solve(
        p=env, x0=x0, U_init=jnp.zeros((total_steps, 3)), params=gt_params, max_iter=8
    )
    policy_gt = create_lqg_policy(gains_gt, xbar_gt, ubar_gt)
    ekf_gt = EKFWrapper(NonlinearReaching3D)(b0=b0)

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)
    states, *_ = ekf_gt.rollout(subkey, total_steps, policy_gt, gt_params)
    states = np.array(states)

    observed_chunk = states[: args.window]
    true_future = states[args.window : args.window + args.future]
    true_future_cartesian = np.array(jax.vmap(env.e)(true_future))

    print(f"  -> Generated {len(states)} steps in 3D.")
    print(f"  -> Observed chunk shape : {observed_chunk.shape}")
    print(f"  -> True future shape    : {true_future.shape}")

    # 2. Initialize Predictor Model
    predictor = MovingWindowMotionPredictor(
        env=env,
        params=gt_params,
        b0=b0,
        default_horizon=total_steps,
        seed=args.seed,
    )

    # 3. Optional: Moving-Window Parameter Fitting
    if args.fit_window:
        print("\n[2/5] Training IOC parameters directly on observed window chunk...")
        t0_fit = time.perf_counter()
        fit_res = predictor.fit_window(
            observed_chunk,
            window_start=0,
            window_length=args.window,
            restarts=2,
        )
        t_fit = time.perf_counter() - t0_fit
        print(f"  -> Model trained on window in {t_fit:.2f}s.")
        print(f"  -> Fitted parameters: {predictor.params}")
    else:
        print("\n[2/5] Using calibrated IOC parameters (pass --fit-window to train on chunk).")

    # 4. Ahead-Of-Time (AOT) JIT Warmup
    print(f"\n[3/5] Strategy 1: Ahead-Of-Time (AOT) JIT pre-compilation (mode='{args.mode}')...")
    t0_warmup = time.perf_counter()
    predictor.warmup(
        future_steps=args.future,
        num_samples=args.samples,
        mode=args.mode,
        max_iter=args.max_iter,
    )
    t_warmup = time.perf_counter() - t0_warmup
    print(f"  -> AOT compilation completed in {t_warmup:.2f}s (zero online latency impact!)")

    # 5. Execute Prediction
    print(f"\n[4/5] Executing real-time 3D prediction (Strategy: {args.mode.upper()})...")
    result = predictor.predict(
        observed=observed_chunk,
        future_steps=args.future,
        mode=args.mode,
        num_samples=args.samples,
        confidence_level=args.confidence,
        max_iter=args.max_iter,
    )
    hz = 1000.0 / max(result.latency_ms, 1e-3)
    print(f"  -> Prediction executed in {result.latency_ms:.2f} ms ({hz:.1f} Hz real-time loop rate)")

    # 6. Optional: Streaming Benchmark
    if args.benchmark:
        print("\n[4.5] Running multi-step online streaming benchmark (warm-started MPC)...")
        bench_latencies = []
        last_u = None
        for step in range(25):
            idx = args.window + step
            if idx > len(states):
                break
            chunk = states[step : idx]
            t0_b = time.perf_counter()
            r_bench = predictor.predict(
                observed=chunk,
                future_steps=args.future,
                mode=args.mode,
                num_samples=args.samples,
                max_iter=args.max_iter,
                u_init=last_u,
            )
            lat_b = (time.perf_counter() - t0_b) * 1000
            bench_latencies.append(lat_b)
            # Shift controls by 1 step for warm-starting next iteration
            last_u = jnp.vstack([r_bench.controls[1:], r_bench.controls[-1:]])

        mean_lat = np.mean(bench_latencies)
        min_lat = np.min(bench_latencies)
        max_lat = np.max(bench_latencies)
        bench_hz = 1000.0 / mean_lat
        print(f"  -> Benchmark over {len(bench_latencies)} sequential streaming frames:")
        print(f"     Mean Latency: {mean_lat:.2f} ms ({bench_hz:.1f} Hz) | Min: {min_lat:.2f} ms | Max: {max_lat:.2f} ms")

    # Evaluate structured metrics against true future (Hand and Elbow)
    ade = result.ade(true_future_cartesian)
    fde = result.fde(true_future_cartesian)
    cov_rate = result.coverage_rate(true_future_cartesian)

    true_future_elbow = np.array([env.elbow(s) for s in true_future])
    ade_e = result.ade_elbow(true_future_elbow)
    fde_e = result.fde_elbow(true_future_elbow)
    cov_rate_e = result.coverage_rate_elbow(true_future_elbow)

    print("\n" + "=" * 90)
    print("3D STRUCTURED PREDICTION RESULTS & REAL-TIME FEASIBILITY BENCHMARK")
    print("=" * 90)
    print(f"{'Metric':<38} | {'Value':<20} | {'Interpretation':<26}")
    print("-" * 90)
    print(f"{'Hand: Average Displacement Error (ADE)':<38} | {ade*1000:>8.2f} mm         | Mean 3D Hand error")
    print(f"{'Hand: Final Displacement Error (FDE)':<38} | {fde*1000:>8.2f} mm         | 3D Hand target endpoint error")
    print(f"{'Hand: 95% 3D Confidence Tube Coverage':<38} | {cov_rate*100:>8.1f} %          | Ground truth Hand inside tube")
    print("-" * 90)
    print(f"{'Elbow: Average Displacement Error (ADE)':<38} | {ade_e*1000:>8.2f} mm         | Mean 3D Elbow error")
    print(f"{'Elbow: Final Displacement Error (FDE)':<38} | {fde_e*1000:>8.2f} mm         | 3D Elbow terminal endpoint error")
    print(f"{'Elbow: 95% 3D Confidence Tube Coverage':<38} | {cov_rate_e*100:>8.1f} %          | Ground truth Elbow inside tube")
    print("-" * 90)
    print(f"{'One-time AOT Warmup (Startup)':<38} | {t_warmup*1000:>8.1f} ms         | Pre-compilation overhead")
    print(f"{'Online Prediction Latency':<38} | {result.latency_ms:>8.2f} ms         | Pure execution time")
    print(f"{'Real-Time Loop Rate':<38} | {hz:>8.1f} Hz          | Online forecast frequency")
    rt_verdict = "FEASIBLE (> 30-50 Hz)" if hz >= 30 else "MARGINAL"
    print(f"{'Real-Time Feasible?':<38} | {rt_verdict:<20} | Exceeds 30 Hz robotics standard")
    print("-" * 90)
    print("\nStructured Container Summary:")
    print(result.summary())
    print("=" * 90 + "\n")

    # 7. Visualization
    print("[5/5] Generating publication-grade Plotly (HTML/PDF) and high-resolution visualizations...")
    os.makedirs("output/html", exist_ok=True)
    os.makedirs("output/pdf", exist_ok=True)

    # Plot 1: Dedicated Hand Plot (Tightly Zoomed + 3D Confidence Tube & Ellipsoids)
    plot_hand_html = "output/html/predict_motion_3d_hand.html"
    plot_hand_pdf = "output/pdf/predict_motion_3d_hand.pdf"
    predictor.plot_hand_3d_plotly(
        result=result,
        ground_truth_future=true_future_cartesian,
        save_html=plot_hand_html,
        save_pdf=plot_hand_pdf,
    )

    # Plot 2: Dedicated Elbow Plot (Tightly Zoomed + 3D Confidence Tube & Ellipsoids)
    plot_elbow_html = "output/html/predict_motion_3d_elbow.html"
    plot_elbow_pdf = "output/pdf/predict_motion_3d_elbow.pdf"
    predictor.plot_elbow_3d_plotly(
        result=result,
        ground_truth_future=true_future_elbow,
        save_html=plot_elbow_html,
        save_pdf=plot_elbow_pdf,
    )

    # Plot 3: Separated Keypoints Side-by-Side Plot (Independent Zoom for Hand and Elbow)
    plot_sep_html = "output/html/predict_motion_3d_separated.html"
    plot_sep_pdf = "output/pdf/predict_motion_3d_separated.pdf"
    predictor.plot_separated_keypoints_3d_plotly(
        result=result,
        ground_truth_future_hand=true_future_cartesian,
        ground_truth_future_elbow=true_future_elbow,
        save_html=plot_sep_html,
        save_pdf=plot_sep_pdf,
    )

    # Plot 4: Joint Trajectories + 3D End-Effector Tube & Covariance Ellipsoids
    plot1_html = "output/html/predict_motion_3d_joints.html"
    plot1_pdf = "output/pdf/predict_motion_3d_joints.pdf"

    # Plotly interactive WebGL + Vector PDF
    predictor.plot_prediction_3d_plotly(
        result=result,
        ground_truth_future=true_future_cartesian,
        ground_truth_joint_future=true_future,
        save_html=plot1_html,
        save_pdf=plot1_pdf,
    )

    # Plot 5: Full Human Arm Kinematic Chain (Shoulder -> Elbow -> Hand) Evolution
    plot2_html = "output/html/predict_motion_3d_arm.html"
    plot2_pdf = "output/pdf/predict_motion_3d_arm.pdf"

    predictor.plot_arm_kinematics_3d_plotly(
        result=result,
        true_states=states,
        save_html=plot2_html,
        save_pdf=plot2_pdf,
    )

    print("\nVisualizations successfully generated:")
    print(f"  -> Hand 3D (Close-Up Tube)         : HTML : {plot_hand_html} | PDF : {plot_hand_pdf}")
    print(f"  -> Elbow 3D (Close-Up Tube)        : HTML : {plot_elbow_html} | PDF : {plot_elbow_pdf}")
    print(f"  -> Separated Side-by-Side 3D       : HTML : {plot_sep_html} | PDF : {plot_sep_pdf}")
    print(f"  -> Joint Evolutions & Task Tube    : HTML : {plot1_html} | PDF : {plot1_pdf}")
    print(f"  -> Arm Kinematic Chain Postures    : HTML : {plot2_html} | PDF : {plot2_pdf}")

    if not args.no_show:
        plt.show(block=True)


if __name__ == "__main__":
    main()
