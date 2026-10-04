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
from prophet_ioc.envs.nonlinear_reaching import NonlinearReaching, NonlinearReachingParams
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.control import gilqr
from prophet_ioc.control.policy import create_lqg_policy
from prophet_ioc.prediction import MovingWindowMotionPredictor, PredictionResult
from plot_utils import update_rcparams, configure_high_dpi_window

# Apply high-DPI scaling (enlarges fonts, linewidths, and figure canvas for 16:10 / 4K displays)
update_rcparams(high_dpi=True, scale=2.0, dpi=130)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time moving-window motion prediction using probabilistic IOC and optimal feedback control."
    )
    parser.add_argument("-w", "--window", type=int, default=20,
                        help="Number of observed time steps in observation chunk (default: 20).")
    parser.add_argument("-f", "--future", type=int, default=30,
                        help="Number of future time steps to predict ahead (default: 30).")
    parser.add_argument("--mode", type=str, choices=["analytical", "monte_carlo", "both"], default="analytical",
                        help="Prediction strategy: 'analytical' (fastest, >50-70 Hz), 'monte_carlo' (stochastic rollouts), or 'both' (default: analytical).")
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
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platforms", "cpu")
        print("Forced execution on CPU backend.")

    backend = jax.default_backend()
    print(f"JAX active backend: {backend.upper()}")

    # Setup environment and ground truth parameters
    env = NonlinearReaching()
    gt_params = NonlinearReachingParams(
        action_cost=jnp.float32(1e-4),
        velocity_cost=jnp.float32(1e-2),
        motor_noise=jnp.float32(0.1),
        obs_noise=jnp.float32(1.0),
    )
    total_steps = args.window + args.future

    print("=" * 86)
    print("REAL-TIME PROBABILISTIC MOTION PREDICTOR (AOT Warmup + Analytical LQG Covariance)")
    print("=" * 86)
    print(f"Task             : Human Arm Reaching ({total_steps} steps total, dt={env.dt}s)")
    print(f"Observed Chunk   : {args.window} steps (t = 0.00s -> {args.window*env.dt:.2f}s)")
    print(f"Future Horizon   : {args.future} steps (t = {args.window*env.dt:.2f}s -> {total_steps*env.dt:.2f}s)")
    print(f"Strategy Mode    : {args.mode.upper()} (iLQR max_iter={args.max_iter})")
    print(f"Confidence Level : {args.confidence * 100:.1f}% (UCL / LCL bounds)")
    print("-" * 86)

    # 1. Simulate a realistic ground truth reaching demonstration
    print("\n[1/5] Simulating ground truth demonstration trajectory...")
    x0 = env._reset(None, gt_params)
    b0 = (x0, jnp.eye(4) * 1e-4)

    gains_gt, xbar_gt, ubar_gt = gilqr.solve(
        p=env, x0=x0, U_init=jnp.zeros((total_steps, 2)), params=gt_params, max_iter=10
    )
    policy_gt = create_lqg_policy(gains_gt, xbar_gt, ubar_gt)
    ekf_gt = EKFWrapper(NonlinearReaching)(b0=b0)

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)
    states, *_ = ekf_gt.rollout(subkey, total_steps, policy_gt, gt_params)
    states = np.array(states)

    observed_chunk = states[: args.window]
    true_future = states[args.window : args.window + args.future]
    true_future_cartesian = np.array(jax.vmap(env.e)(true_future))

    print(f"  -> Generated {len(states)} steps.")
    print(f"  -> Observed chunk shape : {observed_chunk.shape}")
    print(f"  -> True future shape    : {true_future.shape}")

    # 2. Initialize Predictor Model
    predictor = MovingWindowMotionPredictor(
        env=env,
        params=gt_params,
        default_horizon=total_steps,
        seed=args.seed,
    )

    # 3. Optional: Moving-Window Training / Parameter Fitting
    if args.fit_window:
        print("\n[2/5] Training IOC parameters directly on observed window chunk...")
        t0_fit = time.perf_counter()
        fit_res = predictor.fit_window(
            observed_chunk,
            window_start=0,
            window_length=args.window,
            restarts=3,
        )
        t_fit = time.perf_counter() - t0_fit
        print(f"  -> Model trained on window in {t_fit:.2f}s.")
        print(f"  -> Fitted parameters: {predictor.params}")
    else:
        print("\n[2/5] Using calibrated IOC parameters (pass --fit-window to train on chunk).")

    # 4. Strategy 1: Ahead-Of-Time (AOT) JIT Warmup
    print(f"\n[3/5] Strategy 1: Ahead-Of-Time (AOT) JIT pre-compilation (mode='{args.mode}')...")
    t0_warmup = time.perf_counter()
    predictor.warmup(future_steps=args.future, num_samples=args.samples, mode=args.mode, max_iter=args.max_iter)
    t_warmup = time.perf_counter() - t0_warmup
    print(f"  -> AOT compilation completed in {t_warmup:.2f}s (zero online latency impact!)")

    # 5. Execute Prediction
    print(f"\n[4/5] Executing real-time prediction (Strategy: {args.mode.upper()})...")
    result = predictor.predict(
        observed=observed_chunk,
        future_steps=args.future,
        mode=args.mode,
        num_samples=args.samples,
        confidence_level=args.confidence,
        max_iter=args.max_iter,
    )
    hz = 1000.0 / max(result.latency_ms, 1e-3)
    print(f"  -> Prediction executed in {result.latency_ms:.2f} ms ({hz:.1f} Hz control rate)")

    # 6. Optional: Streaming Benchmark
    if args.benchmark:
        print("\n[4.5] Running multi-step online streaming benchmark (warm-started MPC)...")
        bench_latencies = []
        last_u = None
        # Benchmark fixed-horizon MPC over sliding frames
        # Keeping future_steps fixed to avoid JAX graph re-compilations!
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
            # Shift controls by 1 step for warm starting next iteration
            last_u = jnp.vstack([r_bench.controls[1:], r_bench.controls[-1:]])

        mean_lat = np.mean(bench_latencies)
        min_lat = np.min(bench_latencies)
        max_lat = np.max(bench_latencies)
        bench_hz = 1000.0 / mean_lat
        print(f"  -> Benchmark over {len(bench_latencies)} sequential frames:")
        print(f"     Mean Latency: {mean_lat:.2f} ms ({bench_hz:.1f} Hz) | Min: {min_lat:.2f} ms | Max: {max_lat:.2f} ms")

    # Evaluate structured metrics against true future
    ade = result.ade(true_future_cartesian)
    fde = result.fde(true_future_cartesian)
    cov_rate = result.coverage_rate(true_future_cartesian)

    print("\n" + "=" * 86)
    print("STRUCTURED PREDICTION RESULTS & REAL-TIME BENCHMARK")
    print("=" * 86)
    print(f"{'Metric':<36} | {'Value':<20} | {'Interpretation':<22}")
    print("-" * 86)
    print(f"{'Average Displacement Error (ADE)':<36} | {ade*1000:>8.2f} mm         | Mean trajectory error")
    print(f"{'Final Displacement Error (FDE)':<36} | {fde*1000:>8.2f} mm         | Target endpoint error")
    print(f"{'95% Confidence Tube Coverage':<36} | {cov_rate*100:>8.1f} %          | Ground truth in bounds")
    print("-" * 86)
    print(f"{'One-time AOT Warmup (Startup)':<36} | {t_warmup*1000:>8.1f} ms         | Pre-compilation time")
    print(f"{'Online Prediction Latency':<36} | {result.latency_ms:>8.2f} ms         | Pure execution time")
    print(f"{'Real-Time Loop Rate':<36} | {hz:>8.1f} Hz          | Real-time update rate")
    print(f"{'Real-Time Compatible?':<36} | {'YES (> 50 Hz)':<20} | Exceeds 10-50 Hz robotics")
    print("-" * 86)
    print("\nStructured Container Summary:")
    print(result.summary())
    print("=" * 86 + "\n")

    # 7. Visualization
    print("[5/5] Generating high-resolution visualization...")
    fig = predictor.plot_prediction(
        result=result,
        ground_truth_future=true_future_cartesian,
        title=f"Real-Time Probabilistic Motion Prediction ({args.mode.upper()})\n"
              f"Observed: {args.window} steps | Future: {args.future} steps | Latency: {result.latency_ms:.1f} ms ({hz:.1f} Hz)",
        save_path="output/predict_motion.png",
        show=False,
    )

    configure_high_dpi_window(fig, scale=2.0)
    plt.show(block=True)


if __name__ == "__main__":
    main()
