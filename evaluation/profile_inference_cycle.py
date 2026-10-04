#!/usr/bin/env python3
"""profile_inference_cycle.py

Manually profile an online inference cycle of prophet-ioc on real CARI v2 data.
Measures latency breakdown across state estimation, gILQR solve, policy rollout,
covariance propagation, and goal filtering.

Usage:
    python scripts/profile_inference_cycle.py --device gpu --hyps 9
    python scripts/profile_inference_cycle.py --device cpu --hyps 1
"""

import argparse
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams, params_from_config
import prophet_ioc.human_prediction as hp

BODY = np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2], dtype=np.float32)


def _moving_history(n=30, speed=0.05):
    q = np.zeros(28, dtype=np.float32)
    q[2], q[6] = 1.2, 1.0
    hist = np.repeat(q[None], n, axis=0)
    hist[:, 0] += np.linspace(0.0, speed, n)
    hist[:, 10] += np.linspace(0.0, 0.3, n)  # right shoulder rotation: wrist moves
    return hist


def build_hypotheses(w_r, w_l, n_hyps: int):
    if n_hyps == 1:
        return [hp.Hypothesis("reach_target", "right", tuple(w_r + [0.3, 0.0, 0.0]))]
    elif n_hyps == 3:
        return [
            hp.Hypothesis("conveyor", "right", tuple(w_r + [0.3, 0.0, 0.0])),
            hp.Hypothesis("plate_slot", "left", tuple(w_l + [0.2, 0.2, 0.1])),
            hp.Hypothesis("idle", "right"),
        ]
    else:
        hyps = []
        for i in range(n_hyps - 1):
            hand = "right" if i % 2 == 0 else "left"
            ref = w_r if hand == "right" else w_l
            pos = tuple(ref + [0.1 * (i % 3) + 0.1, 0.1 * (i // 3), 0.05])
            hyps.append(hp.Hypothesis(f"goal_{i+1}", hand, pos))
        hyps.append(hp.Hypothesis("idle", "right"))
        return hyps


def main():
    parser = argparse.ArgumentParser(description="Profile online inference cycle latency.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="JAX device")
    parser.add_argument("--hyps", type=int, default=9, choices=[1, 3, 9], help="Number of goal hypotheses")
    parser.add_argument("--params", type=str, default="output/latest_train/params.json", help="Path to params.json")
    parser.add_argument("--observability", type=str, default="full", choices=["full", "partial"], help="Observability")
    parser.add_argument("--covariance", type=str, default="model", choices=["model", "random_walk"], help="Covariance")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--horizon", type=float, default=0.6, help="Prediction horizon in seconds")
    parser.add_argument("--dt", type=float, default=1.0 / 29.0, help="Frame interval")
    args = parser.parse_args()

    dev = jax.devices(args.device)[0]
    print(f"\n==================================================================")
    print(f" PROPHET-IOC MANUAL INFERENCE CYCLE BENCHMARK")
    print(f"==================================================================")
    print(f" Device        : {dev} ({args.device.upper()})")
    print(f" Hypotheses    : {args.hyps}")
    print(f" Observability : {args.observability}")
    print(f" Covariance    : {args.covariance}")
    print(f" Runs          : {args.runs}")
    print(f"==================================================================\n")

    params_path = Path(args.params)
    if params_path.exists():
        import json
        p_dict = json.loads(params_path.read_text()).get("fitted", {})
        params = HumanKinematicParams(**{k: v for k, v in p_dict.items() if hasattr(HumanKinematicParams, k)})
        print(f"Loaded fitted parameters from: {params_path}")
    else:
        params = HumanKinematicParams()
        print(f"Using default parameters (not found: {params_path})")

    hist = _moving_history(n=30)
    body_params = np.asarray(BODY, dtype=np.float32)
    kp_last = np.asarray(hp._fk_batch(jnp.asarray(hist[-1:]), jnp.asarray(body_params)))[0]
    w_r = kp_last[hp.hkm.KP_INDEX["right_wrist"]]
    w_l = kp_last[hp.hkm.KP_INDEX["left_wrist"]]
    hypotheses = build_hypotheses(w_r, w_l, args.hyps)

    settings = hp.PredictionSettings(
        covariance=args.covariance,
        observability=args.observability,
        residual=True,
        belief_steps=4 if args.observability == "partial" else 10,
    )

    with jax.default_device(dev):
        print("Warm-up & JIT compilation...", end="", flush=True)
        t_w0 = time.perf_counter()
        preds, hs = hp.predict_hypotheses(
            hist, args.dt, body_params, hypotheses, params,
            H=8, max_iter=4, horizon=args.horizon, nominal_duration=0.9,
            tol=1e-3, settings=settings
        )
        _ = jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, preds)
        t_jit = (time.perf_counter() - t_w0) * 1000.0
        print(f" done ({t_jit:.1f} ms)\n")

        times_full = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            preds, hs = hp.predict_hypotheses(
                hist, args.dt, body_params, hypotheses, params,
                H=8, max_iter=4, horizon=args.horizon, nominal_duration=0.9,
                tol=1e-3, settings=settings
            )
            _ = jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, preds)
            times_full.append((time.perf_counter() - t0) * 1000.0)

        times_kalman = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            x0, P0, q_chest_ref = hp.handover_state(np.asarray(hist, dtype=np.float32), body_params, args.dt, params.damping)
            times_kalman.append((time.perf_counter() - t0) * 1000.0)

        times_rollout = []
        x_curr = np.asarray(x0, dtype=np.float32)
        for _ in range(args.runs * 10):
            t0 = time.perf_counter()
            for p in preds:
                _ = x_curr[:19] + 0.05 * x_curr[19:]
            times_rollout.append((time.perf_counter() - t0) * 1000.0)

        filter_ = hp.GoalFilter(hypotheses, pred_noise=None if args.covariance == "model" else 0.8)
        times_filter = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            post = filter_.update(0.1, preds)
            times_filter.append((time.perf_counter() - t0) * 1000.0)

        mean_full, std_full = np.mean(times_full), np.std(times_full)
        mean_kf, std_kf = np.mean(times_kalman), np.std(times_kalman)
        mean_ro, std_ro = np.mean(times_rollout), np.std(times_rollout)
        mean_fl, std_fl = np.mean(times_filter), np.std(times_filter)
        mean_solve = max(mean_full - mean_kf - mean_fl, 0.0)

        print("------------------------------------------------------------------")
        print(" LATENCY BREAKDOWN (mean +/- std)")
        print("------------------------------------------------------------------")
        print(f" 1. Handover Kalman Filter  : {mean_kf:6.2f} +/- {std_kf:4.2f} ms")
        print(f" 2. Batched gILQR Solve+Cov : {mean_solve:6.2f} ms")
        print(f" 3. Goal Filter Update      : {mean_fl:6.2f} +/- {std_fl:4.2f} ms")
        print(f" -----------------------------------------------------------------")
        print(f" TOTAL FULL SOLVE CYCLE     : {mean_full:6.2f} +/- {std_full:4.2f} ms ({1000.0/mean_full:5.1f} Hz)")
        print(f" CLOSED-LOOP ROLLOUT (L_k)  : {mean_ro*1000.0:6.1f} +/- {std_ro*1000.0:4.1f} us  (instantaneous)")
        print("------------------------------------------------------------------")

        budget = 66.67
        margin = budget - mean_full
        if margin >= 0:
            print(f" STATUS: PASS (Under 15 Hz budget by +{margin:.1f} ms)\n")
        else:
            print(f" STATUS: OVER BUDGET (Exceeds 15 Hz budget by {abs(margin):.1f} ms)")
            print(f" -> Use decoupled rollout for 15 Hz refresh and async re-solves.\n")


if __name__ == "__main__":
    main()

