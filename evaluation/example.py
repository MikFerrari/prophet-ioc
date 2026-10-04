import os
import sys
import time
import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

# Safe memory preallocation setting for GPU environments
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jax import jit, random, vmap, numpy as jnp

from prophet_ioc.envs import NonlinearReaching
from prophet_ioc.control import gilqr
from prophet_ioc.control.policy import create_lqg_policy
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.infer import FixedLinearizationInverseGILQG, FixedInverseMaxEntBaseline
from prophet_ioc.infer.utils import compute_mle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root (plot_utils)
os.chdir(Path(__file__).resolve().parents[1])  # outputs go to <repository root>/output
from plot_utils import update_rcparams, configure_high_dpi_window

# Apply high-DPI scaling (enlarges fonts, linewidths, and figure canvas for 16:10 / 4K displays)
update_rcparams(high_dpi=True, scale=2.0, dpi=130)


@partial(jit, static_argnames=("trials",))
def simulate_trajectories(key, params, trials=20):
    r"""Simulates stochastic arm reaching trajectories under optimal feedback control (LQG).

    The simulation pipeline proceeds in three steps:
    1. Forward Optimal Control (Generalized iLQR):
       Solves the discrete-time stochastic optimal control problem over horizon $T$:
           $$\min_{u_{0:T-1}} \mathbb{E}\left[ c_f(x_T) + \sum_{t=0}^{T-1} c(x_t, u_t) \right]$$
       taking into account signal-dependent (multiplicative) motor noise.
       This yields nominal trajectory $(\bar{x}, \bar{u})$ and time-varying Riccati feedback gains $L_t$.

    2. Closed-Loop Observer-Controller Policy (LQG):
       Constructs certainty-equivalence feedback policy:
           $$u_t = \pi(t, \hat{x}_t) = \bar{u}_t - L_t (\hat{x}_t - \bar{x}_t)$$
       where $\hat{x}_t$ is the internal belief estimated by an Extended Kalman Filter (EKF).

    3. Stochastic Rollout with Coupled Noise:
       Simulates simultaneous physical process noise $w_t \sim \mathcal{N}(0, \Sigma_w(u_t))$
       and sensory measurement noise $v_t \sim \mathcal{N}(0, \Sigma_v)$.
       End-effector Cartesian positions are computed via forward kinematics: $e(x_t)$.

    Args:
        key: JAX PRNGKey for stochastic noise generation.
        params: Physical, cost, and noise parameters $\theta = (c_a, c_v, \sigma_m, \sigma_o)$.
        trials: Number of stochastic rollout trials.

    Returns:
        xs: Simulated state trajectories of shape (trials, T+1, 4) [q1, q2, qdot1, qdot2].
        pos: Cartesian end-effector coordinates of shape (trials, T+1, 2) [e_x, e_y].
    """
    # 1. Solve the forward control problem via Generalized iLQR
    T = 50
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=jnp.zeros(shape=(T, env.action_shape[0])),
                                    params=params, max_iter=10)

    # 2. Instantiate closed-loop LQG feedback policy and EKF observer
    policy = create_lqg_policy(gains, xbar, ubar)
    ekf = EKFWrapper(NonlinearReaching)(b0=b0)

    # 3. Simulate stochastic rollouts under combined motor and sensory noise
    xs, *_ = ekf.simulate(key=key, steps=T, trials=trials, policy=policy, params=params)

    # 4. Map joint angle trajectories to Cartesian hand trajectories via forward kinematics
    pos = vmap(vmap(env.e))(xs)

    return xs, pos


def _convert_cli_args_for_hydra():
    new_argv = [sys.argv[0]]
    i = 1
    flag_map = {
        "-r": "data.restarts",
        "--restarts": "data.restarts",
        "-t": "data.trials",
        "--trials": "data.trials",
        "-s": "data.seed",
        "--seed": "data.seed",
    }
    bool_flags = {
        "--cpu": ("data.cpu", "true"),
    }
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("-h", "--help"):
            print("Hydra-enabled Reaching Simulation & Inverse Optimal Control.")
            print("\nUsage:")
            print("  python evaluation/example.py [Hydra overrides] [legacy flags]")
            print("\nExamples:")
            print("  python evaluation/example.py")
            print("  python evaluation/example.py data.trials=50 data.restarts=5")
            print("  python evaluation/example.py model.motor_noise=0.2")
            print("  python evaluation/example.py --trials 50 --restarts 5 --cpu")
            sys.exit(0)
        elif arg in bool_flags:
            key, val = bool_flags[arg]
            new_argv.append(f"{key}={val}")
            i += 1
        elif arg in flag_map:
            key = flag_map[arg]
            i += 1
            if i < len(sys.argv):
                val = sys.argv[i]
                new_argv.append(f"{key}={val}")
                i += 1
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv


@hydra.main(version_base=None, config_path="../config", config_name="example")
def main(cfg: DictConfig):
    print("=" * 80)
    print("Hydra Active Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80, flush=True)

    data_cfg = cfg.data
    model_cfg = cfg.model

    restarts = int(data_cfg.get("restarts", 10))
    trials = int(data_cfg.get("trials", 20))
    seed = int(data_cfg.get("seed", 1))
    cpu = bool(data_cfg.get("cpu", False))

    if cpu:
        jax.config.update("jax_platforms", "cpu")
        print("Forced execution on CPU backend.")

    backend = jax.default_backend()
    print(f"JAX active backend: {backend.upper()}")

    # Define true underlying biological motor control parameters from config
    NonlinearReachingParams = NonlinearReaching.get_params_type()
    params = NonlinearReachingParams(
        action_cost=jnp.float32(model_cfg.get("action_cost", 1e-4)),
        velocity_cost=jnp.float32(model_cfg.get("velocity_cost", 1e-2)),
        motor_noise=jnp.float32(model_cfg.get("motor_noise", 0.1)),
        obs_noise=jnp.float32(model_cfg.get("obs_noise", 1.0)),
    )
    print(f"Ground truth parameters: {params}")

    # Initialize environment, starting state (arm rest angle), and initial belief distribution
    global env, x0, b0
    env = NonlinearReaching()
    x0 = env._reset(None, params)
    b0 = (x0, jnp.eye(x0.shape[0]))
    start_pos = np.array(env.e(x0))

    # Setup random seed
    key = random.PRNGKey(seed)

    # Warm-up JIT compilation so timing benchmarks measure pure prediction latency
    print("Warming up JAX JIT compiler for trajectory simulation...")
    _ = simulate_trajectories(random.PRNGKey(0), params, trials=trials)
    _[1].block_until_ready()

    # =========================================================================
    # 1. Ground Truth Simulation & Timing
    # =========================================================================
    print(f"\n[1/3] Simulating {trials} ground truth trajectories...")
    key, subkey = random.split(key)
    t0_gt = time.perf_counter()
    xs, pos = simulate_trajectories(subkey, params, trials=trials)
    pos.block_until_ready()
    t_pred_gt = time.perf_counter() - t0_gt
    pos_gt = np.array(pos)
    ms_per_sample_gt = (t_pred_gt / trials) * 1000
    fps_gt = trials / t_pred_gt

    # =========================================================================
    # 2. Inverse iLQG (Our Method) Training & Timing
    # =========================================================================
    ioc = FixedLinearizationInverseGILQG(env, b0=b0)

    print(f"\n[2/3] Training Inverse ILQG (our method) with {restarts} restarts...")
    key, subkey = random.split(key)
    t0_ours = time.perf_counter()
    result = compute_mle(xs, ioc, subkey, restarts=restarts,
                         bounds=env.get_params_bounds(), optim="L-BFGS-B")
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result.params)
    t_train_ours = time.perf_counter() - t0_ours
    print(f"  -> Trained in {t_train_ours:.2f}s ({t_train_ours/restarts:.2f}s/restart):")
    print(f"     {result.params}")

    # Forward rollout: predict future trajectories using the recovered parameters \hat{\theta}
    key, subkey = random.split(key)
    t0_pred_ours = time.perf_counter()
    xs_sim, pos_sim = simulate_trajectories(subkey, result.params, trials=trials)
    pos_sim.block_until_ready()
    t_pred_ours = time.perf_counter() - t0_pred_ours
    pos_ours = np.array(pos_sim)
    ms_per_sample_ours = (t_pred_ours / trials) * 1000
    fps_ours = trials / t_pred_ours

    # =========================================================================
    # 3. MaxEnt IRL Baseline Training & Timing
    # =========================================================================
    baseline = FixedInverseMaxEntBaseline(env)
    print(f"\n[3/3] Training MaxEnt Baseline with {restarts} restarts...")
    key, subkey = random.split(key)
    t0_base = time.perf_counter()
    baseline_result = compute_mle(xs, baseline, subkey, restarts=restarts,
                                  bounds=env.get_params_bounds(), optim="L-BFGS-B")
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, baseline_result.params)
    t_train_base = time.perf_counter() - t0_base
    print(f"  -> Trained in {t_train_base:.2f}s ({t_train_base/restarts:.2f}s/restart):")
    print(f"     {baseline_result.params}")

    # Forward rollout: predict future trajectories using the baseline recovered parameters
    key, subkey = random.split(key)
    t0_pred_base = time.perf_counter()
    xs_baseline, pos_baseline = simulate_trajectories(subkey, baseline_result.params, trials=trials)
    pos_baseline.block_until_ready()
    t_pred_base = time.perf_counter() - t0_pred_base
    pos_baseline = np.array(pos_baseline)
    ms_per_sample_base = (t_pred_base / trials) * 1000
    fps_base = trials / t_pred_base

    # --- Print Benchmark Summary Table ---
    print("\n" + "=" * 82)
    print(f"ALGORITHM SPEED & TIMING BENCHMARK ({trials} predicted samples, {restarts} restarts, {backend.upper()})")
    print("=" * 82)
    print(f"{'Phase / Model':<28} | {'Total Time':<11} | {'Speed / Per Unit':<20} | {'Throughput':<14}")
    print("-" * 82)
    print(f"{'Training (Ours Inv-iLQG)':<28} | {t_train_ours:>8.2f} s  | {t_train_ours/restarts:>7.2f} s/restart     | {trials/t_train_ours:>6.2f} demos/s")
    print(f"{'Training (Baseline MaxEnt)':<28} | {t_train_base:>8.2f} s  | {t_train_base/restarts:>7.2f} s/restart     | {trials/t_train_base:>6.2f} demos/s")
    print("-" * 82)
    print(f"{'Prediction (Ground Truth)':<28} | {t_pred_gt*1000:>8.1f} ms | {ms_per_sample_gt:>7.2f} ms/sample     | {fps_gt:>6.1f} samples/s")
    print(f"{'Prediction (Ours Inv-iLQG)':<28} | {t_pred_ours*1000:>8.1f} ms | {ms_per_sample_ours:>7.2f} ms/sample     | {fps_ours:>6.1f} samples/s")
    print(f"{'Prediction (Baseline MaxEnt)':<28} | {t_pred_base*1000:>8.1f} ms | {ms_per_sample_base:>7.2f} ms/sample     | {fps_base:>6.1f} samples/s")
    print("=" * 82 + "\n")
    print("-" * 82)
    print(f"{'Prediction (Ground Truth)':<28} | {t_pred_gt*1000:>8.1f} ms | {ms_per_sample_gt:>7.2f} ms/sample     | {fps_gt:>6.1f} samples/s")
    print(f"{'Prediction (Ours Inv-iLQG)':<28} | {t_pred_ours*1000:>8.1f} ms | {ms_per_sample_ours:>7.2f} ms/sample     | {fps_ours:>6.1f} samples/s")
    print(f"{'Prediction (Baseline MaxEnt)':<28} | {t_pred_base*1000:>8.1f} ms | {ms_per_sample_base:>7.2f} ms/sample     | {fps_base:>6.1f} samples/s")
    print("=" * 82 + "\n")

    # --- Plotting: Split into 3 readable side-by-side subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True, sharey=True)

    target_pos = np.array(env.target)
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc")

    panels_data = [
        {
            "ax": axes[0],
            "title": "Ground Truth",
            "pos": pos_gt,
            "color": "C0",
            "mean_color": "#1f77b4",
            "label": "Ground truth",
            "params_text": (
                r"$\mathbf{Parameters:}$" + "\n"
                + rf"$c_a = {params.action_cost:.1e}$" + "\n"
                + rf"$c_v = {params.velocity_cost:.1e}$" + "\n"
                + rf"$\sigma_m = {params.motor_noise:.2f}$" + "\n"
                + rf"$\sigma_o = {params.obs_noise:.2f}$" + "\n"
                + r"$\mathbf{Speed:}$" + f" {ms_per_sample_gt:.1f} ms/sample"
            ),
        },
        {
            "ax": axes[1],
            "title": "Ours (Inverse iLQG)",
            "pos": pos_ours,
            "color": "C1",
            "mean_color": "#d95f02",
            "label": "Ours (MLE)",
            "params_text": (
                r"$\mathbf{Trained:}$" + "\n"
                + rf"$\hat{{c}}_a = {result.params.action_cost:.1e}$" + "\n"
                + rf"$\hat{{c}}_v = {result.params.velocity_cost:.1e}$" + "\n"
                + rf"$\hat{{\sigma}}_m = {result.params.motor_noise:.2f}$" + "\n"
                + rf"$\hat{{\sigma}}_o = {result.params.obs_noise:.2f}$" + "\n"
                + r"$\mathbf{Train:}$" + f" {t_train_ours:.1f}s" + "\n"
                + r"$\mathbf{Predict:}$" + f" {ms_per_sample_ours:.1f} ms/spl"
            ),
        },
        {
            "ax": axes[2],
            "title": "Baseline (MaxEnt IRL)",
            "pos": pos_baseline,
            "color": "C2",
            "mean_color": "#2ca02c",
            "label": "Baseline",
            "params_text": (
                r"$\mathbf{Trained:}$" + "\n"
                + rf"$\hat{{c}}_a = {baseline_result.params.action_cost:.1e}$" + "\n"
                + rf"$\hat{{c}}_v = {baseline_result.params.velocity_cost:.1e}$" + "\n"
                + rf"$\hat{{\sigma}}_m = {baseline_result.params.motor_noise:.2f}$" + "\n"
                + rf"$\hat{{\sigma}}_o = {baseline_result.params.obs_noise:.2f}$" + "\n"
                + r"$\mathbf{Train:}$" + f" {t_train_base:.1f}s" + "\n"
                + r"$\mathbf{Predict:}$" + f" {ms_per_sample_base:.1f} ms/spl"
            ),
        },
    ]

    mean_gt = pos_gt.mean(axis=0)

    for i, p in enumerate(panels_data):
        ax = p["ax"]
        pos_arr = p["pos"]

        # If not Ground Truth panel, show GT mean trajectory as subtle reference
        if i > 0:
            ax.plot(mean_gt[:, 0], mean_gt[:, 1], color="gray", linestyle="--",
                    linewidth=1.8, alpha=0.7, label="GT Mean ref", zorder=2)

        # Plot all simulated trajectory trials
        ax.plot(pos_arr[..., 0].T, pos_arr[..., 1].T, color=p["color"],
                alpha=0.35, linewidth=1.2, zorder=3)

        # Plot mean trajectory
        mean_traj = pos_arr.mean(axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=p["mean_color"],
                linewidth=2.8, label=p["label"] + " mean", zorder=4)

        # Plot start and target markers
        ax.scatter(start_pos[0], start_pos[1], color="black", s=65, zorder=6,
                   label="Start" if i == 0 else "")
        ax.scatter(target_pos[0], target_pos[1], color="crimson", marker="X", s=90,
                   linewidth=2, zorder=6, label="Target" if i == 0 else "")

        # Subplot title and parameter text box
        ax.set_title(p["title"], fontweight="bold", pad=10)
        ax.text(0.04, 0.96, p["params_text"], transform=ax.transAxes,
                verticalalignment="top", fontsize=9.0, bbox=bbox_props)

        ax.set_xlabel("x [m]")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="lower right", frameon=True, framealpha=0.85, fontsize=9)

    axes[0].set_ylabel("y [m]")
    fig.suptitle("Non-linear Reaching Task: Ground Truth vs. IOC Parameter Recovery",
                 fontweight="bold", y=0.98)
    fig.tight_layout()

    # Save high-resolution figure file in output directory
    os.makedirs("output", exist_ok=True)
    fig.savefig("output/example_trajectories.png", dpi=300)
    print("Saved figure to output/example_trajectories.png")

    # Configure high-DPI scaling for TkAgg toolbar and window chrome
    configure_high_dpi_window(fig, scale=2.0)

    plt.show(block=True)


if __name__ == '__main__':
    _convert_cli_args_for_hydra()
    main()

