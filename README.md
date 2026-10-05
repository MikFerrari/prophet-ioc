# PROPHET: PRObabilistic Partially-observable Human Estimation Toolkit

**Uncertainty-Aware Human Reaching Prediction via Inverse Optimal Control on an Anthropomorphic Kinematic Model**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![JAX Accelerated](https://img.shields.io/badge/backend-JAX%20GPU%2FCPU-orange.svg)](https://github.com/google/jax)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-brightgreen.svg)](https://docs.ros.org/en/jazzy/)

**PROPHET** is an open-source Python and ROS 2 toolkit for real-time, probabilistic human motion prediction and goal inference. It unifies belief-space **Inverse Optimal Control (IOC)** under partial observability with a **28-DOF anthropomorphic human kinematic model** (19 active upper-body DOFs), strictly preserving rigid bone lengths and producing calibrated 95% analytical uncertainty covariance tubes.

The toolkit evaluates both offline on real pick-and-place trajectories from the **CARI v2 dataset** and online via an 8-hypothesis Bayesian goal filter, while benchmarking against synthetic motion diffusion from **NVIDIA Kimodo**.

---

## 1. Installation & Environment Setup

### 1.1 Virtual Environment
Create the virtual environment in `.venv` (named `prophet_venv`, the prompt shown when active; Python `>= 3.11`, tested with 3.14) and install the package in editable
mode with all its dependencies (declared in `pyproject.toml`; `cuda` adds the NVIDIA GPU build of JAX, `dev` pytest
and the tools of `human_kinematic_model`):

```bash
python3 -m venv --upgrade-deps --prompt prophet_venv .venv
source .venv/bin/activate
python -m pip install -e ".[cuda,dev]"     # CPU only: python -m pip install -e ".[dev]"
python -m pip show prophet-ioc hydra-core jax | grep -E "^(Name|Version)"   # check: three packages listed
```

Use `python -m pip`, not a bare `pip`: an alias, `~/.local/bin/pip` or conda can make `pip` belong to another
environment even with the venv active, and the packages then land there (`ModuleNotFoundError: No module named
'hydra'` at the first run). `cuda` installs `jax[cuda13]`, which needs NVIDIA driver >= 580 (`nvidia-smi`); with an
older driver (>= 525) install `".[dev]"` and then `python -m pip install -U "jax[cuda12]"`.

Do not rename or move the venv after creating it: its scripts (`pip`, `pytest`, `activate`) hard-code its path.
Recreate it instead.

### 1.2 Human Kinematic Model Dependency
The JAX-differentiable 28-DOF kinematic chain (`human_kinematic_model_jax`) is the pip package `human_model` of the
repository [human_kinematic_model](https://github.com/JRL-CARI-CNR-UNIBS/human_kinematic_model) (branch `jax`), a
dependency in `pyproject.toml`: the installation of 1.1 downloads and installs it (it compiles a small C++ binding,
so a C++ compiler is needed; see that repository's README, Installation A).

To develop the model alongside, with the clone in `../human_kinematic_model`, replace it by an editable install of the
clone (its python modules are then imported from the clone's `scripts/`, changes are live):

```bash
python -m pip install -e ../human_kinematic_model
```

### 1.3 Hardware Acceleration
JAX automatically leverages CUDA GPUs when available (e.g. NVIDIA RTX series). To force CPU execution:
```bash
export JAX_PLATFORMS=cpu
```
The scripts (and the ROS 2 node) set `OPENBLAS_NUM_THREADS=1` by default: JAX's CPU linear algebra (the small
Cholesky / LU factorizations of the belief-space prediction) calls OpenBLAS, whose thread pool made them up to 100x
slower on a loaded machine. Set it before Python starts when using the library directly.

---

## 2. Repository Layout

```text
prophet-ioc/
├── prophet_ioc/              # Core library: control, envs, inference, data, prediction
│   ├── control/              # LQR, LQG, gLQG, iLQR, gILQR, gILQG (partial observability)
│   ├── envs/                 # Human kinematic reaching (19/27 DOFs), 3D reaching, wrappers
│   ├── infer/                # Inverse iLQG, MaxEnt IOC, multi-trial likelihood, baselines
│   ├── baselines/            # Data-driven baselines learned from demonstrations: ProMP, DMP
│   ├── data/                 # CARI v2 dataset loader, trials, joint mappings
│   └── human_prediction.py   # Run-time prediction (predictive distribution), goal hypotheses, GoalFilter
├── train.py                  # CARI v2: IOC cost weight fitting (+ random-walk covariance calibration, ablation)
├── eval.py                   # CARI v2: Kinematic predictor vs baselines & figure generation
├── config/                   # Hydra configuration tree (data, model, infer, experiment)
├── plot_utils/               # High-DPI publication plots, 3D Plotly skeleton & tube renderers
├── evaluation/               # Evaluation experiments, comparisons, and synthetic pipelines
│   ├── example.py            # Simulated reaching demo (Inverse gILQG vs MaxEnt)
│   ├── goal_inference.py     # Online multi-hypothesis goal inference on full CARI sessions
│   ├── kimodo_reaches.py     # NVIDIA Kimodo diffusion synthesis & real-vs-synthetic validation
│   ├── export_replay.py      # CARI session export (.npz, .yaml) for ROS 2 replay
│   └── save_results.py       # Standalone table and publication HTML generator
├── ros2/                     # ROS 2 Jazzy predictor node & custom message definitions
├── docker/                   # ROS 2 Docker container & compose setup
├── tests/                    # Unit and regression test suite
└── output/                   # Auto-generated experiment outputs & latest symlinks
```

---

## 3. Core Workflows & Experiments

All scripts log to dedicated timestamped directories under `output/`, and maintain symbolic links (`output/latest_train`, `output/latest`, `output/latest_goal_inference`) pointing to the most recent run.

### 3.1 CARI v2 IOC Fitting & Held-out Evaluation (`train.py`, `eval.py`)

The cost weights are learned from demonstrations: `train.py` fits them by IOC (maximum likelihood of the probabilistic
model, with the agent's policy solved for each training window) on the complete reaches of all subjects but the
held-out one (`data.test_subjects`, default `sub_13`), all instructions, FAST; `eval.py` predicts the
held-out subject's reaches from 10 / 30 / 50 / 70 % observed with the fitted weights (and, for reference, with the
initial weights of `config/model/human_kinematic.yaml`, which are only the starting point of the fit).

```bash
source .venv/bin/activate
python train.py          # IOC fit (+ random-walk calibration) -> output/train_<ts>/ (params.json, figures/)
python train.py ioc.observability=partial      # partially observed (belief) model
python eval.py           # held-out subject, fitted weights of output/latest_train -> output/eval_<ts>/ (figures/)
```

| Parameter | Default | Description |
|---|---|---|
| `data.subjects` | 10 subjects | CARI v2 subjects |
| `data.test_subjects` | `[sub_13]` | Held out: never used by `train.py`, evaluated by `eval.py` |
| `data.instructions` | `[0 ... 8]` | All instructions (1, 3, 5 objects; 7 robot, both hands; 0, 2, 4, 6, 8 hands home) |
| `data.obs_ratios` | `[0.2, 0.5, 0.7]` | Observed fractions of the reach: the handover points evaluated by `eval.py` and the starts of the training windows of `train.py` |
| `data.state_estimator` | `rts` | Joint states of the training windows: `rts` (RTS smoother under the model's ZOH dynamics, noise `data.rts_accel_noise`, `data.rts_obs_noise`) or `savgol` (Savitzky-Golay) |
| `ioc.objective` | `likelihood` | `likelihood` (negative log-likelihood of the probabilistic IOC model), `open_loop` (open-loop keypoint error, ablation), `none` (initial weights) |
| `ioc.observability` | `full` | `full` (the agent knows its state, paper App. E) or `partial` (EKF belief, paper Algorithm 1; `obs_noise` is also fitted; `full` is its `obs_noise -> 0` limit) |
| `ioc.linearization` | `solve` | `solve` (policy of the model's optimal trajectory from each window start, differentiable unrolled gILQR) or `data` (ablation, paper sec. 3.3: linearized along the data) |
| `ioc.temperature` | `1e-6` | Fixed temperature of the max-ent policy `u ~ N(ubar + l + L (x - xbar), temperature H^-1)`; `0` = deterministic |
| `ioc.likelihood_block` | `full` | Scored state components: `full` or `velocity` (fallback) |
| `ioc.solve_iters`, `ioc.checkpoint` | `8`, `false` | Unrolled solver iterations per window; `jax.checkpoint` of each iteration (less reverse-mode memory) |
| `ioc.restarts` | `4` | Parallel projected-Adam restarts in log10 space (restart 0 from the initial weights) |
| `eval.params` | `output/latest_train` | Fitted weights (and the observability / temperature of their fit, calibrated random-walk noise); `null` = initial weights |
| `eval.compare_initial` | `true` | Also evaluate the initial weights (method `kin_init`) |

The model (`config/model/human_kinematic.yaml`, read by `params_from_config`; the fixed values are saved with the
fitted weights in `params.json`):

| Parameter | Default | Description |
|---|---|---|
| `damping` | `0.0` | `b` of `qdd = u - b qd + noise`, exact zero-order-hold discretization (model, handover filter, prediction covariance, RTS smoother) |
| `motor_noise` | `0.1` | Signal-dependent motor noise intensity: per joint covariance `sigma_m^2 u^2 [[dt^3/3, dt^2/2], [dt^2/2, dt]]` over `(q, qd)` |
| `velocity_cost`, `running_vel_cost`, `base_disp_cost`, `running_target_cost`, `w_act_*` | | Learnable cost weights (`HumanKinematicParams.LEARNABLE`); the terminal wrist-to-target weight is 1 (cost scale; the values are the former ones / 100) |
| `pelvis_displacement_cost`, `running_target` | `true` | Switch the pelvis displacement term (e.g. off for walking tasks) / learn `running_target_cost` |
| `joint_limit_cost`, `w_lim`, `chest_rot_limit` | `true`, `1.0`, `1.0` | Hand-tuned joint-limit penalty (anatomical limits of `human_kinematic_model`; bound on the chest rotation-vector norm) |
| `velocity_floor` | `1e-6` | Lower bound of the running joint-velocity weight |
| `max_iter`, `early_stopping`, `tol` | `3`, `true`, `1e-3` | Prediction solver: at most `max_iter` iterations, stop when the relative cost improvement is below `tol` |
| `prediction_covariance` | `model` | Predictive covariance (3.1.3): `model` (the model's own) or `random_walk` (calibrated joint-velocity random walk, ablation; `pred_noise`) |
| `prediction_residual` | `true` | Model covariance: also the likelihood-only residual noise of the fit (`residual_noise`) |
| `prediction_observability`, `belief_steps` | `auto`, `10` | `auto`: predict with the observability of the fit (`partial` = belief-space prediction), or force `full` / `partial`; prediction steps of the history over which the agent's belief is tracked |
| `pred_noise` | `0.8` | `random_walk` only: noise level with the initial weights (`train.py` calibrates it for the fitted ones) |
| `eval.baselines` | `[promp, dmp]` | Data-driven baselines learned from the training subjects' reaches (3.1.1); `[]` = none |
| `eval.baseline_options` | see `config/config.yaml` | Fit options of each data-driven baseline |

Figures: `output/train_<ts>/figures/` (convergence of the restarts, initial vs fitted weights within the bounds,
open-loop error of the training reaches before / after the fit by handover point and instruction, example
predictions), `output/eval_<ts>/figures/` (errors vs observed fraction, per instruction, along the prediction).

#### 3.1.1 Data-driven baselines: ProMP and DMP (`prophet_ioc/baselines/`)

Besides the analytic baselines (Cartesian point mass, min-jerk, goal-directed and plain constant velocity), `eval.py`
compares two movement-primitive baselines **learned from demonstrations**: they are fitted once, before the
evaluation, on the complete reaches (onset to end) of the training subjects only (`data.subjects` minus
`data.test_subjects`, the reaches `train.py` uses; the held-out subject is never seen), and predict like the other
baselines from the same inputs: the observed prefix of the 9 upper-body keypoints (FK of the filtered IK angles), the
reaching-wrist target only, and the arrival time of `minjerk` / `gcv` (on CARI v2 the end of the reach).

- **Common representation** (`common.py`): Cartesian keypoints, the space where every baseline predicts and is scored
  (and where the wrist goal is a linear constraint; in joint space it would need the forward kinematics). Each reach
  is expressed in a frame fixed at its onset: displacements of every keypoint from its onset position, rotated to the
  body heading (yaw from the shoulders, z up); left-hand reaches are mirrored and their sides swapped, so left and right
  reaches of all subjects, cells and instructions are pooled (81 training reaches).
- **`promp`**: ProMP (Paraschos et al., NeurIPS 2013; for HRC prediction, Maeda et al., Auton. Robots 2017). Reaches
  are normalized to a phase z in [0, 1]; 12 Gaussian basis functions per dimension; one Gaussian weight distribution
  over the 27 dimensions (all joints and phases coupled), pooled over instructions with the wrist goal as a via-point
  at z = 1 (per instruction there would be only 9 demonstrations, with goal locations that differ between cells). The
  phase of the prefix is z = t / T with T the arrival time (option `phase: ml` estimates T by maximum likelihood of the
  prefix instead). Prediction: Gaussian conditioning on 10 frames of the prefix and on the wrist goal, mean and
  covariance on the prediction grid; the covariance (not calibrated, unlike the kinematic model's) gives the 95 %
  coverage. Options: `n_basis`, `width`, `cov_reg` (diagonal ridge of the weight covariance relative to its mean
  variance, 0.03, chosen on held-out *training* subjects), `n_cond`, `sigma_goal`, `phase`.
- **`dmp`**: discrete DMPs (Ijspeert et al., Neural Computation 2013), one per dimension with a shared canonical
  system and the amplitude scaling (g - y0) f(s). The forcing shape of each dimension is the amplitude-weighted average
  of the training demonstrations' forcing terms in the canonical phase (30 basis functions). The DMP of the whole reach
  (tau = arrival time, y0 = observed onset pose) is integrated from the handover: phase at t_obs, observed position and
  Savitzky-Golay velocity. Goal: the target for the reaching wrist; for the other dimensions the end displacement is
  learned (`goal_model: regression`, affine in the wrist goal; or `mean`) and, by default (`infer_goal: true`),
  refined from the observed prefix (the DMP equation is linear in g; Gaussian combination with the learned prior).
  Deterministic: no covariance. Options: `n_basis`, `alpha_z`, `goal_model`, `infer_goal`, `n_cond`, `phase`
  (`continue`, or `restart` the DMP at the handover with tau = remaining time), `dt_int`.

Assumptions shared by both: the arrival time is known (as for `minjerk` / `gcv`), the reaching hand is the dataset's;
the non-wrist goals are only what the training reaches suggest, so these joints are less accurate than the wrist.
Tests: `python -m pytest tests/test_baselines.py` (synthetic CARI-like reaches, CPU, seconds).

#### 3.1.2 Monitoring runs with wandb

Run `train.py` directly from the terminal or in the background:

```bash
python train.py ioc.observability=full
```

`train.py` and `eval.py` log to Weights & Biases (`config.yaml` `wandb:`; `pip install -e ".[tracking]"`, then
`wandb login` once): per fit iteration the loss and gradient norm of every restart, the best loss, the weights of the
best point and of every restart (log10) and the iteration time; then the calibrated prediction noise, the training
errors and the figures; `eval.py` the summary metrics, a results table and the figures. Without a login the runs are
written offline to `output/wandb/` (`wandb sync output/wandb/offline-run-*` uploads them); `wandb.enabled=false`
turns tracking off. `wandb.group` groups a training and its evaluation.

#### 3.1.3 Run-time prediction and its uncertainty (`prophet_ioc/human_prediction.py`)

From the observed prefix (IK joint angles), a Kalman filter under the model's ZOH dynamics gives the handover estimate
`x0_hat` and its covariance `P0` (the observer's uncertainty). The agent's policy (gains `L`, nominal) is solved to the
goal over the expected arrival time, and the predictive distribution is the model's own (`prediction_covariance:
model`):

- **fully observed**: mean = the nominal from `x0_hat`; covariance `Sigma_{k+1} = F_k Sigma_k F_k^T + E[V_k V_k^T] +
  W_k W_k^T`, `Sigma_0 = P0`, with the closed-loop linearization `F = A + B L`, the signal-dependent motor noise
  (ZOH, averaged over the commands the closed loop issues: `sigma_m^2 E[u u^T]`, exact for the linear dynamics) and
  the max-ent decision noise `W = B Gamma` (temperature of the fit). By default the likelihood-only residual noise
  of the fit is added (`prediction_residual`): the fit explains the recorded motion by the model's noises plus that
  residual (the controller's own motor noise is fixed), so the predictive distribution consistent with the fitted
  likelihood contains it; it is fed back by the policy like any disturbance, and does not change the policy.
- **partially observed** (weights fitted with `ioc.observability=partial`): the agent planned from the start of a
  window of the history (`belief_steps` prediction steps) and its belief is tracked over the estimated history
  states with Algorithm 1 (the filtered-form EKF of the likelihood, conditioned on every state); the agent then plans
  from its belief mean, and the joint Gaussian of state and belief is propagated through the joint dynamics *without
  conditioning* (nothing is observed in the future). The fully observed prediction is its `obs_noise -> 0` limit.

Keypoint covariances follow through the FK Jacobians, `J Sigma J^T`. The former calibrated random walk
(`prediction_covariance: random_walk`, `cov_init + pred_noise^2 cov_unit`) is kept as an ablation; `train.py` still
calibrates its level and reports the coverage of the model covariance on the training subjects.
Latency on the CPU (H = 14, single-threaded OpenBLAS): one reach ~58 ms with the model covariance (51 ms with the
former random walk), ~100 ms partially observed; online, 9 goal hypotheses of a CARI cell, ~190 ms (176 ms), ~460 ms
partially observed.

---

### 3.2 Online Prediction on Full Sessions: Goal Inferred or Known (`evaluation/goal_inference.py`)

`goal_inference.py` replays continuous CARI sessions (8 movements across 7 goal locations + idle) as the ROS 2 node
sees them, with the IOC-fitted weights. Two switches in `config/config.yaml` (`online`):

| Parameter | Values | Description |
|---|---|---|
| `online.goal_mode` | `inferred` / `known` | Goal inferred among the goals of the cell (Bayesian filter), or given by the task schedule (the filter then only finds hand and onset) |
| `online.uncertainty` | `mixture` / `map` | Published covariance with the goal uncertainty (posterior-weighted spread of the hypotheses around the published prediction) or of the most probable hypothesis only |
| `online.grasp_offset` | `0.0` m | Goal locations given as object positions: the wrist goal is this far before the object, on the line from the current wrist (`human_prediction.wrist_goal`); 0 = the location is the wrist goal (CARI v2: wrist positions at the end of the reaches). ROS 2 node: `grasp_offset` |

```bash
python evaluation/goal_inference.py                                     # inferred goal, goal-aware covariance
python evaluation/goal_inference.py online.goal_mode=known online.uncertainty=map
python evaluation/goal_inference.py online.reuse=output/latest_goal_inference   # tables / figures only
```

Both goal modes, the oracle (goal and onset known), constant velocity and a frozen pose are always reported, with
the wrist coverage of both covariances. The filter settings are selected on the training subjects; results are
reported on the held-out subject. Figures: goal accuracy over the movement, a session timeline (posterior over the
hypotheses, goal entropy, 95 % radius of the prediction with and without the goal uncertainty) and top-view
snapshots of the goal uncertainty during each reach. The same switches exist in the ROS 2 node (`goal_mode`,
`uncertainty` in `ros2/human_motion_predictor/config/predictor.yaml`).

---

### 3.3 Synthetic Diffusion Reaches with NVIDIA Kimodo (`evaluation/kimodo_reaches.py`, on hold)

> On hold: `kimodo_reaches.py compare` still reads the previous `goal_inference.py` results format.

To evaluate whether motion diffusion models can substitute for expensive optical mocap, the toolkit benchmarks synthetic clips generated with **Kimodo**:

```bash
cd evaluation

# 1. Export workspace layouts from CARI cells
python kimodo_reaches.py layout

# 2. Generate 200 synthetic reaching clips via Kimodo diffusion (10 subjects x 4 tasks x 5 runs)
../../kimodo/.venv/bin/python kimodo_reaches.py generate

# 3. Test online goal inference on clean synthetic keypoints
python goal_inference.py online.source=kimodo online.clips=output/kimodo/clips_text.npz

# 4. Test online goal inference with simulated ZED depth sensor noise
python goal_inference.py online.source=kimodo online.clips=output/kimodo/clips_text.npz online.noise=true

# 5. Generate side-by-side real vs synthetic comparison report
python kimodo_reaches.py compare output/latest_goal_inference \
    output/latest_goal_inference_kimodo \
    output/latest_goal_inference_kimodo_noise
cd ..
```

---

### 3.4 Dedicated Evaluation & Latency Demos (`evaluation/`)

```bash
cd evaluation

# 1. NeurIPS paper reaching demo: simulated reach, Inverse gILQG vs MaxEnt
python example.py -r 50 -t 50 --cpu

# 2. Re-export publication tables from a saved run JSON:
python save_results.py ../output/latest/obs30/results.json

# 3. Export session data for ROS 2 replay:
python export_replay.py --subject sub_4
cd ..
```

---

## 4. ROS 2 Deployment & Docker

The motion predictor is packaged as a standard ROS 2 Jazzy node (`ros2/`), subscribing to keypoint streams and publishing probabilistic joint and Cartesian trajectories with covariance metadata.

To launch the containerized node alongside the camera wrapper:

```bash
docker compose -f docker/compose.yaml up --build
```
See **[docker/README.md](docker/README.md)** for architecture, topics, and replay instructions.

---

## 5. Outputs & Generated Artifacts

Every run writes structured, self-contained artifacts to `output/`:
- `summary.html` & `summary.csv`: Aggregated performance across all observed fractions (MPJPE, Wrist ADE/FDE, Elbow ADE/FDE, bone stretch %, 95% coverage, frequency in Hz).
- `results.json`: Full serialized numerical trajectories, covariances, and parameter checkpoints.
- `html/`: Interactive 3D Plotly visualizations (skeleton motion, reaching uncertainty tubes, comparative baselines).
- `frames/<trial>/skeleton/`: High-resolution PNG frames and 4x slow-motion GIFs (observed green $\to$ predicted red vs ground truth blue).

---

## 6. Running Tests

Run the complete test suite (unit and integration tests, on the CPU with `JAX_PLATFORMS=cpu`):

```bash
./.venv/bin/python -m pytest
```

---

## 7. Package Architecture (`prophet_ioc`)

- **`prophet_ioc.control`**: Optimal control solvers (`lqr`, `lqg`, `glqg`, `ilqr`, `gilqr`, `gilqg`, `ilqg_fixed`), and `ilqr_unrolled` (gILQR without jaxopt: fixed iterations, differentiable, for the IOC fit; early stopping for the online prediction).
- **`prophet_ioc.envs`**: Task environments (`human_kinematic_reaching.py`, `cartesian_reaching.py`, `nonlinear_reaching_3d.py`, `navigation.py`); `zoh.py`: exact zero-order-hold discretization of the damped double integrator shared by the model, the handover filter, the prediction covariance and the RTS smoother.
- **`prophet_ioc.envs.wrappers`**: Belief-state wrappers (`FullyObservedWrapper`, `EKFWrapper`).
- **`prophet_ioc.infer`**: Parameter inference (`inv_ilqg.py`: partially observed likelihood, Algorithm 1; `inv_ilqr.py`: fully observed; `multi_env.py`: per-window likelihoods with the policy solved per window, fully / partially observed; `inv_maxent.py`) and baselines (`constant_velocity.py`, `goal_directed_cv.py`, `minimum_jerk.py`, `cartesian_baseline.py`).
- **`prophet_ioc.baselines`**: Data-driven baselines learned from demonstrations (`promp.py`, `dmp.py`, `common.py`; section 3.1.1).
- **`prophet_ioc.human_prediction`**: Run-time prediction (3.1.3: handover filter, policy, predictive distribution of the fully / partially observed model, `PredictionSettings`), goal hypotheses (`predict_hypotheses`, `wrist_goal`) and `GoalFilter`. The joint-dynamics / belief code is shared with the likelihood (`infer.inv_ilqg.belief_filter`, `joint_predictive_moments`, `infer.inv_ilqr.closed_loop_moments`, `infer.multi_env.prefix_belief`).
- **`prophet_ioc.data`**: Dataset parsing, kinematic joint mapping, trial segmentation and training-state estimation (RTS smoother / Savitzky-Golay) (`cari.py`).

---

## 8. Acknowledgements & Attribution

This toolkit builds upon and integrates several open-source software libraries and datasets:

- **[nioc-neurips](https://github.com/RothkopfLab/nioc-neurips)**: Our core inverse optimal control algorithms and belief-tracking iLQG controllers are built upon and extended from the NeurIPS 2023 codebase by Dominik Straub, Matthias Schultheis, Heinz Koeppl, and Constantin A. Rothkopf ([Paper](https://arxiv.org/abs/2303.16698)).
- **[human_kinematic_model](https://github.com/JRL-CARI-CNR-UNIBS/human_kinematic_model)**: Provides the 28-DOF anthropomorphic human kinematic model, forward kinematics, and JAX-differentiable Jacobians developed by JRL-CARI-CNR-UNIBS.
- **[kimodo](https://github.com/nv-tlabs/kimodo)**: NVIDIA T-Labs library used for kinematic motion diffusion and synthetic reaching analysis.
- **[CARI v2 Dataset](https://drive.google.com/drive/folders/1ytaC6sb4ZdSuqsTpfRiYRbUrc54kMPvQ?usp=sharing)**: Optical motion capture dataset of human pick-and-place reaching movements (*access requires requesting permissions via Google Drive*).

### Citation

If you use this work in your research, please cite our paper as well as the foundational works:

```bibtex
@article{prophet_ioc2026,
  title={Uncertainty-Aware Human Reaching Prediction via Inverse Optimal Control on a Kinematic Model},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2026}
}

@inproceedings{straub2023nioc,
  title={Probabilistic inverse optimal control for non-linear partially observable systems disentangles perceptual uncertainty and behavioral costs},
  author={Straub, Dominik and Schultheis, Matthias and Koeppl, Heinz and Rothkopf, Constantin A},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023}
}
```
