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
pip install -e ".[cuda,dev]"     # CPU only: pip install -e ".[dev]"
```

Do not rename or move the venv after creating it: its scripts (`pip`, `pytest`, `activate`) hard-code its path.
Recreate it instead.

### 1.2 Human Kinematic Model Dependency
The JAX-differentiable 28-DOF kinematic chain (`human_kinematic_model_jax`) is a script of the sibling repository
`human_kinematic_model`, not a pip package: put its `scripts/` folder on the venv's path with a `.pth` file:

```bash
realpath ../human_kinematic_model/scripts \
    > "$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/human_kinematic_model.pth"
```

### 1.3 Hardware Acceleration
JAX automatically leverages CUDA GPUs when available (e.g. NVIDIA RTX series). To force CPU execution:
```bash
export JAX_PLATFORMS=cpu
```

---

## 2. Repository Layout

```text
prophet-ioc/
├── prophet_ioc/              # Core library: control, envs, inference, data, prediction
│   ├── control/              # LQR, LQG, gLQG, iLQR, gILQR, gILQG (partial observability)
│   ├── envs/                 # Human kinematic reaching (19/27 DOFs), 3D reaching, wrappers
│   ├── infer/                # Inverse iLQG, MaxEnt IOC, multi-trial likelihood, baselines
│   ├── data/                 # CARI v2 dataset loader, trials, joint mappings
│   └── human_prediction.py   # Moving-window predictor, goal hypotheses, GoalFilter
├── train.py                  # CARI v2: IOC cost weight fitting & uncertainty calibration
├── eval.py                   # CARI v2: Kinematic predictor vs baselines & figure generation
├── config/                   # Hydra configuration tree (data, model, infer, experiment)
├── plot_utils/               # High-DPI publication plots, 3D Plotly skeleton & tube renderers
├── evaluation/               # Evaluation experiments, comparisons, and synthetic pipelines
│   ├── example.py            # Simulated reaching demo (Inverse gILQG vs MaxEnt)
│   ├── goal_inference.py     # Online multi-hypothesis goal inference on full CARI sessions
│   ├── kimodo_reaches.py     # NVIDIA Kimodo diffusion synthesis & real-vs-synthetic validation
│   ├── compare_spaces.py     # Kinematic joint-space vs Cartesian point-mass prediction
│   ├── predict_motion*.py    # Real-time latency profiling (2D, 3D, and 28-DOF human model)
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

The cost weights are learned from demonstrations: `train.py` fits them by IOC on the complete reaches of all subjects
but the held-out one (`data.test_subjects`, default `sub_13`), all instructions, FAST; `eval.py` predicts the
held-out subject's reaches from 10 / 30 / 50 / 70 % observed with the fitted weights (and, for reference, with the
initial weights of `config/model/human_kinematic.yaml`, which are only the starting point of the fit).

```bash
source .venv/bin/activate
python train.py          # IOC fit + uncertainty calibration -> output/train_<ts>/ (params.json, figures/)
python eval.py           # held-out subject, fitted weights of output/latest_train -> output/eval_<ts>/ (figures/)
```

| Parameter | Default | Description |
|---|---|---|
| `data.subjects` | 10 subjects | CARI v2 subjects |
| `data.test_subjects` | `[sub_13]` | Held out: never used by `train.py`, evaluated by `eval.py` |
| `data.instructions` | `[0 ... 8]` | All instructions (1, 3, 5 objects; 7 robot, both hands; 0, 2, 4, 6, 8 hands home) |
| `data.obs_ratios` | `[0.1, 0.3, 0.5, 0.7]` | Observed fractions of the reach evaluated by `eval.py` |
| `ioc.objective` | `open_loop` | `open_loop` (open-loop keypoint error), `likelihood` (one-step gILQR), `none` (initial weights) |
| `ioc.segment_starts` | `[0.1, 0.3, 0.5, 0.7]` | Training segments: from these handover points to the end of each demonstration |
| `ioc.restarts` | `4` | Parallel projected-Adam restarts in log10 space (restart 0 from the initial weights) |
| `eval.params` | `output/latest_train` | Fitted weights and calibrated noise; `null` = initial weights |
| `eval.compare_initial` | `true` | Also evaluate the initial weights (method `kin_init`) |

Figures: `output/train_<ts>/figures/` (convergence of the restarts, initial vs fitted weights within the bounds,
open-loop error of the training reaches before / after the fit by handover point and instruction, example
predictions), `output/eval_<ts>/figures/` (errors vs observed fraction, per instruction, along the prediction).

---

### 3.2 Online Prediction on Full Sessions: Goal Inferred or Known (`evaluation/goal_inference.py`)

`goal_inference.py` replays continuous CARI sessions (8 movements across 7 goal locations + idle) as the ROS 2 node
sees them, with the IOC-fitted weights. Two switches in `config/config.yaml` (`online`):

| Parameter | Values | Description |
|---|---|---|
| `online.goal_mode` | `inferred` / `known` | Goal inferred among the goals of the cell (Bayesian filter), or given by the task schedule (the filter then only finds hand and onset) |
| `online.uncertainty` | `mixture` / `map` | Published covariance with the goal uncertainty (posterior-weighted spread of the hypotheses around the published prediction) or of the most probable hypothesis only |

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

# 2. Real-time moving-window forecasting rate:
python predict_motion.py --benchmark        # 2-link planar arm
python predict_motion_3d.py                 # 3-link 3D arm
python predict_motion_hkm.py                # 28-DOF anthropomorphic human model

# 3. Space comparison: Joint-space kinematics vs Cartesian point mass
python compare_spaces.py

# 4. Re-export publication tables from a saved run JSON:
python save_results.py ../output/latest/obs30/results.json

# 5. Export session data for ROS 2 replay:
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

Run the complete test suite (35 unit and integration tests):

```bash
./.venv/bin/python -m pytest
```

---

## 7. Package Architecture (`prophet_ioc`)

- **`prophet_ioc.control`**: Optimal control solvers (`lqr`, `lqg`, `glqg`, `ilqr`, `gilqr`, `gilqg`, `ilqg_fixed`).
- **`prophet_ioc.envs`**: Task environments (`human_kinematic_reaching.py`, `cartesian_reaching.py`, `nonlinear_reaching_3d.py`, `navigation.py`).
- **`prophet_ioc.envs.wrappers`**: Belief-state wrappers (`FullyObservedWrapper`, `EKFWrapper`).
- **`prophet_ioc.infer`**: Parameter inference (`inv_ilqg.py`, `inv_ilqr.py`, `inv_maxent.py`, `multi_env.py`) and baselines (`constant_velocity.py`, `goal_directed_cv.py`, `minimum_jerk.py`, `cartesian_baseline.py`).
- **`prophet_ioc.human_prediction`**: Real-time motion forecasting, multi-hypothesis generation, and `GoalFilter`.
- **`prophet_ioc.data`**: Dataset parsing, kinematic joint mapping, and trial segmentation (`cari.py`).

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
