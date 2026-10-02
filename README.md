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
Create a fresh Python virtual environment (`>= 3.10`, tested up to Python 3.14) and install the package with dependencies:

```bash
python -m venv env
source env/bin/activate
pip install -e .
pip install hydra-core omegaconf pandas scipy plotly pytest
```

### 1.2 Human Kinematic Model Dependency
The JAX-differentiable 28-DOF kinematic chain is provided by the sibling repository `human_kinematic_model`:

```bash
pip install -e ../human_kinematic_model
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

### 3.1 CARI v2 Training & Benchmarking (`train.py`, `eval.py`)

Run training and comparative evaluation across observed reaching prefixes (default: 10%, 30%, 50%, 70%):

```bash
source env/bin/activate

# 1. Fit cost weights via IOC and calibrate prediction uncertainty
python train.py

# (Optional) Calibration only with default hand-tuned weights:
python train.py ioc.objective=none

# 2. Evaluate kinematic predictor against baselines (Cartesian, Min-Jerk, GCV, CV)
python eval.py eval.params=output/latest_train
```

#### Key Hydra Command-Line Overrides:
| Parameter | Default | Description |
|---|---|---|
| `data.subjects` | `['sub_1', ...]` | Subjects evaluated (default: all 10 subjects) |
| `data.instructions` | `[1, 3, 5]` | Reaching instructions (1=Obj1, 3=Obj2, 5=Obj3; FAST velocity) |
| `data.obs_ratios` | `[0.1, 0.3, 0.5, 0.7]` | Observed fraction of the reach; remaining trajectory is predicted |
| `ioc.objective` | `open_loop` | `open_loop` (tracking error), `likelihood` (one-step gILQR), or `none` |
| `ioc.restarts` | `10` | Parallel Adam restarts in log10 space |
| `eval.params` | `output/latest_train` | Path to fitted parameters, or `null` for default parameters |
| `eval.plot_trials` | `null` | Specific trials (`subject/instruction`) for visual frame sequences |

---

### 3.2 Online Goal Inference on Full Sessions (`evaluation/goal_inference.py`)

In unstructured environments, the target is unknown ahead of time. `goal_inference.py` replays continuous CARI sessions (8 sequential movements across 7 task targets + idle) as perceived by a perception pipeline:

```bash
cd evaluation
# Run online goal filter on CARI sessions
python goal_inference.py

# Re-generate figures and summary tables from saved cache:
python goal_inference.py --reuse output/latest_goal_inference
cd ..
```

The online predictor evaluates 8 goal hypotheses with a receding horizon and updates a recursive Bayesian filter incorporating heading kinematics, gaze cues, and movement onset evidence.

---

### 3.3 Synthetic Diffusion Reaches with NVIDIA Kimodo (`evaluation/kimodo_reaches.py`)

To evaluate whether motion diffusion models can substitute for expensive optical mocap, the toolkit benchmarks synthetic clips generated with **Kimodo**:

```bash
cd evaluation

# 1. Export workspace layouts from CARI cells
python kimodo_reaches.py layout

# 2. Generate 200 synthetic reaching clips via Kimodo diffusion (10 subjects x 4 tasks x 5 runs)
../../kimodo/.venv/bin/python kimodo_reaches.py generate

# 3. Test online goal inference on clean synthetic keypoints
python goal_inference.py --source kimodo --filter-from output/latest_goal_inference

# 4. Test online goal inference with simulated ZED depth sensor noise
python goal_inference.py --source kimodo --noise --filter-from output/latest_goal_inference

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
./env/bin/python -m pytest tests/
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
