# ROS 2 deployment (Docker, ROS 2 Jazzy)

The predictor runs as the ROS 2 node `human_motion_predictor` ([`../ros2/`](../ros2/)) in its own container, next
to the ZED container (zed-ros2-wrapper): both use the host network and IPC, so the topics are exchanged through DDS
(and shared memory) as long as `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` match.

The image only holds the dependencies (ROS 2 Jazzy, the ZED messages, the node's venv with numpy < 2, the prediction
worker's venv `/opt/jax` with numpy 2 and the current JAX, on the CPU or, in `predictor-gpu`, CUDA). The source
is mounted from the host:

| Host | Container | |
|---|---|---|
| `nioc-neurips/` (this repository) | `/workspace/prophet-ioc` | python code, config, ROS 2 packages `ros2/` |
| `../human_kinematic_model/` (`HKM_SRC`) | `/workspace/human_kinematic_model` | `scripts/human_kinematic_model_jax.py` |
| volume `ros2_ws` | `/ros2_ws` | colcon build / install / log of `ros2/`, python bytecode, JAX compilation cache |

At every start the entrypoint builds `ros2/` (`--symlink-install`), so edits of the python code apply at the next
start of the container, without rebuilding the image (`SKIP_BUILD=1` skips the build). Nothing is written into the
mounted source (bytecode and JAX cache go to the volume). At start-up the prediction worker compiles the prediction
of all the goal hypotheses (~40-65 s) before the node subscribes.

## Run

```bash
cd nioc-neurips
docker compose -f docker/compose.yaml up --build     # predictor on the ZED skeletons (input:=zed), JAX on the CPU
docker compose -f docker/compose.yaml up --build predictor-gpu   # the same with JAX on the GPU (NVIDIA toolkit)

# without camera: replay of a CARI v2 session (home, object 1, home, object 2, home, object 3, home, robot, home),
# either its raw ZED keypoints as zed_msgs/ObjectsStamped (input:=zed, whole pipeline with the IK) or its IK angles
# (input:=joints), with the goal locations of its cell; exported on the host first (the CARI cache is a numpy-2
# pickle, unreadable with the numpy 1.26 of ROS 2 Jazzy)
./.venv/bin/python evaluation/export_replay.py --subject sub_4    # -> output/replay/sub_4_FAST.npz, sub_4_FAST_goals.yaml
docker compose -f docker/compose.yaml run --rm predictor-gpu \
    ros2 launch human_motion_predictor predictor.launch.py input:=zed replay:=true \
    trial_file:=/workspace/prophet-ioc/output/replay/sub_4_FAST.npz

# goal locations of your cell (parameter file, see below); parameters fitted / calibrated by train.py
docker compose -f docker/compose.yaml run --rm predictor ros2 launch human_motion_predictor predictor.launch.py \
    goals_file:=/workspace/prophet-ioc/config/cell_goals.yaml \
    params_file:=/workspace/prophet-ioc/output/train_<ts>/params.json   # default: output/latest_train
```

Environment variables of `compose.yaml`: `ROS_DOMAIN_ID` (0), `RMW_IMPLEMENTATION` (`rmw_cyclonedds_cpp`; Fast DDS is
installed too: it must match the ZED container), `CYCLONEDDS_URI`, `BASE_IMAGE` (`ros:jazzy-ros-base`, e.g. the image
of the robot stack), `HKM_SRC`. `predictor-gpu` is the same image with `jax[cuda12]` (JAX 0.7.1, tested on an RTX
5060), `gpus: all`, no GPU memory preallocation (the GPU is shared with the ZED SDK) and `HMP_DEVICE=gpu`.

Tested with the session replay (sub_4, `input:=zed`: ZED keypoints at 33 Hz -> IK -> 8 hypotheses, 1 s observed, 1 s
predicted every 0.05 s): `predictor-gpu` 15.0 Hz, 47-67 ms per tick (RTX 5060); `predictor` (CPU) 7.8 Hz, ~125 ms per
tick, so set `rate: 7.5` there. Start-up (compilation): ~65 s on the GPU, ~40 s on the CPU.

## Node

| | Topic | Type |
|---|---|---|
| input (`input:=zed`) | `/zed/zed_node/body_trk/skeletons` (`skeleton_topic`) | `zed_msgs/ObjectsStamped` |
| input (`input:=joints`) | `/human_motion_predictor/joints` (`joints_topic`) | `std_msgs/Float64MultiArray` [q (28), body parameters (8)] |
| known target (no goals) | `/human_motion_predictor/target` (`target_topic`), or parameter `target: [x, y, z]` | `geometry_msgs/PointStamped` |
| output | `/human_motion_predictor/predicted_motion` | `human_motion_prediction_msgs/PredictedMotion` (most probable hypothesis) |
| output | `/human_motion_predictor/hypotheses` | `human_motion_prediction_msgs/PredictedMotionHypotheses` (all, by probability) |
| output | `/human_motion_predictor/markers` | `visualization_msgs/MarkerArray` (RViz) |

### Goals and goal inference

The goal of the reach is not given: it is inferred online among the candidate goal locations of the cell, given as
parameters (in the frame of the input), each with the hand(s) that may reach it:

```yaml
human_motion_predictor:
  ros__parameters:
    goal_names: [conveyor, plate_slot_1, plate_slot_2, home]
    goal_positions: [0.6, -0.3, 0.9,  0.4, 0.2, 0.95,  0.4, 0.35, 0.95,  0.3, 0.0, 1.0]   # x, y, z per goal (m)
    goal_hands: [right, both, both, both]                                                 # right | left | both
```

Every (goal, hand) pair is a hypothesis, plus `idle` (coming to rest, no reach). Every tick, all hypotheses are
predicted together (`prophet_ioc.human_prediction.predict_hypotheses`) and a recursive Bayesian filter (`GoalFilter`) gives
their posterior:
- receding horizon: each hypothesis assumes its reach is under way (or starts now) as a minimum-jerk motion of
  `nominal_duration` s (0.95 s on CARI v2 FAST: 1.875 x amplitude / peak speed); its phase, from the remaining
  distance and the wrist speed towards the goal, gives the expected arrival. If the goal is expected within the
  prediction window, the optimal-control problem goes to it (and the posture is held after the arrival); otherwise
  it goes to a **temporary target**, the point of the minimum-jerk path to the goal reached at the end of the window,
  with the path's velocity there as terminal wrist velocity. At the next tick the problem is solved again from the
  new state, so the temporary target moves forward;
- evidence: the observed (Kalman-filtered) positions of both wrists, under the prediction each hypothesis made
  `evidence_lag` s earlier (Student-t with the predicted covariances, robust to IK outliers), tempered by
  `evidence_temperature`; hypotheses switch at rate `switch_rate`;
- cue prior (not accumulated): the direction of the wrist velocity towards the goal (`kappa_heading`, weighted by
  the speed) and the direction of the gaze (`kappa_gaze`: nose - midpoint of the ears of the ZED skeleton,
  `input:=zed` only) towards the goal.

Two switches: `goal_mode: inferred | known` and `uncertainty: mixture | map`. With `goal_mode: known` the goal
is given, not inferred: the known target (`target` or `target_topic`, e.g. from a task scheduler) is reached with either hand,
and the filter only chooses the hand (and idle).

`PredictedMotion` ([`../ros2/human_motion_prediction_msgs/msg/PredictedMotion.msg`](../ros2/human_motion_prediction_msgs/msg/PredictedMotion.msg)):
the positions of the 9 upper-body joints over the prediction window, the 3x3 covariances of the reaching wrist, its
elbow and the other wrist at each sample (95 % region = chi2_3(0.95) = 7.815), the hypothesis (goal name and
position, reaching hand, probability), the target of the optimal-control problem and whether it is temporary, the
expected arrival at the goal, the posterior over all hypotheses and the computation time.
`PredictedMotionHypotheses` holds the `PredictedMotion` of every hypothesis (for a safety layer that needs all of
them, e.g. the union of their 95 % regions).

Observation and prediction windows (online; the 10 / 30 / 50 / 70 % observed fractions are only those of the offline
evaluation `eval.py`):
- `observation_time` (default 1.0 s): the IK configurations are kept in a circular buffer holding the last
  `observation_time` seconds of measurements, so its length follows the input rate (30 frames at 30 Hz). Every
  1 / `rate` s (default 15 Hz) the buffer is read and resampled to `observation_samples` uniform samples (default
  30; constant shapes, so the prediction is compiled once, robust to jitter and dropped frames) for the handover
  Kalman filter. No prediction while the window is not filled or has a gap > 0.25 s.
- `prediction_time` (default 1.0 s, any length) and `prediction_dt` (0.05 s): the published samples, from the last
  observation.

The prediction engine (`human_motion_predictor/engine.py`: all hypotheses in one batched solve, goal filter) runs in a
worker process with its own python environment (`worker_python`, in the images `/opt/jax`: numpy 2 and the current
JAX), because ROS 2 Jazzy's numpy 1.26 caps JAX at 0.7.1, ~3x slower on the GPU for this problem; the node keeps the
ROS I/O, the buffer and the IK. With 8 hypotheses a tick takes ~50-65 ms on the GPU and ~125 ms on the CPU: at 15 Hz
use `predictor-gpu`, or lower `rate` on the CPU. The model parameters come
from the latest `train.py` run (`output/latest_train/params.json`: IOC-fitted weights, calibrated prediction
noise; `params_file` to choose another, `initial` for the initial weights of `config/model/human_kinematic.yaml`).
With `uncertainty: mixture` the published wrist covariances include the goal uncertainty: around the published
hypothesis i, sum_g p_g (Sigma_g + (mu_g - mu_i)(mu_g - mu_i)^T) (law of total covariance over the hypotheses). All parameters: [`../ros2/human_motion_predictor/config/predictor.yaml`](../ros2/human_motion_predictor/config/predictor.yaml).

Offline evaluation of the same pipeline on whole CARI sessions: `evaluation/goal_inference.py`.

### IK

With `input:=zed`, the skeleton of the selected person (`person_id`, default: the first tracked one) goes through the
IK of the human kinematic model, `human_kinematic_model_jax.ik`
([`../ros2/human_motion_predictor/human_motion_predictor/ik.py`](../ros2/human_motion_predictor/human_motion_predictor/ik.py),
`ZedIK`):
- the 13 model keypoints are taken from the ZED keypoints of the message's body format (BODY_18 / 34 / 38); `head`
  is the nose (`head_keypoint: nose`, as in the original CARI v2 data), the centroid of the nose and the ears
  (`centroid`, as in the data of `data.head_keypoint: centroid`) or the midpoint of the ears (`ears`, as in
  `human_kinematics_ros`); it must match the `data.head_keypoint` of the fitted weights;
- joint limits `cari` (default: +-pi, +-pi/2 for the shoulder / hip y rotations, as the IK of the CARI dataset) or
  `model` (anatomical defaults of the model, which reject up to ~70 % of the frames of a CARI reach where, with ZED
  noise, a nearly straight elbow / knee or the head goes slightly past them);
- the previous solution of the same person chooses among the limb solutions; a frame with a missing upper-body
  keypoint or without an upper-body solution is skipped, legs without a solution keep their last values; the body
  parameters are the median over the buffered frames;
- `world_frame`: if set, the keypoints are transformed from the camera frame with TF before the IK (as
  `human_kinematics_ros`), and the prediction is published in that frame.

On the 30 CARI v2 reaches (raw ZED BODY_18 keypoints), every frame is solved, the FK of the solution reproduces the
keypoints to < 0.11 cm, and it matches the dataset's IK (median 0.001 cm; on the ~1 % of the frames where they differ,
the dataset IK is the one that does not reproduce the keypoints).

To run the IK elsewhere, publish its output on the joints topic and launch with `input:=joints`.

Frames: the prediction is published in the frame of the input (or `frame_id`), and the target must be given in that
frame. A z-up frame with gravity (ZED `map` / `odom` with positional tracking), as in the CARI v2 data the model was
tuned on, is preferable.

Online the IK angles are not smoothed with the centred Savitzky-Golay filter of the CARI evaluation (it needs future
frames): the Kalman filter of the handover smooths them, but expect the accuracy of the causal evaluation rather than
the one of `eval.py` (the latter uses the dataset's filtered angles).
