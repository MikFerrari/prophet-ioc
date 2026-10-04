"""ROS 2 node: prediction of human upper-body motion with the 19-DOF kinematic model and online goal inference
(prophet_ioc.human_prediction).

Input (parameter `input`):
- "zed": zed_msgs/ObjectsStamped of the ZED body tracking (zed-ros2-wrapper); the skeleton of the selected person is
  (optionally) transformed to `world_frame` with TF and converted to the 28-DOF configuration by the IK of the human
  kinematic model (ik.ZedIK: human_kinematic_model_jax.ik); its nose and ears give the gaze cue;
- "joints": std_msgs/Float64MultiArray [q (28), body_params (8)] from an external IK node (no gaze cue).

Goals: the candidate goal locations of the cell (`goal_names`, `goal_positions`, `goal_hands`), each with the hand(s)
that may reach it, plus an "idle" hypothesis (coming to rest). Without candidate goals, the known target of the
`target` parameter or of `target_topic` (geometry_msgs/PointStamped), reached with either hand. All in the frame of
the input.

Observation window: the IK configurations of the person are kept in a circular buffer holding the last
`observation_time` seconds of measurements (its length follows the input rate). Every 1/`rate` s the buffer is
read and resampled to `observation_samples` uniform samples (constant shapes: the prediction is compiled once).

Prediction window and receding horizon: every hypothesis is predicted over [0, `prediction_time`] after the last
observation (predict_hypotheses): its reach is assumed under way, or starting now, as a minimum-jerk motion of
`nominal_duration` s; the optimal-control problem goes to the goal if it is expected within the window, otherwise to
the point of the minimum-jerk path reached at the end of the window (temporary target, with its velocity), and is
solved again at the next tick. The goal filter (GoalFilter: evidence of the predictions made `evidence_lag` s
earlier, switching rate, heading and gaze cue prior) gives the posterior over the hypotheses.

Outputs: the most probable hypothesis as human_motion_prediction_msgs/PredictedMotion (joint positions over the
window, wrist / elbow covariances, goal, temporary target, posterior over all hypotheses) on ~/predicted_motion; all
hypotheses (PredictedMotionHypotheses, sorted by probability) on ~/hypotheses; visualization_msgs/MarkerArray
(predicted joint trajectories, final posture, 95 % wrist spheres, goals with their probability) on ~/markers.
"""

import collections
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from human_motion_prediction_msgs.msg import PredictedMotion, PredictedMotionHypotheses
from human_motion_predictor import engine, ik

CHI2_3_95 = 7.815
BONES = [("head", "chest"), ("chest", "pelvis"), ("chest", "right_shoulder"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("chest", "left_shoulder"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist")]


def _nioc_root() -> Path:
    return Path(os.environ.get("PROPHET_ROOT", "/workspace/prophet-ioc"))


def _point(v) -> Point:
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


class HumanMotionPredictor(Node):
    def __init__(self):
        super().__init__("human_motion_predictor")
        p = self.declare_parameter
        self.input = p("input", "zed").value
        p("skeleton_topic", "/zed/zed_node/body_trk/skeletons")
        p("joints_topic", "~/joints")
        p("target_topic", "~/target")
        p("target", [0.0])                      # [x, y, z] known target; a single value = none (use target_topic)
        goal_names = [g for g in p("goal_names", [""]).value if g]                  # candidate goals of the cell
        goal_positions = np.asarray(p("goal_positions", [0.0]).value, dtype=float)  # [x1, y1, z1, x2, ...]
        goal_hands = [h for h in p("goal_hands", [""]).value if h]                  # right | left | both, per goal
        self.idle = p("idle_hypothesis", True).value
        # inferred: the goal is inferred among goal_names | known: the current goal is given (target / target_topic,
        # e.g. by the task scheduler); the filter then only finds the hand and the onset
        self.goal_mode = p("goal_mode", "inferred").value
        # mixture: the published wrist covariance includes the goal uncertainty | map: the hypothesis' own
        self.uncertainty = p("uncertainty", "mixture").value
        if self.goal_mode not in ("inferred", "known") or self.uncertainty not in ("mixture", "map"):
            raise ValueError("goal_mode must be inferred | known, uncertainty mixture | map")
        if self.goal_mode == "known":
            goal_names = []
        elif not goal_names:
            raise ValueError("goal_mode inferred needs the candidate goals (goal_names, goal_positions); "
                             "use goal_mode known with a target otherwise")
        self.person_id = p("person_id", -1).value          # ZED tracking id (label_id); -1 = first tracked person
        self.observation_time = p("observation_time", 1.0).value      # s of measurements used for the prediction
        self.observation_samples = p("observation_samples", 30).value  # uniform samples of the window (filter)
        self.prediction_time = p("prediction_time", 1.0).value        # s predicted after the last observation
        self.prediction_dt = p("prediction_dt", 0.05).value           # s between published prediction samples
        self.rate = p("rate", 15.0).value
        self.nominal_duration = p("nominal_duration", 0.95).value     # s, minimum-jerk duration of a reach
        self.stop_time = p("stop_time", 0.3).value                    # s, idle hypothesis: time to come to rest
        filter_cfg = {k: p(k, v).value for k, v in (("switch_rate", 0.5), ("evidence_lag", 0.3),
                                                    ("evidence_temperature", 0.25), ("evidence_obs_noise", 0.01))}
        self.kappa_heading = p("kappa_heading", 2.0).value
        self.kappa_gaze = p("kappa_gaze", 1.0).value
        self.hand = p("hand", "auto").value                 # auto (goal_hands) | right | left
        self.device = p("device", "").value or os.environ.get("HMP_DEVICE", "cpu")   # JAX device: cpu | gpu
        # python of another environment for the prediction engine (engine.py); "" = $HMP_WORKER_PYTHON or in-process
        self.worker_python = p("worker_python", "").value or os.environ.get("HMP_WORKER_PYTHON", "")
        self.model_config = p("model_config", "").value     # "" = <PROPHET_ROOT>/config/model/human_kinematic.yaml
        self.params_file = p("params_file", "").value       # train.py output (cost weights + pred_noise), optional
        self.publish_markers = p("publish_markers", True).value
        self.publish_hypotheses = p("publish_hypotheses", True).value
        self.frame_id = p("frame_id", "").value              # "" = frame of the input messages
        self.warmup = p("warmup", True).value
        self.world_frame = p("world_frame", "").value       # "" = keep the frame of the ZED message
        head_keypoint = p("head_keypoint", "nose").value    # nose (as CARI) | ears (midpoint, as human_kinematics_ros)
        joint_limits = p("joint_limits", "cari").value      # cari (+-pi, IK of the CARI dataset) | model (anatomical)

        # JAX and the model are imported here (slow): after the parameters, so that errors in them show up first
        from prophet_ioc import human_prediction as hp
        self.hp = hp
        cfg = yaml.safe_load(Path(self.model_config or _nioc_root() / "config/model/human_kinematic.yaml").read_text())
        model_params = {k: v for k, v in cfg.items() if k not in ("horizon", "max_iter", "pred_noise")}
        self.H, self.max_iter, self.pred_noise = int(cfg["horizon"]), int(cfg["max_iter"]), float(cfg["pred_noise"])
        latest = _nioc_root() / "output/latest_train/params.json"
        if not self.params_file and latest.exists():   # default: the IOC-fitted weights of the latest train.py run
            self.params_file = str(latest)
        if self.params_file == "initial":                # the initial weights of config/model
            self.params_file = ""
        if self.params_file:
            data = json.loads(Path(self.params_file).read_text())
            model_params = data["params"]
            noise = data.get("pred_noise", self.pred_noise)   # per observed fraction: their median
            self.pred_noise = float(np.median(list(noise.values()))) if isinstance(noise, dict) else float(noise)

        hands = self.hands = ("right", "left") if self.hand == "auto" else (self.hand,)
        if goal_names:
            if goal_positions.size != 3 * len(goal_names) or len(goal_hands) not in (0, len(goal_names)):
                raise ValueError("goal_positions must hold 3 values and goal_hands one entry per goal of goal_names")
            self.goals = {n: goal_positions[3 * i: 3 * i + 3] for i, n in enumerate(goal_names)}
            self.goal_hands = {n: (("right", "left") if not goal_hands or goal_hands[i] == "both" else (goal_hands[i],))
                               for i, n in enumerate(goal_names)}
        else:  # known target (parameter or topic), reached with either hand
            self.goals, self.goal_hands = {}, {"target": hands}
        fixed = list(self.get_parameter("target").value)
        self.target = np.array(fixed, dtype=float) if len(fixed) == 3 else None
        self.hypotheses = self.make_hypotheses(hands)   # [(name, hand, goal (3,) | None)]
        if not self.hypotheses:
            raise ValueError(f"no hypothesis: no goal can be reached with hand {self.hand}")
        self.times = np.arange(0.0, self.prediction_time + 1e-9, self.prediction_dt)
        engine_cfg = {
            "hypotheses": [[n, h, None if g is None else [float(x) for x in g]] for n, h, g in self.hypotheses],
            "params": {k: float(v) for k, v in model_params.items()}, "H": self.H, "max_iter": self.max_iter,
            "pred_noise": self.pred_noise, "horizon": self.prediction_time, "nominal_duration": self.nominal_duration,
            "stop_time": self.stop_time, "times": self.times.tolist(),
            "filter": {"switch_rate": filter_cfg["switch_rate"], "evidence_lag": filter_cfg["evidence_lag"],
                       "temperature": filter_cfg["evidence_temperature"], "obs_noise": filter_cfg["evidence_obs_noise"]},
            "kappa_heading": self.kappa_heading, "kappa_gaze": self.kappa_gaze, "device": self.device,
            "uncertainty": self.uncertainty,
            "warmup_samples": self.observation_samples,
            "warmup_dt": self.observation_time / (self.observation_samples - 1)}
        t0 = time.perf_counter()
        if self.worker_python:
            root = _nioc_root()
            pythonpath = ":".join([str(root), str(Path(os.environ.get("HKM_ROOT", root.parent / "human_kinematic_model"))
                                                   / "scripts"), str(Path(__file__).resolve().parents[1])])
            self.engine = engine.WorkerClient(self.worker_python, engine_cfg, pythonpath)
            where = f"worker {self.worker_python}"
        else:
            self.engine = engine.Engine(engine_cfg)
            self.engine.compile_s = self.engine.warmup() if self.warmup else 0.0
            where = "in-process"
        self.get_logger().info(
            f"model parameters from {self.params_file or self.model_config or 'config'}, horizon {self.H}, "
            f"prediction noise {self.pred_noise:.3g}; prediction engine {where} on {self.engine.device}, "
            f"compiled in {self.engine.compile_s:.1f} s; hypotheses: "
            f"{', '.join(n + ('' if g is None else '/' + h) for n, h, g in self.hypotheses)}")

        self.target_frame = ""
        self.buffer = collections.deque()   # last observation_time s: (stamp, q (28,), body (8,), gaze or None)
        self.input_frame = ""
        self.lock = threading.Lock()
        self.tracked_id = None
        self.reset_filter = False
        self.gap = True
        self.ik = ik.ZedIK(head=head_keypoint, joint_limits=joint_limits) if self.input == "zed" else None
        self.tf_buffer = None
        if self.input == "zed" and self.world_frame:
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        inputs, timer = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        if self.input == "zed":
            from zed_msgs.msg import ObjectsStamped
            self.create_subscription(ObjectsStamped, self.get_parameter("skeleton_topic").value, self.on_skeletons,
                                     10, callback_group=inputs)
        elif self.input == "joints":
            self.create_subscription(Float64MultiArray, self.get_parameter("joints_topic").value, self.on_joints, 50,
                                     callback_group=inputs)
        else:
            raise ValueError(f"input must be 'zed' or 'joints', got {self.input}")
        if not self.goals:
            self.create_subscription(PointStamped, self.get_parameter("target_topic").value, self.on_target, 10,
                                     callback_group=inputs)
        self.pub = self.create_publisher(PredictedMotion, "~/predicted_motion", 10)
        self.pub_all = self.create_publisher(PredictedMotionHypotheses, "~/hypotheses", 10) \
            if self.publish_hypotheses else None
        self.pub_markers = self.create_publisher(MarkerArray, "~/markers", 10) if self.publish_markers else None
        if self.warmup and self.ik is not None:   # the first IK call is otherwise slow (compilation)
            self.ik._ik(np.zeros((13, 3)), self.ik.limits, self.ik.q_previous)
        self.create_timer(1.0 / self.rate, self.on_timer, callback_group=timer)
        self.get_logger().info(f"predicting at {self.rate:g} Hz from the last {self.observation_time:g} s of the "
                               f"{self.input} input, {self.prediction_time:g} s ahead")

    def make_hypotheses(self, hands):
        """(name, hand, goal (3,) | None) hypotheses: (goal, hand) pairs (+ idle); with a known target, its value."""
        if self.goals:
            hyps = [(n, h, np.asarray(pos, dtype=float)) for n, pos in self.goals.items()
                    for h in self.goal_hands[n] if h in hands]
        else:
            pos = self.target if self.target is not None else np.zeros(3)
            hyps = [("target", h, np.asarray(pos, dtype=float)) for h in self.goal_hands["target"]]
        return hyps + ([("idle", hands[0], None)] if self.idle else [])

    # ------------------------------------------------------------------ inputs
    def on_skeletons(self, msg):
        people = [o for o in msg.objects if o.skeleton_available and (self.person_id < 0 or o.label_id == self.person_id)]
        if not people:
            return
        obj = people[0]
        if obj.label_id != self.tracked_id:  # another person: restart the history, IK solution choice and goal filter
            with self.lock:
                self.buffer.clear()
                self.reset_filter = True
            self.ik.reset()
            self.tracked_id = obj.label_id
        kpts = ik.zed_keypoints(obj)
        header = msg.header
        if self.tf_buffer is not None:
            try:
                tf = self.tf_buffer.lookup_transform(self.world_frame, msg.header.frame_id, rclpy.time.Time())
            except Exception as exc:  # tf2 lookup / connectivity / extrapolation errors
                self.get_logger().warn(f"no transform {msg.header.frame_id} -> {self.world_frame}: {exc}",
                                       throttle_duration_sec=5.0)
                return
            from scipy.spatial.transform import Rotation
            r, t = tf.transform.rotation, tf.transform.translation
            kpts = kpts @ Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix().T + np.array([t.x, t.y, t.z])
            header = type(msg.header)(stamp=msg.header.stamp, frame_id=self.world_frame)
        out = self.ik(kpts, obj.body_format)
        if out is not None:
            self.add_frame(header, *out, gaze=ik.zed_gaze(kpts, obj.body_format))

    def on_joints(self, msg):
        data = np.asarray(msg.data, dtype=float)
        if data.shape != (36,):
            self.get_logger().warn(f"joints message with {data.size} values, expected 36 (q 28 + body params 8)",
                                   throttle_duration_sec=5.0)
            return
        self.add_frame(None, data[:28], data[28:])

    def add_frame(self, header, q, body, gaze=None):
        stamp = (header.stamp.sec + 1e-9 * header.stamp.nanosec) if header is not None else \
            self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            if header is not None:
                self.input_frame = header.frame_id
            if self.buffer and stamp <= self.buffer[-1][0]:  # out-of-order / repeated stamp
                return
            self.buffer.append((stamp, np.asarray(q, dtype=np.float32), np.asarray(body, dtype=np.float32), gaze))
            while self.buffer and self.buffer[0][0] < stamp - self.observation_time - 0.1:
                self.buffer.popleft()

    def on_target(self, msg):
        self.target = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.target_frame = msg.header.frame_id

    # ------------------------------------------------------------------ prediction
    def on_timer(self):
        with self.lock:
            if len(self.buffer) < 4 or (not self.goals and self.target is None):
                return
            frames = list(self.buffer)
            frame_id = self.frame_id or self.input_frame or self.target_frame
            reset, self.reset_filter = self.reset_filter, False
        stamps = np.array([f[0] for f in frames])
        t_end = float(stamps[-1])
        if t_end - stamps[0] < 0.9 * self.observation_time or np.max(np.diff(stamps)) > 0.25:
            self.gap = True   # window not filled yet, or gap in the measurements: restart the goal filter after it
            return
        t0 = time.perf_counter()
        q = np.stack([f[1] for f in frames])
        body = np.median(np.stack([f[2] for f in frames]), axis=0)  # the IK estimates the body parameters per frame
        gaze = frames[-1][3]
        hist = self.hp.resample_history(stamps, q, t_end, self.observation_time, self.observation_samples)
        if not self.goals:
            self.hypotheses = self.make_hypotheses(self.hands)
        req = {"t": t_end, "hist": hist, "dt": self.observation_time / (self.observation_samples - 1),
               "body": body.astype(np.float32), "head": None if gaze is None else np.asarray(gaze[0]),
               "gaze": None if gaze is None else np.asarray(gaze[1]), "reset": bool(reset or self.gap),
               "goals": None if self.goals else np.stack([np.zeros(3) if g is None else g
                                                         for _, _, g in self.hypotheses])}
        self.gap = False
        res = self.engine.predict(req)
        post = res["post"]
        order = np.argsort(-post)
        elapsed = time.perf_counter() - t0

        header = rclpy.time.Time(seconds=t_end).to_msg()
        publish_all = self.pub_all is not None and self.pub_all.get_subscription_count() > 0
        msgs = [self.message(header, frame_id, res, i, post, elapsed) for i in (order if publish_all else order[:1])]
        self.pub.publish(msgs[0])
        if publish_all:
            all_msg = PredictedMotionHypotheses()
            all_msg.header = msgs[0].header
            all_msg.hypotheses = msgs
            self.pub_all.publish(all_msg)
        if self.pub_markers is not None and self.pub_markers.get_subscription_count() > 0:
            self.pub_markers.publish(self.markers(msgs[0].header, res, int(order[0]), post))

    def message(self, stamp, frame_id, res, i, post, elapsed):
        name, hand, goal = self.hypotheses[i]
        msg = PredictedMotion()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.person_id = int(self.tracked_id if self.tracked_id is not None else self.person_id)
        msg.reaching_hand = hand
        msg.joint_names = list(self.hp.JOINTS)
        msg.time_from_start = self.times.tolist()
        msg.positions = [_point(p) for p in res["joints"][i].reshape(-1, 3)]
        cov = res["cov"][i]   # (n_t, 3: wrist, elbow, passive wrist, 3, 3)
        msg.wrist_covariance = cov[:, 0].reshape(-1).tolist()
        msg.elbow_covariance = cov[:, 1].reshape(-1).tolist()
        msg.passive_wrist_covariance = cov[:, 2].reshape(-1).tolist()
        msg.goal_name = name
        msg.goal = _point(res["target"][i] if goal is None else goal)
        msg.target = _point(res["target"][i])
        msg.target_is_temporary = bool(res["temporary"][i])
        msg.arrival_time = float(res["arrival"][i])
        msg.probability = float(post[i])
        msg.hypothesis_names = [n + ("" if g is None else "/" + h) for n, h, g in self.hypotheses]
        msg.hypothesis_probabilities = [float(x) for x in post]
        msg.computation_time = elapsed
        return msg

    def markers(self, header, res, best, post):
        arr = MarkerArray()
        red, blue = ColorRGBA(r=0.86, g=0.15, b=0.15, a=0.9), ColorRGBA(r=0.15, g=0.39, b=0.92, a=0.9)
        joints = {j: res["joints"][best][:, k] for k, j in enumerate(self.hp.JOINTS)}
        hand = self.hypotheses[best][1]

        def marker(ns, mid, mtype, color, scale):
            m = Marker()
            m.header, m.ns, m.id, m.type, m.action = header, ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = color
            return m

        for i, j in enumerate(self.hp.JOINTS):
            m = marker("trajectories", i, Marker.LINE_STRIP, red, 0.01)
            m.points = [_point(v) for v in joints[j]]
            arr.markers.append(m)
        m = marker("final_posture", 0, Marker.LINE_LIST, blue, 0.02)
        m.points = [_point(joints[x][-1]) for a, b in BONES for x in (a, b)]
        arr.markers.append(m)
        radii = np.sqrt(CHI2_3_95 * np.trace(res["cov"][best][:, 0], axis1=1, axis2=2) / 3.0)
        for k in range(len(radii)):
            m = marker("wrist_95", k, Marker.SPHERE, ColorRGBA(r=0.86, g=0.15, b=0.15, a=0.12), 2.0 * radii[k])
            m.pose.position = _point(joints[f"{hand}_wrist"][k])
            arr.markers.append(m)
        if res["temporary"][best]:
            m = marker("temporary_target", 0, Marker.SPHERE, ColorRGBA(r=0.95, g=0.75, b=0.1, a=0.9), 0.04)
            m.pose.position = _point(res["target"][best])
            arr.markers.append(m)
        goal_p = collections.defaultdict(float)
        for (name, _, goal), p_ in zip(self.hypotheses, post):
            if goal is not None:
                goal_p[(name, tuple(goal))] += p_
        for k, ((name, pos), p_) in enumerate(goal_p.items()):
            m = marker("goals", k, Marker.SPHERE, ColorRGBA(r=0.1, g=0.7, b=0.2, a=0.15 + 0.8 * p_), 0.04 + 0.08 * p_)
            m.pose.position = _point(pos)
            arr.markers.append(m)
            t = marker("goal_labels", k, Marker.TEXT_VIEW_FACING, ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9), 0.04)
            t.pose.position = _point(np.asarray(pos) + [0.0, 0.0, 0.08])
            t.text = f"{name} {100 * p_:.0f}%"
            arr.markers.append(t)
        return arr


def main():
    rclpy.init()
    node = HumanMotionPredictor()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
