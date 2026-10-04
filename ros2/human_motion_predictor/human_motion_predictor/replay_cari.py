"""Replays a CARI v2 session for the predictor, without camera.

Publishes, in real time and in a loop, either (mode:=zed) the raw ZED BODY_18 keypoints of the session as the ZED
body tracking does (zed_msgs/ObjectsStamped, one person), or (mode:=joints) their IK angles with the body parameters
on the joints topic (std_msgs/Float64MultiArray [q (28), body_params (8)]). The session (home, object 1, home,
object 2, home, object 3, home, robot, home) is a .npz written on the host by evaluation/export_replay.py, with the
goal locations of the cell as a parameter file of the predictor (the CARI cache itself is a numpy-2 pickle,
unreadable with the numpy 1.26 of ROS 2 Jazzy).

    ros2 run human_motion_predictor replay_cari --ros-args -p trial_file:=/workspace/prophet-ioc/output/replay/sub_4_FAST.npz
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class CariReplay(Node):
    def __init__(self):
        super().__init__("cari_replay")
        p = self.declare_parameter
        trial_file = p("trial_file", "/workspace/prophet-ioc/output/replay/sub_4_FAST.npz").value
        self.mode = p("mode", "joints").value  # joints | zed
        skeleton_topic = p("skeleton_topic", "/zed/zed_node/body_trk/skeletons").value
        self.frame_id = p("frame_id", "world").value
        self.speed = p("speed", 1.0).value
        rate = p("rate", 30.0).value      # Hz published (the ZED body tracking runs at 15-30 Hz; the data at 100 Hz)
        joints_topic = p("joints_topic", "/human_motion_predictor/joints").value

        data = np.load(trial_file)
        self.q, self.body, self.dt = data["q28"], data["body_params"], float(data["dt"])
        self.movements, self.movement_goals = data["movements"], data["movement_goals"]
        if self.mode == "zed":
            from zed_msgs.msg import ObjectsStamped
            self.zed_kpts, self.body_format = data["zed_kpts"], int(data["zed_body_format"])
            self.pub_zed = self.create_publisher(ObjectsStamped, skeleton_topic, 10)
        elif self.mode != "joints":
            raise ValueError(f"mode must be 'joints' or 'zed', got {self.mode}")
        self.pub_joints = self.create_publisher(Float64MultiArray, joints_topic, 50)
        self.k = 0
        self.stride = max(int(round(1.0 / (rate * self.dt))), 1)
        self.create_timer(self.stride * self.dt / self.speed, self.step)
        self.get_logger().info(f"replaying {data['session']} ({len(self.q) * self.dt:.1f} s, {len(self.movements)} "
                               f"movements) at {1 / (self.stride * self.dt):.0f} Hz")

    def step(self):
        for (segment, onset, offset), goal in zip(self.movements, self.movement_goals):
            if self.k <= onset < self.k + self.stride:
                self.get_logger().info(f"movement {segment} towards {goal} starts")
        if self.mode == "zed":
            self.pub_zed.publish(self.zed_message(self.zed_kpts[self.k]))
        else:
            self.pub_joints.publish(Float64MultiArray(data=np.concatenate([self.q[self.k], self.body]).tolist()))
        self.k = (self.k + self.stride) % len(self.q)


    def zed_message(self, kpts):
        from zed_msgs.msg import Object, ObjectsStamped
        msg = ObjectsStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        obj = Object()
        obj.label, obj.label_id, obj.confidence = "Person", 0, 99.0
        obj.tracking_available, obj.skeleton_available, obj.body_format = True, True, self.body_format
        all_kpts = np.full((len(obj.skeleton_3d.keypoints), 3), np.nan, dtype=np.float32)
        all_kpts[: len(kpts)] = kpts
        for k, v in zip(obj.skeleton_3d.keypoints, all_kpts):
            k.kp = v
        msg.objects = [obj]
        return msg


def main():
    rclpy.init()
    node = CariReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
