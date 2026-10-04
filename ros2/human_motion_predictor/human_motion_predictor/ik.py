"""Inverse kinematics of ZED skeletons: ZED body-tracking keypoints -> 28-DOF configuration of the human kinematic
model, with the model's own IK (human_kinematic_model_jax.ik).

The 13 keypoints of the model (human_kinematic_model_jax.KEYPOINT_NAMES: head, left shoulder / elbow / wrist,
left hip / knee / ankle, right shoulder / elbow / wrist, right hip / knee / ankle) are taken from the ZED keypoints
of the message's body format (zed_msgs/Object.body_format: 0 BODY_18, 1 BODY_34, 2 BODY_38; keypoint order of the
ZED SDK). "head" is the nose (as in the CARI v2 dataset the predictor was tuned on,
human_motion_dataset/config/params.yaml) or the midpoint of the ears (as in human_kinematics_ros); the head distance
parameter and the head angles depend on this choice.

The IK returns q (28): [0:3] chest position, [3:7] chest quaternion (x, y, z, w), [7] shoulder rot x, [8:10] hip rot
z/x, [10:14] right arm, [14:18] left arm, [18:22] right leg, [22:26] left leg, [26:28] head rot x/y; and the 8 body
parameters (shoulder, chest-hip and hip distances, upper / lower arm, upper / lower leg, head distance), estimated
from every frame. Invalid solutions (outside the joint limits) are NaN.
"""

from typing import Optional, Tuple

import numpy as np

BODY_18, BODY_34, BODY_38 = 0, 1, 2

# ZED keypoint index of each model keypoint (KEYPOINT_NAMES order, "head" as a tuple of alternatives)
ZED_INDEX = {
    BODY_18: {"nose": 0, "ears": (16, 17), "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7, "left_hip": 11,
              "left_knee": 12, "left_ankle": 13, "right_shoulder": 2, "right_elbow": 3, "right_wrist": 4,
              "right_hip": 8, "right_knee": 9, "right_ankle": 10},
    BODY_34: {"nose": 27, "ears": (29, 31), "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7, "left_hip": 18,
              "left_knee": 19, "left_ankle": 20, "right_shoulder": 12, "right_elbow": 13, "right_wrist": 14,
              "right_hip": 22, "right_knee": 23, "right_ankle": 24},
    BODY_38: {"nose": 5, "ears": (8, 9), "left_shoulder": 12, "left_elbow": 14, "left_wrist": 16, "left_hip": 18,
              "left_knee": 20, "left_ankle": 22, "right_shoulder": 13, "right_elbow": 15, "right_wrist": 17,
              "right_hip": 19, "right_knee": 21, "right_ankle": 23},
}
UPPER_BODY_DOFS = np.r_[0:18, 26:28]   # chest, trunk, arms, head (the legs are only nominal joints of the predictor)
LEG_DOFS = slice(18, 26)
UPPER_BODY_PARAMS = [0, 1, 2, 3, 4, 7]   # shoulder, chest-hip, hip distances, upper / lower arm, head distance


def cari_joint_limits() -> np.ndarray:
    """Joint limits (28, 2) of the CARI v2 IK (human_motion_dataset perform_inverse_kinematics.py / params.yaml):
    +-pi everywhere, +-pi/2 for the y rotation of the shoulders and hips (q 12, 16, 20, 24). The angles the predictor
    was tuned on were computed with them; the model's defaults (human_kinematic_model_jax.default_joint_limits) are
    anatomical and reject frames where, with ZED noise, a nearly straight elbow / knee or the head rotation goes
    slightly past them (on CARI up to ~70 % of the frames of a reach)."""
    limits = np.tile([-np.pi, np.pi], (28, 1))
    limits[[12, 16, 20, 24]] = [-np.pi / 2, np.pi / 2]
    return limits


def zed_keypoints(obj) -> np.ndarray:
    """3D keypoints (n, 3) of a zed_msgs/Object (obj.skeleton_3d.keypoints[i].kp); invalid keypoints are NaN."""
    return np.array([k.kp for k in obj.skeleton_3d.keypoints], dtype=np.float64)


def zed_gaze(kpts: np.ndarray, body_format: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Head position (nose) and unit gaze direction (nose - midpoint of the ears) of a ZED skeleton, or None if one
    of the three keypoints is missing: the gaze cue of the goal inference."""
    index = ZED_INDEX.get(int(body_format))
    if index is None:
        return None
    nose, ears = kpts[index["nose"]], kpts[list(index["ears"])]
    if not (np.all(np.isfinite(nose)) and np.all(np.isfinite(ears))):
        return None
    d = nose - ears.mean(axis=0)
    n = np.linalg.norm(d)
    return (nose, d / n) if n > 1e-6 else None


class ZedIK:
    """IK of the ZED skeletons of one person, with the previous solution for the choice among the limb solutions
    (human_kinematic_model_jax.ik, jitted). joint_limits: "cari" (cari_joint_limits, default) or "model"
    (human_kinematic_model_jax.default_joint_limits)."""

    def __init__(self, head: str = "nose", joint_limits: str = "cari"):
        import jax
        import human_kinematic_model_jax as hkm

        if head not in ("nose", "ears"):
            raise ValueError(f"head must be 'nose' or 'ears', got {head}")
        self.hkm, self.head = hkm, head
        if joint_limits not in ("cari", "model"):
            raise ValueError(f"joint_limits must be 'cari' or 'model', got {joint_limits}")
        self.limits = cari_joint_limits() if joint_limits == "cari" else np.asarray(hkm.default_joint_limits())
        self._ik = jax.jit(hkm.ik)
        self.q_previous = np.zeros(hkm.N_DOF)
        self.has_previous = False

    def reset(self):
        self.q_previous[:] = 0.0
        self.has_previous = False

    def model_keypoints(self, kpts: np.ndarray, body_format: int) -> Optional[np.ndarray]:
        """The 13 model keypoints (13, 3) from the ZED keypoints, or None if one of them is missing (NaN)."""
        index = ZED_INDEX.get(int(body_format))
        if index is None:
            raise ValueError(f"unknown ZED body format {body_format} (0 BODY_18, 1 BODY_34, 2 BODY_38)")
        rows = []
        for name in self.hkm.KEYPOINT_NAMES:
            if name == "head":
                rows.append(kpts[index["nose"]] if self.head == "nose" else kpts[list(index["ears"])].mean(axis=0))
            else:
                rows.append(kpts[index[name]])
        out = np.stack(rows)
        return None if not np.all(np.isfinite(out)) else out

    def __call__(self, kpts: np.ndarray, body_format: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(q (28,), body_params (8,)) of the ZED keypoints (n, 3), or None if the frame is unusable (missing
        keypoint, or no IK solution of the upper body within the joint limits). Legs without a solution keep their
        last valid values (or zero)."""
        keypoints = self.model_keypoints(kpts, body_format)
        if keypoints is None:
            return None
        q, param, _ = self._ik(keypoints, self.limits, self.q_previous)
        q, param = np.array(q, dtype=np.float64), np.array(param, dtype=np.float64)
        if not (np.all(np.isfinite(q[UPPER_BODY_DOFS])) and np.all(np.isfinite(param[UPPER_BODY_PARAMS]))):
            return None
        if not np.all(np.isfinite(q[LEG_DOFS])):
            q[LEG_DOFS] = self.q_previous[LEG_DOFS]
        param = np.where(np.isfinite(param), param, 0.0)
        self.q_previous, self.has_previous = q, True
        return q, param
