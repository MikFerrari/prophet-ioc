"""Shared representation of the data-driven baselines (ProMP, DMP): reaches in a canonical, body-centred frame.

Both baselines are learned from complete reaches of the training subjects and predict the 9 upper-body joints of the
evaluation (`JOINTS`, as in `prophet_ioc.human_prediction`) in Cartesian space, like the other baselines. Reaches of
different subjects, cells and hands are made comparable by expressing every reach in a canonical frame fixed at its
onset (the first observed frame, always available to a predictor):

- origin: every keypoint is a displacement from its own position at the onset (so all demonstrations start at 0);
- rotation: heading of the body at the onset, yaw only (x forward, y from the right to the left shoulder in the
  horizontal plane, z = world up), so that subjects facing different directions in their cell look alike;
- mirroring: reaches with the left hand are mirrored (y -> -y) and the left / right joints are swapped, so the
  canonical dimensions are always (head, chest, pelvis, reaching shoulder / elbow / wrist, other shoulder / elbow /
  wrist) and the reaching wrist is always dimensions WRIST_DIMS. Left- and right-hand reaches are then pooled.

The canonical vector of a frame is the 9 displacements stacked (27 values, `N_DIMS`).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter

# The 9 upper-body joints of the prediction, same names and order as prophet_ioc.human_prediction.JOINTS
JOINTS = ["head", "chest", "pelvis", "left_shoulder", "left_elbow", "left_wrist",
          "right_shoulder", "right_elbow", "right_wrist"]
# Canonical joint order: "reach" = side of the reaching hand, "other" = the other side
_CANONICAL = ["head", "chest", "pelvis", "reach_shoulder", "reach_elbow", "reach_wrist",
              "other_shoulder", "other_elbow", "other_wrist"]
N_DIMS = 3 * len(_CANONICAL)
WRIST_DIMS = np.arange(3 * 5, 3 * 6)   # reaching wrist in the canonical vector
UP = np.array([0.0, 0.0, 1.0])         # world vertical (CARI v2: z up)


def canonical_joint_names(hand: str) -> list:
    """Real joint name of each canonical joint for a reach with `hand` ("right" or "left")."""
    other = "left" if hand == "right" else "right"
    return [n.replace("reach", hand).replace("other", other) for n in _CANONICAL]


@dataclass
class CanonicalFrame:
    """Onset-fixed frame of one reach: canonical = M^T (world - origin), with M = R diag(1, +-1, 1)."""
    names: list            # real joint name of each canonical joint
    origin: np.ndarray     # (9, 3) onset position of each canonical joint
    M: np.ndarray          # (3, 3) canonical -> world (rotation, times the mirroring for a left-hand reach)

    def to_canonical(self, joints: Dict[str, np.ndarray]) -> np.ndarray:
        """{joint: (n, 3) world} -> (n, 27) canonical."""
        P = np.stack([np.asarray(joints[n], dtype=float) for n in self.names], axis=1)   # (n, 9, 3)
        return ((P - self.origin) @ self.M).reshape(len(P), N_DIMS)

    def point_to_canonical(self, p: np.ndarray, joint_index: int) -> np.ndarray:
        """A world point (3,) as the displacement of canonical joint `joint_index`."""
        return (np.asarray(p, dtype=float) - self.origin[joint_index]) @ self.M

    def from_canonical(self, Y: np.ndarray) -> Dict[str, np.ndarray]:
        """(n, 27) canonical -> {joint: (n, 3) world}."""
        P = Y.reshape(len(Y), len(self.names), 3) @ self.M.T + self.origin
        return {n: P[:, k] for k, n in enumerate(self.names)}

    def cov_from_canonical(self, S: np.ndarray) -> Dict[str, np.ndarray]:
        """(n, 27, 27) canonical covariances -> {joint: (n, 3, 3) world position covariance}."""
        return {n: self.M @ S[:, 3 * k: 3 * k + 3, 3 * k: 3 * k + 3] @ self.M.T for k, n in enumerate(self.names)}


def canonical_frame(onset_pose: Dict[str, np.ndarray], hand: str) -> CanonicalFrame:
    """Canonical frame of a reach from its onset pose ({joint: (3,)}) and reaching hand."""
    names = canonical_joint_names(hand)
    lateral = np.asarray(onset_pose["left_shoulder"], float) - np.asarray(onset_pose["right_shoulder"], float)
    lateral = lateral - lateral.dot(UP) * UP
    y = lateral / max(np.linalg.norm(lateral), 1e-9)
    x = np.cross(y, UP)
    R = np.stack([x, y, UP], axis=1)                       # columns: canonical axes in the world frame
    mirror = np.diag([1.0, -1.0 if hand == "left" else 1.0, 1.0])
    origin = np.stack([np.asarray(onset_pose[n], dtype=float) for n in names])
    return CanonicalFrame(names, origin, R @ mirror)


@dataclass
class Reach:
    """One demonstration: the 9 joints from the onset to the end of the reach (offset), every frame."""
    joints: Dict[str, np.ndarray]   # {joint: (n, 3)}
    hand: str                       # reaching hand
    dt: float                       # frame period (s)

    def canonical(self) -> np.ndarray:
        """(n, 27) canonical trajectory (starts at 0)."""
        frame = canonical_frame({j: v[0] for j, v in self.joints.items()}, self.hand)
        return frame.to_canonical(self.joints)


@dataclass
class BaselinePrediction:
    """Prediction of the 9 joints on the prediction time grid, with optional position covariances (ProMP)."""
    joints: Dict[str, np.ndarray]                 # {joint: (H+1, 3)}
    cov: Optional[Dict[str, np.ndarray]] = None   # {joint: (H+1, 3, 3)}


def sg_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Velocity at the last frame (Savitzky-Golay, window <= 7, order 2), as prophet_ioc.human_prediction."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return np.zeros(positions.shape[1:])
    w = min(7, len(positions))
    w -= 1 - w % 2
    if w >= 5:
        return savgol_filter(positions, w, 2, deriv=1, delta=dt, axis=0)[-1]
    return (positions[-1] - positions[-2]) / dt


def check_reaches(reaches: Sequence[Reach]) -> None:
    if not reaches:
        raise ValueError("no training reaches")
    for r in reaches:
        missing = [j for j in JOINTS if j not in r.joints]
        if missing or r.hand not in ("left", "right") or len(next(iter(r.joints.values()))) < 3:
            raise ValueError(f"invalid training reach (hand {r.hand}, missing {missing})")
