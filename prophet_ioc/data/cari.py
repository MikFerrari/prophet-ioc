"""CARI v2 Human Motion Dataset loader and utilities for NIOC.

Provides structured access to real human pick-and-place reaching trials from the CARI v2
mocap dataset, including 28-DOF IK joint angles, 8 anatomical body parameters,
and 3D optical marker keypoint trajectories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Standard paths to search for cached CARI v2 pickle
SEARCH_PATHS = [
    Path(__file__).resolve().parents[3] / "prob_ioc" / "cache" / "pick_place_joints.pkl",
    Path(__file__).resolve().parents[1] / "cache" / "pick_place_joints.pkl",
    Path("/home/dirichlet/projects/human_motion_prediction/prob_ioc/cache/pick_place_joints.pkl"),
]

INSTRUCTION_METADATA = {
    1: {"name": "Reach Object 1 (Right Hand)", "hand": "right", "target_obj": "Object 1"},
    2: {"name": "Return to Home", "hand": "both", "target_obj": "Home"},
    3: {"name": "Reach Object 2 (Left Hand)", "hand": "left", "target_obj": "Object 2"},
    4: {"name": "Return to Home", "hand": "both", "target_obj": "Home"},
    5: {"name": "Reach Object 3 (Cross Reaching)", "hand": "right", "target_obj": "Object 3"},
    6: {"name": "Return to Home", "hand": "both", "target_obj": "Home"},
    7: {"name": "Reach Robot EE (Bimanual)", "hand": "both", "target_obj": "Robot EE"},
    8: {"name": "Return to Home", "hand": "both", "target_obj": "Home"},
    0: {"name": "Home Position", "hand": "both", "target_obj": "Home"},
}

BODY_PARAM_NAMES = [
    "shoulder_distance",
    "chest_hip_distance",
    "hip_distance",
    "upper_arm_length",
    "lower_arm_length",
    "upper_leg_length",
    "lower_leg_length",
    "head_distance",
]


@dataclass
class CariTrial:
    """Encapsulates a single human reaching trial from the CARI v2 dataset."""
    subject: str
    velocity: str
    instruction_id: int
    reaching_hand: str
    task_description: str
    dt: float
    time: np.ndarray
    duration: float
    onset_idx: int
    offset_idx: int
    body_params: np.ndarray         # (8,) anatomical link lengths in meters
    q0_28: np.ndarray               # (28,) full kinematic configuration at onset
    q_chest_ref: np.ndarray         # (4,) [x, y, z, w] unit quaternion at onset
    legs_nominal: np.ndarray        # (8,) leg joint angles at onset
    q0_dof: np.ndarray              # (19,) upper-body active DOFs for NIOC at onset
    target_pos: np.ndarray          # (3,) Cartesian reaching target [x, y, z] in meters
    wrist_meas: np.ndarray          # (N, 3) measured wrist keypoint trajectory
    wrist_filt: np.ndarray          # (N, 3) filtered wrist keypoint trajectory
    elbow_meas: np.ndarray          # (N, 3) measured elbow keypoint trajectory
    elbow_filt: np.ndarray          # (N, 3) filtered elbow keypoint trajectory
    shoulder_meas: np.ndarray       # (N, 3) measured shoulder keypoint trajectory
    shoulder_filt: np.ndarray       # (N, 3) filtered shoulder keypoint trajectory
    chest_pos_meas: np.ndarray      # (N, 3) chest 3D translation
    q28_raw: np.ndarray             # (N, 28) raw 28-DOF joint angles
    q28_filt: np.ndarray            # (N, 28) filtered 28-DOF joint angles

    @property
    def reach_frames(self) -> int:
        """Number of frames during the active reach phase."""
        return self.offset_idx - self.onset_idx

    @property
    def reach_duration(self) -> float:
        """Active reach duration in seconds."""
        return self.reach_frames * self.dt


class CariDataset:
    """Manages access to the CARI v2 Pick-and-Place dataset."""

    _cached_df: Optional[pd.DataFrame] = None

    def __init__(self, pkl_path: Optional[str | Path] = None):
        if CariDataset._cached_df is None:
            path = self._resolve_path(pkl_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"CARI v2 cached dataset not found at {path}. Checked locations: {SEARCH_PATHS}"
                )
            CariDataset._cached_df = pd.read_pickle(path)
        self.df = CariDataset._cached_df

    @staticmethod
    def _resolve_path(pkl_path: Optional[str | Path]) -> Path:
        if pkl_path is not None and Path(pkl_path).exists():
            return Path(pkl_path)
        for p in SEARCH_PATHS:
            if p.exists():
                return p
        return SEARCH_PATHS[0]

    def available_subjects(self) -> List[str]:
        return sorted(list(self.df["Subject"].unique()))

    def available_velocities(self) -> List[str]:
        return list(self.df["Velocity"].unique())

    def get_candidate_goals(
        self, subject: str, velocity: str = "FAST"
    ) -> Dict[str, np.ndarray]:
        """Extracts the spatial coordinates of workspace target objects for a subject.

        Returns candidate 3D goal positions for: Object 1, Object 3, and Home.
        """
        goals = {}
        for inst_id, obj_name in [(1, "Object 1"), (5, "Object 3"), (2, "Home")]:
            subset = self.df[
                (self.df["Subject"] == subject)
                & (self.df["Velocity"] == velocity)
                & (self.df["Instruction_id"] == inst_id)
            ]
            if len(subset) > 0:
                # Steady state hold position at end of reach
                wrist_pts = subset[
                    ["filt_human_kp4_x", "filt_human_kp4_y", "filt_human_kp4_z"]
                ].iloc[-25:].mean().to_numpy()
                goals[obj_name] = wrist_pts
        return goals

    def load_trial(
        self,
        subject: str = "sub_4",
        velocity: str = "FAST",
        instruction_id: int = 1,
        v_thresh_ratio: float = 0.05,
    ) -> CariTrial:
        """Extracts and formats a reaching trial for NIOC forecasting.

        The reaching hand is the one whose (filtered) wrist moves most over the instruction's segment (the instructions
        "both hands" and "any hand" do not name one: after a left-hand reach, the left hand returns home). A segment
        longer than 1.5 times the subject's median segment at this velocity (the last one, 8: the subject leaves the
        station after placing the hands) is cut to that median length.
        """
        session = self.df[(self.df["Subject"] == subject) & (self.df["Velocity"] == velocity)]
        subset = session[session["Instruction_id"] == instruction_id].sort_values("Time")

        if len(subset) == 0:
            raise ValueError(
                f"No trial found for Subject={subject}, Velocity={velocity}, Instruction={instruction_id}"
            )
        seg_len = float(np.median(session.groupby("Instruction_id").size()))
        if len(subset) > 1.5 * seg_len:
            subset = subset.iloc[: int(seg_len)]

        time_arr = subset["Time"].to_numpy()
        dt = float(np.median(np.diff(time_arr)))
        duration = float(time_arr[-1] - time_arr[0])

        meta = INSTRUCTION_METADATA.get(
            instruction_id,
            {"name": f"Instruction {instruction_id}", "hand": "right", "target_obj": "Unknown"},
        )
        disp = {}
        for side, kp in (("right", "human_kp4"), ("left", "human_kp7")):
            w = subset[[f"filt_{kp}_x", f"filt_{kp}_y", f"filt_{kp}_z"]].to_numpy()
            disp[side] = float(np.linalg.norm(np.nanmean(w[-25:], axis=0) - np.nanmean(w[:25], axis=0)))
        hand = max(disp, key=disp.get)

        # Determine keypoint prefixes
        kp_wrist = "human_kp4" if hand == "right" else "human_kp7"
        kp_elbow = "human_kp3" if hand == "right" else "human_kp6"
        kp_shldr = "human_kp2" if hand == "right" else "human_kp5"

        wrist_meas = subset[[f"{kp_wrist}_x", f"{kp_wrist}_y", f"{kp_wrist}_z"]].to_numpy()
        wrist_filt = subset[[f"filt_{kp_wrist}_x", f"filt_{kp_wrist}_y", f"filt_{kp_wrist}_z"]].to_numpy()

        elbow_meas = subset[[f"{kp_elbow}_x", f"{kp_elbow}_y", f"{kp_elbow}_z"]].to_numpy()
        elbow_filt = subset[[f"filt_{kp_elbow}_x", f"filt_{kp_elbow}_y", f"filt_{kp_elbow}_z"]].to_numpy()

        shoulder_meas = subset[[f"{kp_shldr}_x", f"{kp_shldr}_y", f"{kp_shldr}_z"]].to_numpy()
        shoulder_filt = subset[[f"filt_{kp_shldr}_x", f"filt_{kp_shldr}_y", f"filt_{kp_shldr}_z"]].to_numpy()

        chest_pos_meas = subset[["q_chest_pos_x", "q_chest_pos_y", "q_chest_pos_z"]].to_numpy()

        # Motion segmentation (onset and offset)
        wrist_speed = np.linalg.norm(np.gradient(wrist_filt, dt, axis=0), axis=1)
        peak_idx = int(np.argmax(wrist_speed))
        v_peak = float(wrist_speed[peak_idx])
        v_thresh = v_thresh_ratio * v_peak

        below_thresh_pre = np.where(wrist_speed[:peak_idx] < v_thresh)[0]
        onset_idx = int(below_thresh_pre[-1]) if len(below_thresh_pre) > 0 else 0

        below_thresh_post = np.where(wrist_speed[peak_idx:] < v_thresh)[0]
        offset_idx = int(peak_idx + below_thresh_post[0]) if len(below_thresh_post) > 0 else len(wrist_speed) - 1

        # Target: hold phase average of reaching wrist
        target_pos = wrist_filt[offset_idx:].mean(axis=0)

        # Body anatomical parameters
        body_params = np.nanmedian(subset[BODY_PARAM_NAMES].to_numpy(dtype=float), axis=0).astype(np.float32)

        # 28 joint columns
        q_cols = [c for c in subset.columns if c.startswith("q_")]
        q28_raw = subset[q_cols].to_numpy().astype(np.float32)

        filt_q_cols = [c for c in subset.columns if c.startswith("filt_q_")]
        q28_filt = subset[filt_q_cols].to_numpy().astype(np.float32)

        # Ensure chest quaternion hemisphere continuity and unit normalization:
        # Linear filtering on quaternions causes antipodal cancellation (norm collapse).
        # We unwrap raw quaternions to ensure continuous trajectory on S^3.
        from scipy.signal import medfilt
        quat_seq = q28_raw[:, 3:7].copy()
        quat_seq /= np.linalg.norm(quat_seq, axis=1, keepdims=True) + 1e-8
        for t in range(1, len(quat_seq)):
            if np.dot(quat_seq[t], quat_seq[t - 1]) < 0.0:
                quat_seq[t] *= -1.0
        # Median filter with window 5 to remove single-frame optical marker dropouts
        quat_seq = medfilt(quat_seq, (5, 1)).astype(np.float32)
        quat_seq /= np.linalg.norm(quat_seq, axis=1, keepdims=True) + 1e-8
        q28_raw[:, 3:7] = quat_seq
        q28_filt[:, 3:7] = quat_seq

        # Posture at onset
        q0_28 = q28_raw[onset_idx]
        q_chest_ref = q0_28[3:7]  # [x, y, z, w]
        legs_nominal = q0_28[18:26]


        # 19 DOFs for upper body:
        # [0:3] chest_pos, [3:6] rotvec w=0, [6:7] shoulder rot x, [7:9] hip rot,
        # [9:13] right arm, [13:17] left arm, [17:19] head
        q0_dof = np.zeros(19, dtype=np.float32)
        q0_dof[0:3] = q0_28[0:3]
        q0_dof[3:6] = 0.0  # reference rotation vector is 0
        q0_dof[6:7] = q0_28[7:8]
        q0_dof[7:9] = q0_28[8:10]
        q0_dof[9:13] = q0_28[10:14]
        q0_dof[13:17] = q0_28[14:18]
        q0_dof[17:19] = q0_28[26:28]

        return CariTrial(
            subject=subject,
            velocity=velocity,
            instruction_id=instruction_id,
            reaching_hand=hand,
            task_description=meta["name"],
            dt=dt,
            time=time_arr,
            duration=duration,
            onset_idx=onset_idx,
            offset_idx=offset_idx,
            body_params=body_params,
            q0_28=q0_28,
            q_chest_ref=q_chest_ref,
            legs_nominal=legs_nominal,
            q0_dof=q0_dof,
            target_pos=target_pos,
            wrist_meas=wrist_meas,
            wrist_filt=wrist_filt,
            elbow_meas=elbow_meas,
            elbow_filt=elbow_filt,
            shoulder_meas=shoulder_meas,
            shoulder_filt=shoulder_filt,
            chest_pos_meas=chest_pos_meas,
            q28_raw=q28_raw,
            q28_filt=q28_filt,
        )


def load_cari_trial(
    subject: str = "sub_4",
    velocity: str = "FAST",
    instruction_id: int = 1,
    pkl_path: Optional[str | Path] = None,
) -> CariTrial:
    """Convenience function to load a CARI v2 reaching trial."""
    ds = CariDataset(pkl_path)
    return ds.load_trial(subject=subject, velocity=velocity, instruction_id=instruction_id)


# =============================================================================
# 19-DOF upper-body joint state (HumanKinematicReaching, pelvis-as-root layout)
# =============================================================================
# Savitzky-Golay settings of the CARI v2 preprocessing (human_motion_dataset/config/params.yaml:
# SAVGOL_FILTER_WINDOW, SAVGOL_ORDER), which produced the filt_q_* / filt_human_kp* columns and the dq_* velocities.
CARI_SG_WINDOW = 101
CARI_SG_ORDER = 3


def relative_rotvec(q_ref: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Rotation vectors w such that q_ref ⊗ exp(w) = q, for quaternions (x, y, z, w) of shape (N, 4)."""
    from scipy.spatial.transform import Rotation

    return (Rotation.from_quat(q_ref).inv() * Rotation.from_quat(quats)).as_rotvec()


def upper_body_dofs(
    q28: np.ndarray, body_params: np.ndarray, q_chest_ref: Optional[np.ndarray] = None
) -> np.ndarray:
    """Maps 28-DOF configurations (N, 28) to the 19 upper-body DOFs of HumanKinematicReaching (pelvis root).

    [0:3] pelvis position, [3:6] chest rotation vector relative to q_chest_ref, [6] shoulder rot x, [7:9] hip rot,
    [9:13] right arm, [13:17] left arm, [17:19] head. If q_chest_ref is None, the rotation vector is set to zero
    (the legacy behaviour of predict_cari.py, which loses the trunk rotation of every frame but q_chest_ref's own).
    """
    q28 = np.atleast_2d(np.asarray(q28, dtype=np.float64))
    quats = q28[:, 3:7] / (np.linalg.norm(q28[:, 3:7], axis=1, keepdims=True) + 1e-12)
    from scipy.spatial.transform import Rotation

    chest_z = Rotation.from_quat(quats).as_matrix()[:, :, 2]
    q = np.zeros((len(q28), 19))
    q[:, 0:3] = q28[:, 0:3] - float(body_params[1]) * chest_z
    if q_chest_ref is not None:
        q[:, 3:6] = relative_rotvec(np.asarray(q_chest_ref, dtype=np.float64), quats)
    q[:, 6:7] = q28[:, 7:8]
    q[:, 7:9] = q28[:, 8:10]
    q[:, 9:13] = q28[:, 10:14]
    q[:, 13:17] = q28[:, 14:18]
    q[:, 17:19] = q28[:, 26:28]
    return q.astype(np.float32)


def sg_upper_body_state(
    trial: "CariTrial",
    q_chest_ref: np.ndarray,
    end: Optional[int] = None,
    window: int = CARI_SG_WINDOW,
    order: int = CARI_SG_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay joint positions and velocities (19 DOFs) of frames [0, end], with the dataset's filter settings.

    The filter is applied to the raw IK output (q28_raw) mapped to the 19 DOFs (over the whole trial it reproduces the
    dataset's dq_* velocities for the arm, hip and head joints, except within half a window of the ends, where the
    window polynomial is evaluated (mode="interp") instead of padding). Pelvis position and chest rotation vector are
    filtered after the mapping (dq_chest_rot_* are derivatives of quaternion components, not angular velocities).
    With `end`, only the frames up to `end` are used (the window is shortened if needed), so no later sample enters
    the estimates.
    """
    from scipy.signal import savgol_filter

    end = len(trial.q28_raw) - 1 if end is None else end
    q_raw = upper_body_dofs(trial.q28_raw[: end + 1], trial.body_params, q_chest_ref).astype(np.float64)
    n = len(q_raw)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= order:
        raise ValueError(f"{n} frames are too few for a Savitzky-Golay filter of order {order}")
    q = savgol_filter(q_raw, w, order, deriv=0, axis=0, mode="interp")
    qd = savgol_filter(q_raw, w, order, deriv=1, delta=trial.dt, axis=0, mode="interp")
    return q.astype(np.float32), qd.astype(np.float32)
