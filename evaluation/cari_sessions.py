"""CARI v2 recording sessions for the evaluation of the online prediction with goal inference (goal_inference.py).

The instructions 0-8 of a subject at one velocity are consecutive segments of one recording (the wrists jump by
< 1.5 cm between segments): home, reach object 1 (right hand), home, object 2 (left hand), home, object 3 (right
hand, cross reach), home, robot end effector (both hands), home. Concatenated, they are a continuous session with 8
movements towards 7 goal locations, replayed as the ROS 2 node would see it.

The goal locations (the layout of the cell) are measured on the sessions of the *other* velocities of the same
subject (same cell, same objects), not on the evaluated one: the end of each segment, FK of the filtered IK angles
(the frame of the model), averaged.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.signal import medfilt

import human_kinematic_model_jax as hkm
from prophet_ioc.data import CariDataset
from prophet_ioc.data.cari import BODY_PARAM_NAMES
from prophet_ioc.human_prediction import Hypothesis

_fk_batch = jax.jit(jax.vmap(hkm.fk, in_axes=(0, None)))

# goal location -> (segment, wrist) whose final position measures it; reaching hand of the location
GOAL_LOCATIONS = {
    "object_1": (1, "right"),
    "object_2": (3, "left"),
    "object_3": (5, "right"),
    "home_right": (0, "right"),
    "home_left": (0, "left"),
    "robot_right": (7, "right"),
    "robot_left": (7, "left"),
}
# hands that may reach each location (task knowledge of the cell): object 3 is "reach with ANY HAND" (right in 26 of the
# 30 sessions, left in 4); the others are the instructed hand
GOAL_HANDS = {name: (("right", "left") if name == "object_3" else (side,)) for name, (_, side) in GOAL_LOCATIONS.items()}
# goal of the movement of each segment (return-home and bimanual segments: the locations of the hands that move)
SEGMENT_GOALS = {1: ["object_1"], 2: ["home_right", "home_left"], 3: ["object_2"], 4: ["home_right", "home_left"],
                 5: ["object_3"], 6: ["home_right", "home_left"], 7: ["robot_right", "robot_left"],
                 8: ["home_right", "home_left"]}
# preprocessed table of the CARI v2 recordings (raw ZED keypoints), in probabilistic_ioc/datasets
CSV = Path(__file__).resolve().parents[3] / "datasets/cari_v2/6_preprocessed/dataset_PICK-&-PLACE.csv"


@dataclass
class Movement:
    segment: int
    onset: int                 # frame indices in the session
    offset: int
    goals: List[str]           # correct goal locations (of the hands that move, the one moving most first)
    hands: List[str]           # hands that move (displacement > 50 % of the largest), the one moving most first


@dataclass
class CariSession:
    subject: str
    velocity: str
    dt: float
    q28_raw: np.ndarray        # (N, 28) IK of the raw ZED keypoints (what the node computes online)
    q28_filt: np.ndarray       # (N, 28) centred Savitzky-Golay filtered angles (ground truth of the future)
    body_params: np.ndarray    # (8,)
    kp: np.ndarray             # (N, 13, 3) FK of q28_filt
    head: Optional[np.ndarray]  # (N, 3) nose (raw ZED), for the gaze cue
    gaze: Optional[np.ndarray]  # (N, 3) unit nose - mid-ears direction
    movements: List[Movement]

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.q28_raw)) * self.dt


def _segments(subject: str, velocity: str) -> List[pd.DataFrame]:
    df = CariDataset().df
    d = df[(df.Subject == subject) & (df.Velocity == velocity)]
    return [g.sort_values("Time") for _, g in sorted(d.groupby("Instruction_id"))]


def _angles(seg: pd.DataFrame, prefix: str) -> np.ndarray:
    """28-DOF angles: IK of the raw keypoints ("q_") or their filtered version ("filt_q_")."""
    return seg[[c for c in seg.columns if c.startswith(prefix)]].to_numpy().astype(np.float64)


def _fill_invalid(q: np.ndarray) -> np.ndarray:
    """Frames without an IK solution (NaN, a few % of the MEDIUM / SLOW recordings, none in FAST) linearly
    interpolated in time."""
    bad = ~np.all(np.isfinite(q), axis=1)
    if bad.any() and not bad.all():
        idx = np.arange(len(q))
        q = q.copy()
        for j in range(q.shape[1]):
            q[bad, j] = np.interp(idx[bad], idx[~bad], q[~bad, j])
    return q


def _unwrap_quaternions(q28: np.ndarray) -> np.ndarray:
    quat = q28[:, 3:7] / np.linalg.norm(q28[:, 3:7], axis=1, keepdims=True)
    for t in range(1, len(quat)):
        if quat[t] @ quat[t - 1] < 0.0:
            quat[t] *= -1.0
    q28 = q28.copy()
    q28[:, 3:7] = quat
    return q28


def layout_goals(subject: str, velocities: Sequence[str] = ("MEDIUM", "SLOW")) -> Dict[str, np.ndarray]:
    """Goal locations {name: (3,)} of a subject's cell, measured on the sessions of `velocities`: the wrist that
    reaches the location (the hand that moves most for the objects, object 3 being "any hand"; the location's side for
    the robot) at the end of the movements towards it (offset of the speed profile, 0.2 s average; the instruction
    segments are timed cues, and on the slower sessions a segment may end before or after the hold), and the wrists at
    rest before the first movement (home)."""
    found: Dict[str, List[np.ndarray]] = {k: [] for k in GOAL_LOCATIONS}
    for velocity in velocities:
        S = load_session(subject, velocity, csv=None)
        wrist = lambda side, a, b: S.kp[a: b, hkm.KP_INDEX[f"{side}_wrist"]].mean(axis=0)
        found["home_right"].append(wrist("right", 0, S.movements[0].onset))
        found["home_left"].append(wrist("left", 0, S.movements[0].onset))
        for m in S.movements:
            for name, (segment, side) in GOAL_LOCATIONS.items():
                if segment == m.segment and not name.startswith("home"):
                    hand = m.hands[0] if name.startswith("object") else side
                    found[name].append(wrist(hand, m.offset, m.offset + int(round(0.2 / S.dt))))
    return {k: np.mean(v, axis=0) for k, v in found.items() if v}


def layout_hypotheses(goals: Dict[str, np.ndarray], idle: bool = True) -> List[Hypothesis]:
    """One hypothesis per goal location and hand that may reach it (GOAL_HANDS, task knowledge of the cell), plus
    "idle"."""
    hyps = [Hypothesis(name, hand, tuple(float(x) for x in pos)) for name, pos in goals.items()
            for hand in GOAL_HANDS[name]]
    return hyps + ([Hypothesis("idle", "right")] if idle else [])


def segment_window(kp: np.ndarray, start: int, stop: int, dt: float, v_thresh_ratio: float = 0.12,
                   hands: Optional[List[str]] = None) -> Tuple[List[str], int, int]:
    """The movement within the frames [start, stop) of keypoints kp (N, 13, 3): the hands that move (given, or those
    whose wrist moves > 50 % of the largest displacement, by displacement: the speed has IK spikes; the one moving
    most first) and the onset / offset frames at v_thresh_ratio of the peak speed of their wrists."""
    w = {s: kp[start: stop, hkm.KP_INDEX[f"{s}_wrist"]] for s in ("right", "left")}
    speed = {s: np.linalg.norm(np.gradient(w[s], dt, axis=0), axis=1) for s in w}
    disp = {s: float(np.linalg.norm(w[s][-25:].mean(axis=0) - w[s][:25].mean(axis=0))) for s in w}
    if hands is None:
        hands = [s for s in sorted(disp, key=disp.get, reverse=True) if disp[s] > 0.5 * max(disp.values())]
    else:
        hands = sorted(hands, key=disp.get, reverse=True)
    v = np.max([speed[s] for s in hands], axis=0)
    i_peak = int(np.argmax(v))
    below = np.where(v[:i_peak] < v_thresh_ratio * v[i_peak])[0]
    onset = int(below[-1]) if len(below) else 0
    below = np.where(v[i_peak:] < v_thresh_ratio * v[i_peak])[0]
    offset = i_peak + int(below[0]) if len(below) else stop - start - 1
    return hands, start + onset, start + offset


def load_session(subject: str, velocity: str = "FAST", csv: Optional[Path] = CSV, v_thresh_ratio: float = 0.12
                 ) -> CariSession:
    """The concatenated session; movements segmented on the wrist speed of each segment (onset / offset at
    v_thresh_ratio of the peak speed of the hand(s) that move, as CariDataset.load_trial). With csv, the raw ZED nose
    and ears of the recording give the gaze cue."""
    segs = _segments(subject, velocity)
    dt = float(np.median(np.diff(segs[0]["Time"].to_numpy())))
    q_raw = _unwrap_quaternions(_fill_invalid(np.concatenate([_angles(s, "q_") for s in segs])))
    q_filt = _unwrap_quaternions(_fill_invalid(np.concatenate([_angles(s, "filt_q_") for s in segs])))
    # component-wise filtered quaternions are not unit and flip sign: median-filtered raw ones (as CariDataset)
    quat = medfilt(q_raw[:, 3:7], (5, 1))
    q_filt[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    body = np.nanmedian(np.concatenate([s[BODY_PARAM_NAMES].to_numpy(dtype=float) for s in segs]),
                        axis=0).astype(np.float32)   # IK estimate of every valid frame
    kp = np.array(_fk_batch(jnp.asarray(q_filt, dtype=jnp.float32), jnp.asarray(body)))

    movements, start = [], 0
    for k, seg in enumerate(segs):
        n = len(seg)
        if k in SEGMENT_GOALS:
            hands, onset, offset = segment_window(kp, start, start + n, dt, v_thresh_ratio)
            goals = [g for g in SEGMENT_GOALS[k] if g.startswith("object") or GOAL_LOCATIONS[g][1] in hands]
            goals.sort(key=lambda g: hands.index(GOAL_LOCATIONS[g][1]) if GOAL_LOCATIONS[g][1] in hands else len(hands))
            movements.append(Movement(k, onset, offset, goals, hands))
        start += n

    head = gaze = None
    if csv is not None and Path(csv).exists():
        cols = [f"human_kp{i}_{a}" for i in (0, 16, 17) for a in "xyz"]
        df = pd.read_csv(csv, usecols=["Subject", "Velocity", "Instruction_id", "Time"] + cols)
        rows = df[(df.Subject == subject) & (df.Velocity == velocity)].sort_values(["Instruction_id", "Time"])
        if len(rows) == len(q_raw):
            k = rows[cols].to_numpy().reshape(-1, 3, 3)
            head = k[:, 0]
            gaze = k[:, 0] - 0.5 * (k[:, 1] + k[:, 2])
            gaze = gaze / (np.linalg.norm(gaze, axis=1, keepdims=True) + 1e-9)
    return CariSession(subject, velocity, dt, q_raw.astype(np.float32), q_filt.astype(np.float32), body, kp, head,
                       gaze, movements)


# =============================================================================
# Synthetic sessions (Kimodo clips, kimodo_reaches.py)
# =============================================================================
def cari_joint_limits() -> np.ndarray:
    """IK joint limits of the CARI v2 dataset (as ros2/human_motion_predictor ik.cari_joint_limits): +-pi, +-pi/2
    for the y rotation of the shoulders and hips."""
    limits = np.tile([-np.pi, np.pi], (28, 1))
    limits[[12, 16, 20, 24]] = [-np.pi / 2, np.pi / 2]
    return limits


UPPER_BODY_DOFS = np.r_[0:18, 26:28]


@jax.jit
def _ik_scan(kpts, limits):
    """Sequential IK of keypoints (T, 13, 3), each frame choosing the solution closest to the previous one (as the
    node, ik.ZedIK); a frame without an upper-body solution keeps the previous configuration, legs without a solution
    their previous values."""
    def step(q_prev, kp):
        q, param, _ = hkm.ik(kp, limits, q_prev)
        ok = jnp.all(jnp.isfinite(q[UPPER_BODY_DOFS]))
        q = jnp.where(jnp.isfinite(q), q, q_prev)
        q = jnp.where(ok, q, q_prev)
        return q, (q, param, ok)
    _, (q, param, ok) = jax.lax.scan(step, jnp.zeros(28, dtype=kpts.dtype), kpts)
    return q, param, ok


def ik_sequence(kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """28-DOF configurations (T, 28), body parameters (T, 8) and validity (T,) of a keypoint sequence (T, 13, 3)."""
    q, param, ok = _ik_scan(jnp.asarray(kpts, dtype=jnp.float32), jnp.asarray(cari_joint_limits(), jnp.float32))
    return np.array(q), np.array(param), np.array(ok)


def noise_bank(subjects: Sequence[str], velocity: str = "FAST") -> Dict[str, np.ndarray]:
    """Per subject, the keypoint noise of the real recording (N, 13, 3): FK of the IK of the raw ZED keypoints minus FK
    of the filtered angles (ZED noise, IK flips, glitches)."""
    out = {}
    for s in subjects:
        S = load_session(s, velocity, csv=None)
        kp_raw = np.array(_fk_batch(jnp.asarray(S.q28_raw), jnp.asarray(S.body_params)))
        out[s] = kp_raw - S.kp
    return out


def load_kimodo_sessions(path: Path, subjects: Optional[Sequence[str]] = None, noise: Optional[Dict] = None,
                         seed: int = 0, dt: float = 0.01) -> List[CariSession]:
    """CariSessions of the Kimodo clips (kimodo_reaches.py generate): keypoints resampled to 1 / dt (cubic), IK (as
    online; with noise = noise_bank(...), of the keypoints plus a random block of the subject's real noise), ground
    truth = IK of the clean keypoints; movements = the reach and the return, segmented on the wrist speed within
    their scheduled windows."""
    from scipy.interpolate import CubicSpline
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    fps = float(d["fps"])
    rng = np.random.default_rng(seed)
    sessions = []
    for i, m in enumerate(meta):
        if subjects is not None and m["subject"] not in subjects:
            continue
        T = d["kp"].shape[1]
        t_src = np.arange(T) / fps
        t = np.arange(0.0, t_src[-1] + 1e-9, dt)
        kp = CubicSpline(t_src, d["kp"][i], axis=0)(t)
        head = CubicSpline(t_src, d["head"][i], axis=0)(t)
        gaze = CubicSpline(t_src, d["gaze"][i], axis=0)(t)
        gaze /= np.linalg.norm(gaze, axis=1, keepdims=True)
        q_clean, param, _ = ik_sequence(kp)
        body = np.nanmedian(np.where(np.isfinite(param), param, np.nan), axis=0).astype(np.float32)
        body = np.where(np.isfinite(body), body, 0.4).astype(np.float32)
        if noise is not None:
            bank = noise[m["subject"]]
            k0 = rng.integers(0, len(bank) - len(t))
            q_raw, _, _ = ik_sequence(kp + bank[k0: k0 + len(t)])
        else:
            q_raw = q_clean
        q_clean, q_raw = _unwrap_quaternions(q_clean.astype(np.float64)), _unwrap_quaternions(q_raw.astype(np.float64))
        kp_gt = np.array(_fk_batch(jnp.asarray(q_clean, dtype=jnp.float32), jnp.asarray(body)))
        hands = m["hands"]
        movements = []
        windows = [(m["t_leave"] - 0.4, m["t_back"] - 0.1, [f"robot_{h}" for h in hands] if m["task"] == "robot"
                    else [m["task"]]),
                   (m["t_back"] - 0.1, min(m["t_home"] + 0.6, t[-1]), [f"home_{h}" for h in hands])]
        for k, (a, b, goals) in enumerate(windows):
            mh, on, off = segment_window(kp_gt, int(a / dt), int(b / dt), dt, hands=list(hands))
            goals = sorted(goals, key=lambda g: mh.index(g.rsplit("_", 1)[-1]) if g.rsplit("_", 1)[-1] in mh else 0)
            movements.append(Movement(1 + k, on, off, goals, mh))
        S = CariSession(m["subject"], "KIMODO", dt, q_raw.astype(np.float32), q_clean.astype(np.float32), body, kp_gt,
                        head, gaze, movements)
        S.name = f"{m['subject']}/{m['task']}/{m['rep']}"
        sessions.append(S)
    return sessions


def pelvis_offsets(subject: str, velocities: Sequence[str] = ("MEDIUM", "SLOW")) -> Dict[str, np.ndarray]:
    """Pelvis (hip midpoint) displacement {goal: (3,)} at the end of the movements towards each goal location from
    the pelvis at rest before the first movement, measured on the sessions of `velocities` (as layout_goals)."""
    found: Dict[str, List[np.ndarray]] = {k: [] for k in GOAL_LOCATIONS if not k.startswith("home")}
    for velocity in velocities:
        S = load_session(subject, velocity, csv=None)
        pelvis = 0.5 * (S.kp[:, hkm.KP_INDEX["left_hip"]] + S.kp[:, hkm.KP_INDEX["right_hip"]])
        rest = pelvis[: S.movements[0].onset].mean(axis=0)
        for m in S.movements:
            for name in found:
                if GOAL_LOCATIONS[name][0] == m.segment:
                    found[name].append(pelvis[m.offset: m.offset + int(round(0.2 / S.dt))].mean(axis=0) - rest)
    return {k: np.mean(v, axis=0) for k, v in found.items() if v}
