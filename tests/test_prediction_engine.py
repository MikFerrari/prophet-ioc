"""Prediction engine of the ROS 2 node (ros2/human_motion_predictor/human_motion_predictor/engine.py): message
encoding, and the worker process gives the same result as the engine in-process."""

import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "human_motion_predictor"))
from human_motion_predictor import engine  # noqa: E402


def _config():
    return {"hypotheses": [["a", "right", [0.45, -0.25, 1.15]], ["b", "left", [0.4, 0.3, 1.2]], ["idle", "right", None]],
            "params": {"action_cost": 1e-4, "velocity_cost": 1e-2, "posture_cost": 1e-3, "running_vel_cost": 1e-2,
                       "pred_noise": 0.8},
            "H": 6, "max_iter": 2, "pred_noise": 0.8, "horizon": 0.6, "nominal_duration": 0.9, "stop_time": 0.3,
            "times": np.arange(0.0, 0.61, 0.1).tolist(),
            "filter": {"switch_rate": 0.5, "evidence_lag": 0.2, "temperature": 0.5, "obs_noise": 0.01},
            "kappa_heading": 2.0, "kappa_gaze": 1.0, "device": "cpu", "warmup_samples": 12, "warmup_dt": 0.05}


def _request(t):
    q = np.zeros(28, dtype=np.float32)
    q[2], q[6] = 1.2, 1.0
    hist = np.repeat(q[None], 12, axis=0)
    hist[:, 0] += np.linspace(0.0, 0.02, 12) + 0.01 * t
    return {"t": t, "hist": hist, "dt": 0.05, "body": np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2],
                                                                dtype=np.float32),
            "head": np.array([0.0, 0.0, 1.6]), "gaze": np.array([1.0, 0.0, 0.0]), "goals": None, "reset": t == 0.0}


def test_message_roundtrip():
    msg = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.array([True, False]), "s": "x", "n": None,
           "f": 1.5, "l": [1, [2, None]]}
    buf = io.BytesIO()
    engine.send(buf, msg)
    buf.seek(0)
    out = engine.recv(buf)
    assert out["s"] == "x" and out["n"] is None and out["f"] == 1.5 and out["l"] == [1, [2, None]]
    np.testing.assert_array_equal(out["a"], msg["a"])
    assert out["a"].dtype == np.float32 and out["b"].dtype == bool


def test_worker_matches_in_process():
    local = engine.Engine(_config())
    local.warmup()
    pythonpath = os.pathsep.join([str(ROOT), str(ROOT.parent / "human_kinematic_model" / "scripts"),
                                  str(ROOT / "ros2" / "human_motion_predictor")])
    worker = engine.WorkerClient(sys.executable, _config(), pythonpath)
    try:
        for k in range(3):
            a, b = local.predict(_request(k / 15)), worker.predict(_request(k / 15))
            for key in ("post", "joints", "cov", "target", "arrival", "temporary"):
                np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-7)
        assert a["joints"].shape == (3, 7, 9, 3) and a["cov"].shape == (3, 7, 3, 3, 3)
    finally:
        worker.close()


def test_goal_aware_covariance():
    """Certain goal: the hypothesis covariance; two equally likely hypotheses: the law of total covariance."""
    class Hyp:
        def __init__(self, hand):
            self.hand = hand

    eng = engine.Engine.__new__(engine.Engine)
    from prophet_ioc import human_prediction as hp
    eng.hp, eng.hypotheses = hp, [Hyp("right"), Hyp("left")]
    rng = np.random.default_rng(0)
    joints = rng.normal(size=(2, 4, len(hp.JOINTS), 3))
    A = rng.normal(size=(2, 4, 3, 3, 3))
    covs = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(3)          # (K, n_t, wrist/elbow/passive, 3, 3)
    same = eng.goal_aware(joints, covs, np.array([1.0, 0.0]))
    np.testing.assert_allclose(same[0], covs[0], atol=1e-12)
    half = eng.goal_aware(joints, covs, np.array([0.5, 0.5]))
    r = hp.JOINTS.index("right_wrist")
    d = joints[1, :, r] - joints[0, :, r]
    # right wrist of hypothesis 0: its reaching wrist; of hypothesis 1 (left hand): its passive wrist
    expected = 0.5 * covs[0, :, 0] + 0.5 * (covs[1, :, 2] + d[:, :, None] * d[:, None, :])
    np.testing.assert_allclose(half[0, :, 0], expected, atol=1e-12)
    np.testing.assert_allclose(half[0, :, 1], covs[0, :, 1])     # elbow unchanged
