"""Synthetic-session helpers: CARI <-> Kimodo cell frame (evaluation/kimodo_reaches.py) and the sequential IK of
keypoint sequences (evaluation/cari_sessions.py)."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import cari_sessions as cs  # noqa: E402
import kimodo_reaches as kr  # noqa: E402


def test_cell_frame_roundtrip_and_axes():
    lay = {"pelvis": [0.8, -0.2, 1.1], "left": [0.0, -1.0, 0.0], "forward": [-1.0, 0.0, 0.0]}
    f = kr.CellFrame(lay, scale=1.1, root_y=0.95)
    p = np.random.default_rng(0).normal(size=(20, 3)) * 0.4 + lay["pelvis"]
    np.testing.assert_allclose(f.from_kimodo(f.to_kimodo(p)), p, atol=1e-12)
    np.testing.assert_allclose(f.to_kimodo(np.array(lay["pelvis"])), [0.0, 0.95, 0.0], atol=1e-12)
    k = f.to_kimodo(np.array(lay["pelvis"]) + [-0.5, 0.0, 0.0])          # 0.5 m forward in the cell
    np.testing.assert_allclose(k, [0.0, 0.95, 0.55], atol=1e-12)
    k = f.to_kimodo(np.array(lay["pelvis"]) + [0.0, -0.2, 0.1])          # 0.2 m to the left, 0.1 m up
    np.testing.assert_allclose(k, [0.22, 0.95 + 0.11, 0.0], atol=1e-12)


def test_hand_rotation_points_the_hand():
    d = np.array([0.3, -0.2, 0.9])
    d /= np.linalg.norm(d)
    for rest in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
        R = kr._hand_rotation(rest, d)
        np.testing.assert_allclose(R @ rest, d, atol=1e-12)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_ik_sequence_reproduces_model_keypoints():
    rng = np.random.default_rng(1)
    T = 40
    q = np.zeros((T, 28), dtype=np.float32)
    q[:, 2], q[:, 6] = 1.2, 1.0
    t = np.linspace(0.0, 1.0, T)[:, None]
    q[:, 10:14] = np.array([0.3, 0.4, 0.2, 0.8]) * t + 0.1      # right arm reaching
    q[:, 14:18] = np.array([-0.2, 0.3, -0.1, 0.5])
    q[:, 18:26] = rng.uniform(-0.2, 0.2, 8) + 0.05
    body = np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2], dtype=np.float32)
    kp = np.array(cs._fk_batch(jnp.asarray(q), jnp.asarray(body)))
    q_ik, param, ok = cs.ik_sequence(kp)
    assert ok.all()
    np.testing.assert_allclose(np.nanmedian(param, axis=0)[:5], body[:5], atol=2e-3)
    kp_ik = np.array(cs._fk_batch(jnp.asarray(q_ik), jnp.asarray(body)))
    assert np.max(np.linalg.norm(kp_ik - kp, axis=-1)) < 5e-3


def test_known_goal_mask_keeps_the_filter_on_the_allowed_hypotheses():
    import goal_inference as gi
    T, K = 30, 3
    rng = np.random.default_rng(0)
    LL = rng.normal(size=(T, 5, K)) * 3.0
    sess = {"heading": np.zeros((T, K)), "gaze": np.zeros((T, K))}
    mask = np.ones((T, K), bool)
    mask[:15, 1] = False                       # hypothesis 1 allowed only in the second half
    post = gi.filter_posteriors(sess, LL, 1.0, 0.2, 0.5, 0.0, 0.0, 15.0, mask=mask)
    assert np.allclose(post[:15, 1], 0.0) and np.allclose(post.sum(axis=1), 1.0)
    free = gi.filter_posteriors(sess, LL, 1.0, 0.2, 0.5, 0.0, 0.0, 15.0)
    assert np.all(free[:, 1] > 0)
