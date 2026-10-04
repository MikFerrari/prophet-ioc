"""Data-driven baselines (prophet_ioc.baselines: ProMP, DMP) on synthetic reaches shaped like CARI v2 trials.

Fast, CPU only, numpy: python -m pytest tests/test_baselines.py
"""

import numpy as np
import pytest

from prophet_ioc.baselines import JOINTS, DMP, DMPBaseline, ProMPBaseline, Reach, canonical_frame
from prophet_ioc.baselines.common import N_DIMS, WRIST_DIMS

DT = 0.01
H = 14


def _min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def synthetic_reach(rng, hand=None, n=None):
    """A reach like the CARI v2 trials (onset -> offset, 9 joints, dt = 0.01 s): random body position and heading,
    the reaching wrist on a minimum-jerk path (slightly curved) to a random target in front of the body, the elbow
    following halfway, the trunk leaning a little towards the target, the other arm almost still."""
    hand = hand or ("right" if rng.random() < 0.5 else "left")
    n = n or int(rng.integers(80, 130))
    yaw = rng.uniform(-np.pi, np.pi)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    base = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0])
    side = 1.0 if hand == "left" else -1.0          # body y: right -> left shoulder
    body = {"pelvis": [0, 0, 0.95], "chest": [0, 0, 1.35], "head": [0, 0, 1.6],
            "left_shoulder": [0, 0.18, 1.42], "right_shoulder": [0, -0.18, 1.42],
            "left_elbow": [0.05, 0.22, 1.15], "right_elbow": [0.05, -0.22, 1.15],
            "left_wrist": [0.25, 0.15, 1.05], "right_wrist": [0.25, -0.15, 1.05]}
    goal = np.array([rng.uniform(0.35, 0.6), side * rng.uniform(0.0, 0.4), rng.uniform(0.95, 1.25)])
    s = _min_jerk(np.linspace(0, 1, n))[:, None]
    bump = (np.sin(np.pi * np.linspace(0, 1, n)) * rng.uniform(0.0, 0.06))[:, None] * np.array([0, 0, 1.0])
    reach_w = f"{hand}_wrist"
    w0 = np.array(body[reach_w])
    joints = {}
    for j, p in body.items():
        p = np.array(p, dtype=float)
        if j == reach_w:
            traj = w0 + s * (goal - w0) + bump
        elif j == f"{hand}_elbow":
            traj = p + 0.5 * s * (goal - w0) + 0.5 * bump
        elif j in ("chest", "head", f"{hand}_shoulder"):
            traj = p + 0.15 * s * (goal - w0) * np.array([1, 1, 0])
        else:
            traj = p + 0.02 * s * (goal - w0)
        joints[j] = (traj @ R.T + base) + rng.normal(0, 2e-4, size=traj.shape)
    return Reach(joints, hand, DT)


def eval_inputs(reach, obs_ratio):
    """Observed prefix, target, prediction times (since the onset) and ground truth, as eval.py builds them."""
    n = len(reach.joints["head"])
    f_obs = max(int(round((n - 1) * obs_ratio)), 3)
    fut = np.round(np.linspace(f_obs, n - 1, H + 1)).astype(int)
    fut_time = np.linspace(f_obs * DT, (n - 1) * DT, H + 1)
    obs = {j: v[: f_obs + 1] for j, v in reach.joints.items()}
    gt = {j: v[fut] for j, v in reach.joints.items()}
    return obs, reach.joints[f"{reach.hand}_wrist"][-1], fut_time, gt


def mpjpe(pred, gt):
    return float(np.mean([np.linalg.norm(pred[j] - gt[j], axis=1).mean() for j in JOINTS]))


@pytest.fixture(scope="module")
def reaches():
    rng = np.random.default_rng(0)
    return [synthetic_reach(rng) for _ in range(40)]


def test_canonical_frame_roundtrip_and_mirroring():
    rng = np.random.default_rng(1)
    r = synthetic_reach(rng, hand="left")
    frame = canonical_frame({j: v[0] for j, v in r.joints.items()}, r.hand)
    Y = frame.to_canonical(r.joints)
    assert Y.shape == (len(r.joints["head"]), N_DIMS)
    assert np.allclose(Y[0], 0.0)
    back = frame.from_canonical(Y)
    assert all(np.allclose(back[j], r.joints[j]) for j in JOINTS)
    # the mirror image of a left-hand reach, as a right-hand reach, has the same canonical trajectory
    flip = np.diag([1.0, -1.0, 1.0])
    mirrored = {j.replace("left", "tmp").replace("right", "left").replace("tmp", "right"): v @ flip
                for j, v in r.joints.items()}
    frame_m = canonical_frame({j: v[0] for j, v in mirrored.items()}, "right")
    assert np.allclose(frame_m.to_canonical(mirrored), Y, atol=1e-9)


def test_promp_reproduces_training_trajectory_from_its_prefix(reaches):
    model = ProMPBaseline.fit(reaches)
    for r in reaches[:5]:
        obs, target, fut_time, gt = eval_inputs(r, 0.4)
        pred = model.predict(obs, r.hand, target, fut_time, DT)
        err = mpjpe(pred.joints, gt)
        # prior mean (no conditioning) for reference
        frame = canonical_frame({j: v[0] for j, v in obs.items()}, r.hand)
        prior = frame.from_canonical(model.promp.mean(fut_time / fut_time[-1]))
        assert err < 0.01, err                       # < 1 cm
        assert err < 0.5 * mpjpe(prior, gt)
        w = f"{r.hand}_wrist"
        assert np.linalg.norm(pred.joints[w][-1] - target) < 0.01


def test_promp_covariance_shrinks_at_conditioned_points(reaches):
    model = ProMPBaseline.fit(reaches, n_cond=5)
    promp = model.promp
    r = reaches[0]
    Y = r.canonical()
    z_obs = np.array([0.0, 0.1, 0.2, 0.3])
    idx = np.round(z_obs * (len(Y) - 1)).astype(int)
    all_dims = np.arange(N_DIMS)
    post = promp.condition(list(z_obs) + [1.0], [Y[i] for i in idx] + [Y[-1, WRIST_DIMS]],
                           [all_dims] * len(idx) + [WRIST_DIMS], [promp.sigma_y] * (len(idx) + 1))
    prior_c, post_c = promp.cov(np.r_[z_obs, 1.0], noise=False), post.cov(np.r_[z_obs, 1.0], noise=False)
    for k in range(1, len(z_obs)):  # observed phases (z = 0: every reach starts at 0, prior already ~0)
        assert np.trace(post_c[k]) < 0.1 * np.trace(prior_c[k])
    W = np.ix_(WRIST_DIMS, WRIST_DIMS)
    assert np.trace(post_c[-1][W]) < 0.1 * np.trace(prior_c[-1][W])   # goal: reaching wrist at z = 1
    # posterior covariance is a valid covariance and never larger than the prior along the whole movement
    z = np.linspace(0, 1, 21)
    pc, qc = promp.cov(z), post.cov(z)
    assert np.all(np.linalg.eigvalsh(qc) > 0)
    assert np.all(np.trace(qc, axis1=1, axis2=2) <= np.trace(pc, axis1=1, axis2=2) + 1e-12)
    # world-frame covariances of the predictor: (H+1, 3, 3), symmetric positive definite
    obs, target, fut_time, _ = eval_inputs(r, 0.3)
    pred = model.predict(obs, r.hand, target, fut_time, DT)
    for j in JOINTS:
        assert pred.cov[j].shape == (H + 1, 3, 3)
        assert np.allclose(pred.cov[j], np.swapaxes(pred.cov[j], 1, 2))
        assert np.all(np.linalg.eigvalsh(pred.cov[j]) > 0)


def test_dmp_reproduces_a_demo_with_its_start_and_goal():
    rng = np.random.default_rng(2)
    r = synthetic_reach(rng, hand="right", n=110)
    Y = r.canonical()
    dmp = DMP.fit([Y], [DT])
    t = np.arange(len(Y)) * DT
    tau = t[-1]
    out = dmp.rollout(Y[0], np.zeros(N_DIMS), Y[-1], Y[0], tau, 0.0, t)       # from the onset at rest
    assert np.max(np.abs(out - Y)) < 4e-3, np.max(np.abs(out - Y))   # ~1 % of the reach
    # from a handover in the middle of the demo (its position and velocity), the rest of the demo
    k = 40
    v = np.gradient(Y, DT, axis=0)[k]
    out = dmp.rollout(Y[k], v, Y[-1], Y[0], tau, t[k], t[k:])
    assert np.max(np.abs(out - Y[k:])) < 4e-3, np.max(np.abs(out - Y[k:]))
    # the same through the predictor, in the world frame, with the demo's own goal for every dimension
    model = DMPBaseline.fit([r])
    obs, target, fut_time, gt = eval_inputs(r, 0.4)
    pred = model.predict(obs, r.hand, target, fut_time, DT, goal=Y[-1])
    assert pred.cov is None
    assert mpjpe(pred.joints, gt) < 3e-3


def test_dmp_reaches_the_goal(reaches):
    model = DMPBaseline.fit(reaches)
    rng = np.random.default_rng(3)
    for _ in range(5):
        r = synthetic_reach(rng)                     # unseen reach
        obs, target, fut_time, gt = eval_inputs(r, 0.3)
        target = target + rng.normal(0, 0.05, 3)     # and a goal moved by ~5 cm
        pred = model.predict(obs, r.hand, target, fut_time, DT)
        w = f"{r.hand}_wrist"
        assert np.linalg.norm(pred.joints[w][-1] - target) < 0.01
        assert np.allclose(pred.joints[w][0], obs[w][-1], atol=1e-9)   # starts at the handover


@pytest.mark.parametrize("cls,kwargs", [(ProMPBaseline, {}), (ProMPBaseline, {"phase": "ml"}),
                                        (DMPBaseline, {}), (DMPBaseline, {"goal_model": "mean", "phase": "restart"}),
                                        (DMPBaseline, {"infer_goal": False})])
@pytest.mark.parametrize("obs_ratio", [0.1, 0.3, 0.5, 0.7])
def test_baselines_on_cari_shaped_trials(reaches, cls, kwargs, obs_ratio):
    model = cls.fit(reaches[:30], **kwargs)
    rng = np.random.default_rng(4)
    r = synthetic_reach(rng)                         # held-out reach
    obs, target, fut_time, gt = eval_inputs(r, obs_ratio)
    pred = model.predict(obs, r.hand, target, fut_time, DT)
    assert set(pred.joints) == set(JOINTS)
    for j in JOINTS:
        assert pred.joints[j].shape == (H + 1, 3) and np.all(np.isfinite(pred.joints[j]))
    assert mpjpe(pred.joints, gt) < 0.05             # synthetic reaches are easy: well below 5 cm
    if cls is ProMPBaseline:
        assert all(np.all(np.isfinite(pred.cov[j])) for j in JOINTS)


def test_joint_names_match_the_evaluation():
    hp = pytest.importorskip("prophet_ioc.human_prediction")
    assert list(JOINTS) == list(hp.JOINTS)
