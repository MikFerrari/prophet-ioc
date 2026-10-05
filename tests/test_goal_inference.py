"""Receding-horizon prediction with goal hypotheses and the goal filter (prophet_ioc.human_prediction)."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_t

from prophet_ioc import human_prediction as hp
from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams

BODY = np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2], dtype=np.float32)


def _history(n=30, speed=0.0):
    q = np.zeros(28, dtype=np.float32)
    q[2], q[6] = 1.2, 1.0
    hist = np.repeat(q[None], n, axis=0)
    hist[:, 0] += np.linspace(0.0, speed, n)
    return hist


def test_minimum_jerk_state_boundary_conditions():
    p0, v0, a0, pf = np.zeros(3), np.array([0.3, 0.1, 0.0]), np.array([1.0, 0.0, -0.5]), np.array([0.4, 0.2, 0.1])
    p, v = hp.minimum_jerk_state(p0, v0, a0, pf, 0.8, 0.0)
    np.testing.assert_allclose(p, p0, atol=1e-12)
    np.testing.assert_allclose(v, v0, atol=1e-12)
    p, v = hp.minimum_jerk_state(p0, v0, a0, pf, 0.8, 0.8)
    np.testing.assert_allclose(p, pf, atol=1e-12)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
    eps = 1e-5
    pa, _ = hp.minimum_jerk_state(p0, v0, a0, pf, 0.8, eps)
    pb, _ = hp.minimum_jerk_state(p0, v0, a0, pf, 0.8, 2 * eps)
    np.testing.assert_allclose((pb - 2 * pa + p0) / eps ** 2, a0, atol=1e-3)


def test_minimum_jerk_remaining_time():
    D = 0.9
    assert abs(hp.minimum_jerk_remaining_time(0.4, 0.0, D) - D) < 1e-9          # at rest: starts now
    times = [hp.minimum_jerk_remaining_time(0.4, v, D) for v in (0.1, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(times, times[1:]))                          # faster -> sooner
    # on a minimum-jerk motion of duration D, the phase is recovered from the remaining distance and the speed
    tau, A = 0.4, 0.5
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    v = A / D * 30 * tau ** 2 * (1 - tau) ** 2
    assert abs(hp.minimum_jerk_remaining_time(A * (1 - s), v, D) - D * (1 - tau)) < 2e-3


def test_env_batch_matches_single_environments():
    params = HumanKinematicParams()
    x0 = np.concatenate([np.r_[0.0, 0.0, 1.0, np.zeros(16)], np.zeros(19)]).astype(np.float32)
    q_ref = np.array([0.0, 0.0, 0.0, 1.0])
    targets = np.array([[0.4, -0.2, 1.1], [0.3, 0.3, 1.3]])
    vels = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.1]])
    dts = np.array([0.05, 0.07])
    batch = hp.reaching_env_batch(BODY, np.zeros(8), x0, q_ref, targets, vels, dts, ["right", "left"])
    U = jnp.zeros((10, 19))
    settings = hp.PredictionSettings(covariance="random_walk")
    out = hp._predict_batch(batch, jnp.broadcast_to(jnp.asarray(x0), (2, 1, 38)), jnp.zeros((38, 38)), params,
                            settings, 10, 10)   # converged: float32 rounding
    X = out["mean"]
    for b in range(2):
        # the batch's convention for the left goal: the hypothesis' target (reaching_env_batch: targets_left default)
        env = hp.make_reaching_env(BODY, np.zeros(8), x0[:19], q_ref, targets[b], dts[b], ["right", "left"][b],
                                   target_left=targets[b])
        env.target_vel = jnp.asarray(vels[b], dtype=jnp.float32)
        env.target_vel_left = env.target_vel       # the batch's convention (target_vels_left default)
        env.x0 = jnp.asarray(x0)
        single = jax.tree.map(lambda leaf, i=b: leaf[i], batch)
        for a, c in zip(jax.tree.leaves(env), jax.tree.leaves(single)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(c), atol=1e-6)
        gains, Xs, _ = hp.solve_kinematic(env, env.x0, U, params, max_iter=10)
        np.testing.assert_allclose(np.asarray(X[b]), np.asarray(Xs), atol=1e-3)


def test_predict_hypotheses_targets():
    params = HumanKinematicParams()
    hist = _history()
    kp = np.asarray(hp._fk_batch(jnp.asarray(hist[-1:]), jnp.asarray(BODY)))[0]
    wrist = kp[hp.hkm.KP_INDEX["right_wrist"]]
    hyps = [hp.Hypothesis("near", "right", tuple(wrist + [0.2, 0.0, 0.0])),
            hp.Hypothesis("far", "right", tuple(wrist + [0.6, 0.2, 0.2])),
            hp.Hypothesis("idle", "right")]
    preds, hs = hp.predict_hypotheses(hist, 1 / 29, BODY, hyps, params, 10, 3, horizon=0.5, nominal_duration=0.9,
                                      tol=1e-3)
    near, far, idle = preds
    # at rest every reach starts now and takes the nominal duration: beyond the 0.5 s window -> temporary targets on
    # the minimum-jerk path, with its velocity at the end of the window
    for p, goal in ((near, hyps[0].goal), (far, hyps[1].goal)):
        assert p.temporary and abs(p.arrival - 0.9) < 1e-6 and abs(p.prediction.t_pred - 0.5) < 1e-6
        tgt, vel = hp.minimum_jerk_state(wrist, np.zeros(3), np.zeros(3), np.asarray(goal), 0.9, 0.5)
        np.testing.assert_allclose(p.target, tgt, atol=1e-6)
        np.testing.assert_allclose(p.target_vel, vel, atol=1e-6)
        end = p.prediction.joints["right_wrist"][-1]
        assert np.linalg.norm(end - p.target) < 0.03
    assert not idle.temporary and np.linalg.norm(idle.target - wrist) < 1e-6
    preds, _ = hp.predict_hypotheses(hist, 1 / 29, BODY, hyps[:1], params, 10, 3, horizon=2.0, nominal_duration=0.9,
                                     settings=hp.PredictionSettings(covariance="both"))
    assert not preds[0].temporary and np.allclose(preds[0].target, hyps[0].goal)
    assert set(preds[0].prediction.cov(0.8)) == {"wrist", "elbow", "passive_wrist"}   # random walk
    assert set(preds[0].prediction.cov()) == {"wrist", "elbow", "passive_wrist"}      # model covariance


def test_grasp_offset():
    """The wrist goal is grasp_offset before the object on the line from the wrist; 0 = the object itself."""
    wrist, obj = np.array([0.0, 0.0, 1.0]), np.array([0.4, 0.3, 1.0])
    np.testing.assert_allclose(hp.wrist_goal(obj, wrist, 0.0), obj)
    np.testing.assert_allclose(hp.wrist_goal(obj, wrist, 0.1), [0.32, 0.24, 1.0], atol=1e-12)
    np.testing.assert_allclose(hp.wrist_goal(obj, wrist, 1.0), wrist, atol=1e-12)   # within the offset: stay
    params = HumanKinematicParams()
    hist = _history()
    kp = np.asarray(hp._fk_batch(jnp.asarray(hist[-1:]), jnp.asarray(BODY)))[0]
    w = kp[hp.hkm.KP_INDEX["right_wrist"]]
    hyp = [hp.Hypothesis("obj", "right", tuple(w + [0.3, 0.0, 0.0]))]
    preds, _ = hp.predict_hypotheses(hist, 1 / 29, BODY, hyp, params, 10, 3, horizon=2.0, nominal_duration=0.9,
                                     grasp_offset=0.1)
    np.testing.assert_allclose(preds[0].goal, w + [0.2, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(preds[0].target, preds[0].goal)


def test_wrist_logdensity_is_student_t():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    S = A @ A.T + 0.1 * np.eye(3)
    r = rng.normal(size=3)
    assert abs(hp.wrist_logdensity(r, S, nu=4.0) - multivariate_t(np.zeros(3), S, df=4.0).logpdf(r)) < 1e-9


def _fake_prediction(name, hand, start, velocity, horizon=1.0, H=10):
    t = np.linspace(0.0, horizon, H + 1)[:, None]
    other = "left" if hand == "right" else "right"
    joints = {j: np.zeros((H + 1, 3)) for j in hp.JOINTS}
    joints[f"{hand}_wrist"] = start + t * velocity
    joints[f"{other}_wrist"] = np.tile([0.0, 0.5, 1.0], (H + 1, 1))
    cov = {k: np.tile(np.eye(3) * 1e-4, (H + 1, 1, 1)) for k in ("wrist", "elbow", "passive_wrist")}
    pred = hp.KinematicPrediction(joints, horizon, horizon / H, cov_model=cov)
    return hp.HypothesisPrediction(hp.Hypothesis(name, hand, (0.0, 0.0, 0.0)), pred, np.zeros(3), np.zeros(3),
                                   horizon, False)


def test_goal_filter_follows_the_evidence_and_switches():
    hyps = [hp.Hypothesis("a", "right", (1.0, 0.0, 0.0)), hp.Hypothesis("b", "right", (0.0, 1.0, 0.0))]
    f = hp.GoalFilter(hyps, pred_noise=None, switch_rate=1.0, evidence_lag=0.2, temperature=0.5, obs_noise=0.01)
    rate = 15.0
    pos = np.zeros(3)
    post = None
    for k in range(40):
        vel = np.array([0.5, 0.0, 0.0]) if k < 20 else np.array([0.0, 0.5, 0.0])   # towards a, then towards b
        pos = pos + vel / rate
        preds = [_fake_prediction("a", "right", pos, [0.5, 0.0, 0.0]),
                 _fake_prediction("b", "right", pos, [0.0, 0.5, 0.0])]
        post = f.update(k / rate, preds)
        if k == 19:
            assert post[0] > 0.95
    assert post[1] > 0.95
    # the cue prior multiplies the output without entering the filter state
    g = copy.deepcopy(f)
    with_prior = f.update(40 / rate, preds, np.array([0.0, 8.0]))
    without = g.update(40 / rate, preds)
    np.testing.assert_allclose(f.log_belief, g.log_belief)
    assert with_prior[1] > without[1]
