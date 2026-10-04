"""Run-time predictive distribution of the kinematic model (prophet_ioc.human_prediction.predictive_distribution):
the fully observed model covariance against a Monte Carlo simulation of the closed loop, the partially observed
(belief-space) prediction against its sigma_o -> 0 limit, positive semi-definiteness, the covariance options and the
settings read from a train.py record."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prophet_ioc import human_prediction as hp
from prophet_ioc.control import ilqr_unrolled
from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams, HumanKinematicReaching
from prophet_ioc.infer.multi_env import simulate_trial

BODY = np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2], dtype=np.float32)
predictive_distribution = jax.jit(hp.predictive_distribution, static_argnames=("settings", "H", "max_iter", "tol"))


def _env(dtype=jnp.float64, dt=0.06):
    env = HumanKinematicReaching(dt=dt, target=jnp.array([0.35, -0.15, 1.05]), dt_scaled_cost=True)
    return jax.tree.map(lambda a: jnp.asarray(a, dtype), env)


def _P0(d=38, q=1e-4, qd=1e-2):
    return jnp.diag(jnp.concatenate([jnp.full(d // 2, q), jnp.full(d // 2, qd)]))


@pytest.mark.parametrize("motor_noise, temperature, P0_scale",
                         [(0.0, 1e-4, 1.0),     # linear-Gaussian closed loop: the propagation is exact
                          (0.3, 1e-6, 1e-2)])   # signal-dependent noise: first order, small deviations
def test_model_covariance_matches_monte_carlo(motor_noise, temperature, P0_scale):
    """Sigma_(k+1) = F Sigma F^T + V V^T + W W^T (closed-loop linearization, motor noise at the nominal command,
    max-ent decision noise, residual folded into the additive noise) against the sample covariance of the simulated
    closed loop: x_0 ~ N(x0, P0), u = U + L (x - X) + Gamma xi, x' = f(x, u, v). Without signal-dependent noise the
    closed loop is linear-Gaussian and the propagation exact; with it, the noise std sigma_m |u| also depends on the
    deviation L (x - X) + Gamma xi, a second-order term that the linearization (noise at the nominal command U)
    neglects: small deviations there."""
    H, N = 6, 40000
    params = HumanKinematicParams(running_vel_cost=1e-4, motor_noise=motor_noise, motor_noise_add=0.05,
                                  residual_noise=0.3)
    settings = hp.PredictionSettings(covariance="model", residual=True, temperature=temperature)
    with jax.enable_x64(True):
        env = _env()
        x0, P0 = env.x0, P0_scale * _P0()
        dist = predictive_distribution(env, x0[None], P0, params, settings, H, max_iter=6)
        gains, X, U = ilqr_unrolled.solve(env, x0, jnp.zeros((H, 19)), params, max_iter=6)
        np.testing.assert_allclose(dist["mean"], X, atol=1e-8)
        policy = hp._plan_policy(gains, X, U, settings.temperature)
        prop = hp.propagation_params(params, settings)
        k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
        xs = x0 + jax.random.normal(k0, (N, 38)) @ jnp.linalg.cholesky(P0).T
        xis = jax.random.normal(k1, (H, N, 19))
        vs = jax.random.normal(k2, (H, N, 4 * 19))
        step = jax.vmap(lambda x, t, xi, v: env._dynamics(x, policy(t, x, xi), v, prop), in_axes=(0, None, 0, 0))
        for t in range(H):
            xs = step(xs, t, xis[t], vs[t])
            S_mc = np.cov(np.asarray(xs), rowvar=False)
            S = np.asarray(dist["Sigma"][t + 1])
            # every variance within 5 % (sampling error ~ sqrt(2 / N) = 0.7 %), correlations within 0.03
            np.testing.assert_allclose(np.diag(S_mc), np.diag(S), rtol=0.05)
            sd, sd_mc = np.sqrt(np.diag(S)), np.sqrt(np.diag(S_mc))
            np.testing.assert_allclose(S_mc / np.outer(sd_mc, sd_mc), S / np.outer(sd, sd), atol=0.03)
            np.testing.assert_allclose(np.asarray(xs).mean(axis=0), X[t + 1], atol=3 * sd.max() / np.sqrt(N) + 1e-6)


@pytest.mark.parametrize("n_prefix", [0, 4])
def test_partially_observed_prediction_tends_to_fully_observed(n_prefix):
    """Belief-space prediction (belief tracked over the prefix, joint (x, b) propagated without conditioning) ->
    the fully observed prediction as the observation noise vanishes (float64): mean and covariance."""
    H = 5
    params = HumanKinematicParams(running_vel_cost=1e-4, motor_noise_add=0.05, residual_noise=0.3)
    with jax.enable_x64(True):
        env = _env()
        x = simulate_trial(env, params, jax.random.PRNGKey(3), 8, solve_iters=4)
        x_prefix = x[4 - n_prefix: 5]
        env_h = hp._with_start(env, x_prefix[-1])
        P0 = _P0(q=1e-5, qd=1e-3)
        full = predictive_distribution(env_h, x_prefix, P0, params, hp.PredictionSettings(), H, max_iter=4)
        gaps = []
        for s in (1e-1, 1e-3, 1e-5):
            part = predictive_distribution(env_h, x_prefix, P0, params._replace(obs_noise=s),
                                           hp.PredictionSettings(observability="partial"), H, max_iter=4)
            scale = np.abs(np.asarray(full["Sigma"])).max()
            gaps.append((float(np.abs(part["mean"] - full["mean"]).max()),
                         float(np.abs(part["Sigma"] - full["Sigma"]).max() / scale)))
    assert gaps[2][0] < 1e-4 and gaps[2][1] < 1e-3, gaps
    assert gaps[2][1] < gaps[0][1], gaps


def _moving_history(n=30, speed=0.05):
    q = np.zeros(28, dtype=np.float32)
    q[2], q[6] = 1.2, 1.0
    hist = np.repeat(q[None], n, axis=0)
    hist[:, 0] += np.linspace(0.0, speed, n)
    hist[:, 10] += np.linspace(0.0, 0.3, n)    # right shoulder rotation: the wrist moves
    return hist


@pytest.mark.parametrize("observability", ["full", "partial"])
def test_prediction_covariances_are_psd(observability):
    params = HumanKinematicParams(obs_noise=0.1, residual_noise=1.0)
    settings = hp.PredictionSettings(observability=observability, belief_steps=4)
    hist = _moving_history()
    kp = np.asarray(hp._fk_batch(jnp.asarray(hist[-1:]), jnp.asarray(BODY)))[0]
    target = kp[hp.hkm.KP_INDEX["right_wrist"]] + np.array([0.25, -0.1, 0.05])
    pred, _ = hp.predict_motion(hist, 1 / 29, BODY, target, params, 8, 3, t_max=0.8, hand="right", tol=1e-3,
                                settings=settings)
    for part, C in pred.cov().items():
        assert C.shape == (9, 3, 3)
        np.testing.assert_allclose(C, np.swapaxes(C, 1, 2), atol=1e-9)
        ev = np.linalg.eigvalsh(C)
        assert np.all(ev > -1e-9 * ev.max()), (part, ev.min())
    assert np.all(np.isfinite(pred.joints["right_wrist"]))


def test_covariance_options():
    """model / random_walk / both: the same mean; cov() (model) and cov(pred_noise) (random walk) select the
    covariance, and a missing one is an error; without the residual the model covariance is smaller."""
    params = HumanKinematicParams(residual_noise=1.0)
    hist = _moving_history()
    kp = np.asarray(hp._fk_batch(jnp.asarray(hist[-1:]), jnp.asarray(BODY)))[0]
    target = kp[hp.hkm.KP_INDEX["right_wrist"]] + np.array([0.25, -0.1, 0.05])
    run = lambda **kw: hp.predict_motion(hist, 1 / 29, BODY, target, params, 8, 3, t_max=0.8, hand="right",
                                         settings=hp.PredictionSettings(**kw))[0]
    model, walk, both, bare = run(), run(covariance="random_walk"), run(covariance="both"), run(residual=False)
    for p in (walk, both, bare):
        np.testing.assert_allclose(p.joints["right_wrist"], model.joints["right_wrist"], atol=1e-6)
    np.testing.assert_allclose(both.cov()["wrist"], model.cov()["wrist"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(both.cov(0.8)["wrist"], walk.cov(0.8)["wrist"], rtol=1e-5, atol=1e-12)
    with pytest.raises(ValueError):
        walk.cov()
    with pytest.raises(ValueError):
        model.cov(0.8)
    tr = lambda p: np.trace(p.cov()["wrist"][-1])
    assert tr(bare) < tr(model)


def test_prediction_settings_from_record():
    model_cfg = dict(prediction_covariance="random_walk", prediction_residual=False, belief_steps=6)
    s = hp.prediction_settings(model_cfg, dict(objective="likelihood", observability="partial", temperature=1e-3))
    assert (s.covariance, s.residual, s.observability, s.temperature, s.belief_steps) == \
        ("random_walk", False, "partial", 1e-3, 6)
    assert hp.prediction_settings(None, dict(objective="open_loop", observability="partial")).observability == "full"
    assert hp.prediction_settings(None, None, dict(temperature=0.0)) == hp.PredictionSettings(temperature=0.0)
    forced = hp.prediction_settings(dict(prediction_observability="full"), dict(objective="likelihood",
                                                                              observability="partial"))
    assert forced.observability == "full"
    with pytest.raises(ValueError):
        hp.PredictionSettings(covariance="calibrated")


def test_partially_observed_hypotheses():
    """Online prediction of several hypotheses in belief space (batched prefix windows)."""
    params = HumanKinematicParams(obs_noise=0.1, residual_noise=1.0)
    hist = _moving_history()
    kp = np.asarray(hp._fk_batch(jnp.asarray(hist[-1:]), jnp.asarray(BODY)))[0]
    w = kp[hp.hkm.KP_INDEX["right_wrist"]]
    hyps = [hp.Hypothesis("a", "right", tuple(w + [0.3, 0.0, 0.0])), hp.Hypothesis("b", "left", (0.3, 0.4, 1.2)),
            hp.Hypothesis("idle", "right")]
    settings = hp.PredictionSettings(observability="partial", belief_steps=4)
    preds, _ = hp.predict_hypotheses(hist, 1 / 29, BODY, hyps, params, 8, 4, horizon=0.6, nominal_duration=0.9,
                                     tol=1e-3, settings=settings)
    for p in preds:
        assert np.all(np.isfinite(p.prediction.joints["right_wrist"]))
        assert np.all(np.linalg.eigvalsh(p.prediction.cov()["wrist"]) > -1e-12)
