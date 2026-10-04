import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest

# Ensure tests run deterministically on CPU
jax.config.update("jax_platforms", "cpu")

from prophet_ioc.envs.nonlinear_reaching import NonlinearReaching, NonlinearReachingParams
from prophet_ioc.control import gilqr
from prophet_ioc.control.policy import create_lqg_policy
from prophet_ioc.envs.wrappers import EKFWrapper
from prophet_ioc.prediction import MovingWindowMotionPredictor, PredictionResult, cartesian_to_joint


def test_inverse_kinematics_roundtrip():
    """Verify that cartesian_to_joint accurately inverts forward kinematics."""
    env = NonlinearReaching()
    x0 = np.array(env.x0)
    pos0 = np.array(env.e(x0))

    q0 = cartesian_to_joint(pos0, l1=env.l1, l2=env.l2)
    # Check joint angles match
    assert np.allclose(q0[:2], x0[:2], atol=1e-5)

    # Check forward kinematics of recovered joint angles gives original Cartesian point
    pos_rec = np.array(env.e(np.hstack([q0[:2], [0.0, 0.0]])))
    assert np.allclose(pos_rec, pos0, atol=1e-5)


def test_prediction_result_structure():
    """Verify structured PredictionResult container and metric calculations."""
    H_obs = 15
    H_future = 25
    state_dim = 4
    num_samples = 40

    mean_s = np.zeros((H_future, state_dim))
    std_s = np.ones((H_future, state_dim)) * 0.1
    cov_s = np.zeros((H_future, state_dim, state_dim))
    lcl_s = mean_s - 1.96 * std_s
    ucl_s = mean_s + 1.96 * std_s
    samples_s = np.zeros((num_samples, H_future, state_dim))

    c_mean = np.zeros((H_future, 2))
    c_std = np.ones((H_future, 2)) * 0.05
    c_cov = np.zeros((H_future, 2, 2))
    c_lcl = c_mean - 1.96 * c_std
    c_ucl = c_mean + 1.96 * c_std
    c_samples = np.zeros((num_samples, H_future, 2))

    res = PredictionResult(
        mean=mean_s,
        std=std_s,
        cov=cov_s,
        lcl=lcl_s,
        ucl=ucl_s,
        samples=samples_s,
        cartesian_mean=c_mean,
        cartesian_std=c_std,
        cartesian_cov=c_cov,
        cartesian_lcl=c_lcl,
        cartesian_ucl=c_ucl,
        cartesian_samples=c_samples,
        confidence_level=0.95,
        observed_steps=H_obs,
        future_steps=H_future,
        observed_state=np.zeros((H_obs, state_dim)),
        observed_cartesian=np.zeros((H_obs, 2)),
    )

    # Check properties and aliases
    assert np.array_equal(res.uncertainty, res.std)
    assert np.array_equal(res.cartesian_uncertainty, res.cartesian_std)
    assert res.observed_steps == H_obs
    assert res.future_steps == H_future

    # Test ADE, FDE, coverage
    gt_future = np.zeros((H_future, 2))
    assert res.ade(gt_future) == 0.0
    assert res.fde(gt_future) == 0.0
    assert res.coverage_rate(gt_future) == 1.0

    # Test summary string
    summary = res.summary()
    assert "PredictionResult" in summary
    assert "95.0%" in summary


def test_predictor_predict_from_observed_state():
    """Verify that predictor predicts future trajectory chunk with correct shapes and bounds."""
    env = NonlinearReaching()
    params = NonlinearReachingParams(action_cost=1e-4, velocity_cost=1e-2, motor_noise=0.05, obs_noise=0.5)
    predictor = MovingWindowMotionPredictor(env=env, params=params, seed=42)

    # Generate a ground truth trajectory of 35 steps
    x0 = env._reset(None, params)
    b0 = (x0, jnp.eye(4) * 1e-4)
    gains, xbar, ubar = gilqr.solve(p=env, x0=x0, U_init=jnp.zeros((35, 2)), params=params, max_iter=5)
    policy = create_lqg_policy(gains, xbar, ubar)
    ekf = EKFWrapper(NonlinearReaching)(b0=b0)

    states, *_ = ekf.rollout(jax.random.PRNGKey(123), 35, policy, params)
    states = np.array(states)

    # Observe prefix of 15 steps
    H_obs = 15
    H_future = 20
    observed_chunk = states[:H_obs]

    result = predictor.predict(
        observed=observed_chunk,
        future_steps=H_future,
        mode="analytical",
        num_samples=30,
        confidence_level=0.95,
    )

    # Verify structured analytical outputs
    assert isinstance(result, PredictionResult)
    assert result.mean.shape == (H_future, 4)
    assert result.std.shape == (H_future, 4)
    assert result.cov.shape == (H_future, 4, 4)
    assert result.lcl.shape == (H_future, 4)
    assert result.ucl.shape == (H_future, 4)
    assert result.samples is None
    assert result.cartesian_mean.shape == (H_future, 2)
    assert result.cartesian_std.shape == (H_future, 2)
    assert result.cartesian_cov.shape == (H_future, 2, 2)
    assert result.cartesian_lcl.shape == (H_future, 2)
    assert result.cartesian_ucl.shape == (H_future, 2)
    assert result.cartesian_samples is None

    # Verify UCL > LCL pointwise
    assert np.all(result.ucl >= result.lcl)
    assert np.all(result.cartesian_ucl >= result.cartesian_lcl)

    # Verify metrics against true future
    true_future = states[H_obs : H_obs + H_future]
    ade = result.ade(np.array(jax.vmap(env.e)(true_future)))
    assert ade < 0.15  # within 15 cm of actual human arm path
    coverage = result.coverage_rate(np.array(jax.vmap(env.e)(true_future)))
    assert coverage > 0.6  # substantial coverage under 95% confidence bounds

    # Also verify mode="both" returns both analytical bounds and Monte Carlo samples
    result_both = predictor.predict(
        observed=observed_chunk,
        future_steps=H_future,
        mode="both",
        num_samples=30,
    )
    assert result_both.samples.shape == (30, H_future, 4)
    assert result_both.cartesian_samples.shape == (30, H_future, 2)


def test_predictor_predict_from_cartesian():
    """Verify that predictor accepts 2D Cartesian positions directly."""
    env = NonlinearReaching()
    params = NonlinearReachingParams()
    predictor = MovingWindowMotionPredictor(env=env, params=params, seed=7)

    # Fake observed Cartesian points
    p0 = np.array(env.e(env.x0))
    cart_obs = np.tile(p0, (10, 1))

    result = predictor.predict(observed=cart_obs, future_steps=15, num_samples=20)
    assert result.future_steps == 15
    assert result.cartesian_mean.shape == (15, 2)
    assert result.cartesian_ucl.shape == (15, 2)


def test_predictor_fit_window():
    """Verify that predictor fits IOC parameters on a moving window chunk."""
    env = NonlinearReaching()
    params = NonlinearReachingParams(action_cost=1e-4, velocity_cost=1e-2, motor_noise=0.1, obs_noise=1.0)
    predictor = MovingWindowMotionPredictor(env=env, params=params, seed=1)

    # Generate 3 trajectory demonstrations of 30 steps
    x0 = env._reset(None, params)
    b0 = (x0, jnp.eye(4) * 1e-4)
    gains, xbar, ubar = gilqr.solve(p=env, x0=x0, U_init=jnp.zeros((30, 2)), params=params, max_iter=5)
    policy = create_lqg_policy(gains, xbar, ubar)
    ekf = EKFWrapper(NonlinearReaching)(b0=b0)

    trajs, *_ = ekf.simulate(jax.random.PRNGKey(42), steps=30, trials=3, policy=policy, params=params)

    # Fit on moving window from step 0 to 15 with 1 restart
    res = predictor.fit_window(trajs, window_start=0, window_length=15, restarts=1)
    assert predictor.is_trained
    assert predictor.params.action_cost > 0
    assert predictor.params.motor_noise > 0

