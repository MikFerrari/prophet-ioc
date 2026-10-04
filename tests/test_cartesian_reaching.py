import pytest
import jax.numpy as jnp
import numpy as np

from prophet_ioc.envs.cartesian_reaching import CartesianReaching, CartesianReaching3D, CartesianReachingParams, CartesianReaching3DParams
from prophet_ioc.prediction import MovingWindowMotionPredictor


def test_cartesian_reaching_2d_shapes():
    env = CartesianReaching()
    assert env.state_shape == (4,)
    assert env.action_shape == (2,)
    params = env.get_params_type()()
    x0 = env._reset(None, params)
    assert x0.shape == (4,)
    assert env.e(x0).shape == (2,)


def test_cartesian_reaching_3d_shapes():
    target = jnp.array([0.40, 0.12, 0.25])
    env = CartesianReaching3D(target=target)
    assert env.state_shape == (6,)
    assert env.action_shape == (3,)
    params = env.get_params_type()()
    x0 = env._reset(None, params)
    assert x0.shape == (6,)
    assert env.e(x0).shape == (3,)
    assert jnp.allclose(env.target, target)


def test_cartesian_reaching_3d_dynamics_and_cost():
    env = CartesianReaching3D()
    params = CartesianReaching3DParams()
    state = env._reset(None, params)
    action = jnp.array([1.0, -0.5, 0.2])
    noise = jnp.zeros(6)

    next_state = env._dynamics(state, action, noise, params)
    assert next_state.shape == (6,)

    step_cost = env._cost(state, action, params)
    assert float(step_cost) > 0.0

    final_cost = env._final_cost(state, params)
    assert float(final_cost) > 0.0


def test_cartesian_predictor_analytical():
    target = jnp.array([0.40, 0.12, 0.25])
    env = CartesianReaching3D(target=target)
    pred = MovingWindowMotionPredictor(env)

    # Generate a dummy trajectory
    obs = jnp.tile(env._reset(None, pred.params), (20, 1))
    result = pred.predict(observed=obs, future_steps=30, mode="analytical")

    assert result.mean.shape == (30, 6)
    assert result.cov.shape == (30, 6, 6)
    assert result.cartesian_mean.shape == (30, 3)
    assert result.cartesian_cov.shape == (30, 3, 3)
    assert result.latency_ms > 0
    assert result.mode == "analytical"


def test_cartesian_multipoint_shapes_and_keypoints():
    from prophet_ioc.envs.cartesian_reaching import CartesianMultiPointReaching3D, CartesianMultiPointReaching3DParams
    target_hand = jnp.array([0.40, 0.12, 0.25])
    target_elbow = jnp.array([0.18, 0.05, -0.05])
    env = CartesianMultiPointReaching3D(target_hand=target_hand, target_elbow=target_elbow)

    assert env.state_shape == (12,)
    assert env.action_shape == (6,)
    params = env.get_params_type()()
    x0 = env._reset(None, params)
    assert x0.shape == (12,)

    # Keypoints
    ps, pe, ph = env.keypoints(x0)
    assert ps.shape == (3,)
    assert pe.shape == (3,)
    assert ph.shape == (3,)
    assert jnp.allclose(env.e(x0), ph)
    assert jnp.allclose(env.elbow(x0), pe)


def test_cartesian_multipoint_predictor():
    from prophet_ioc.envs.cartesian_reaching import CartesianMultiPointReaching3D
    target_hand = jnp.array([0.40, 0.12, 0.25])
    target_elbow = jnp.array([0.18, 0.05, -0.05])
    env = CartesianMultiPointReaching3D(target_hand=target_hand, target_elbow=target_elbow)
    pred = MovingWindowMotionPredictor(env)

    obs = jnp.tile(env._reset(None, pred.params), (20, 1))
    result = pred.predict(observed=obs, future_steps=30, mode="analytical")

    assert result.mean.shape == (30, 12)
    assert result.cov.shape == (30, 12, 12)
    assert result.cartesian_mean.shape == (30, 3)
    assert result.cartesian_cov.shape == (30, 3, 3)
    assert result.latency_ms > 0


