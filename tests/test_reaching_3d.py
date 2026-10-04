import numpy as np
import jax.numpy as jnp

from prophet_ioc.envs.nonlinear_reaching_3d import NonlinearReaching3D, NonlinearReaching3DParams
from prophet_ioc.control import gilqr
from prophet_ioc.prediction import MovingWindowMotionPredictor, cartesian_to_joint_3d


def test_forward_and_inverse_kinematics():
    env = NonlinearReaching3D()
    test_positions = [
        jnp.array([0.34, 0.02, 0.18], dtype=jnp.float32),
        jnp.array([0.40, 0.12, 0.25], dtype=jnp.float32),
        jnp.array([0.38, -0.08, 0.20], dtype=jnp.float32),
    ]
    for pos in test_positions:
        q = env.ik(pos)
        state = jnp.concatenate([q, jnp.zeros(3, dtype=jnp.float32)])
        pos_rec = env.e(state)
        err = float(jnp.linalg.norm(pos - pos_rec))
        assert err < 1e-5, f"IK/FK mismatch: {err} m"


def test_arm_keypoints_3d():
    env = NonlinearReaching3D()
    state = env.x0
    ps, pe, ph = env.keypoints(state)
    assert ps.shape == (3,)
    assert pe.shape == (3,)
    assert ph.shape == (3,)
    assert np.allclose(ps, 0.0)

    # Upper arm link length
    d_se = float(jnp.linalg.norm(pe - ps))
    assert np.isclose(d_se, env.l1, atol=1e-5)

    # Forearm link length
    d_eh = float(jnp.linalg.norm(ph - pe))
    assert np.isclose(d_eh, env.l2, atol=1e-5)


def test_gilqr_solve_convergence_3d():
    env = NonlinearReaching3D()
    params = env.get_params_type()()
    gains, xbar, ubar = gilqr.solve(
        p=env,
        x0=env.x0,
        U_init=jnp.zeros((30, 3), dtype=jnp.float32),
        params=params,
        max_iter=3,
    )
    assert not jnp.any(jnp.isnan(xbar))
    final_hand = env.e(xbar[-1])
    err = float(jnp.linalg.norm(final_hand - env.target))
    assert err < 0.005, f"Final reach error too large: {err*1000:.2f} mm"


def test_cartesian_to_joint_3d():
    env = NonlinearReaching3D()
    pos = np.array([[0.34, 0.02, 0.18], [0.36, 0.04, 0.20]], dtype=np.float32)
    states = cartesian_to_joint_3d(pos, l1=env.l1, l2=env.l2, dt=env.dt)
    assert states.shape == (2, 6)
    assert not np.any(np.isnan(states))


def test_analytical_prediction_3d():
    env = NonlinearReaching3D()
    params = env.get_params_type()()
    predictor = MovingWindowMotionPredictor(env=env, params=params)
    predictor.warmup(future_steps=20, mode="analytical", max_iter=2)

    # Make observed trajectory prefix
    obs = np.tile(np.array(env.x0), (15, 1))
    result = predictor.predict(observed=obs, future_steps=20, mode="analytical", max_iter=2)

    assert result.mean.shape == (20, 6)
    assert result.cartesian_mean.shape == (20, 3)
    assert result.cartesian_cov.shape == (20, 3, 3)
    assert not np.any(np.isnan(result.cartesian_mean))
    assert result.latency_ms > 0.0
