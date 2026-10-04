"""Tests of the learnable HumanKinematicReaching parameters, its pytree registration, the multi-trial IOC
likelihood (prophet_ioc.infer.multi_env) and the handover Kalman filter (cari_kinematic)."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams, HumanKinematicReaching, stack_envs
from prophet_ioc.infer import MultiTrialInverseGILQR, trial_loglikelihood


def _random_state(env, rng, scale=0.1):
    return env.x0 + scale * jnp.asarray(rng.standard_normal(env.state_shape[0]), dtype=jnp.float32)


def test_default_params_reproduce_former_cost_and_noise():
    """With default parameters the running cost and the dynamics equal the former hard-coded formulas."""
    rng = np.random.default_rng(0)
    for hand in ("right", "left"):
        env = HumanKinematicReaching(reaching_hand=hand)
        params = HumanKinematicParams(running_vel_cost=1e-2)
        w_former = np.ones(19)
        w_former[0:3], w_former[3:9], w_former[17:19] = 20.0, 3.0, 2.0
        w_former[13:17] = 3.0 if hand == "right" else 1.0  # passive arm 3, reaching arm 1
        w_former[9:13] = 1.0 if hand == "right" else 3.0
        assert np.allclose(env.action_weights(params), w_former)

        x, u = _random_state(env, rng), jnp.asarray(rng.standard_normal(19), dtype=jnp.float32)
        q, qd = x[:19], x[19:]
        former = (0.5 * params.action_cost * jnp.sum(jnp.asarray(w_former) * u ** 2)
                  + 0.5 * params.base_disp_cost * jnp.sum((q[0:3] - env.q0[0:3]) ** 2)
                  + 0.5 * params.posture_cost * jnp.sum((q[3:] - env.q0[3:]) ** 2)
                  + 0.5 * params.running_vel_cost * jnp.sum(qd ** 2))
        assert np.allclose(env._cost(x, u, params), former, rtol=1e-6)

        noise = jnp.asarray(rng.standard_normal(38), dtype=jnp.float32)
        expected = jnp.concatenate([q + env.dt * qd, (1 - 0.2 * env.dt) * qd + env.dt * u
                                    + jnp.sqrt(env.dt) * params.motor_noise * u * noise[19:]])
        assert np.allclose(env._dynamics(x, u, noise, params), expected, atol=1e-6)


def test_dt_scaled_cost_and_posture_reference():
    rng = np.random.default_rng(1)
    params = HumanKinematicParams()
    ref = HumanKinematicReaching()._default_nominal_q0() + 0.05
    plain = HumanKinematicReaching(dt=0.02, q_posture_ref=ref)
    scaled = HumanKinematicReaching(dt=0.02, q_posture_ref=ref, dt_scaled_cost=True, dt_ref=0.05)
    x, u = _random_state(plain, rng), jnp.ones(19, dtype=jnp.float32)
    assert np.allclose(scaled._cost(x, u, params), plain._cost(x, u, params) * 0.02 / 0.05, rtol=1e-6)
    # at the posture reference with zero velocity and action the running cost vanishes
    x_ref = jnp.concatenate([ref, jnp.zeros(19)])
    assert np.isclose(plain._cost(x_ref, jnp.zeros(19), params), 0.0, atol=1e-7)


def test_env_pytree_roundtrip_and_stacking():
    env = HumanKinematicReaching(reaching_hand="left", dt=0.03, dt_scaled_cost=True, target=jnp.array([0.2, 0.3, 1.1]))
    leaves, treedef = jax.tree_util.tree_flatten(env)
    env2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert env2.reaching_hand == "left" and env2.dt_scaled_cost and env2.state_shape == (38,)
    x = env.x0 + 0.1
    assert np.allclose(env2.e(x), env.e(x))
    batch = stack_envs([env, HumanKinematicReaching(reaching_hand="left", dt=0.04, dt_scaled_cost=True)])
    assert batch.dt.shape == (2,) and batch.target.shape == (2, 3)
    assert np.allclose(jax.vmap(lambda e: e.e(e.x0))(batch)[0], env.e(env.x0), atol=1e-6)


def test_multi_trial_likelihood_is_sum_of_trials():
    rng = np.random.default_rng(2)
    T = 5
    params = HumanKinematicParams(running_vel_cost=1e-2, motor_noise_add=0.2)
    envs, xs = [], []
    for k in range(3):
        env = HumanKinematicReaching(dt=0.05, target=jnp.array([0.3, -0.2, 1.0 + 0.05 * k]))
        envs.append(env)
        xs.append(np.stack([np.asarray(_random_state(env, rng, 0.05)) for _ in range(T + 1)]))
    for linearization in ("solve", "data"):
        ioc = MultiTrialInverseGILQR([(stack_envs(envs), jnp.asarray(np.stack(xs)))], params,
                                     infer=("action_cost",), velocity_block=slice(19, 38), linearization=linearization,
                                     solve_iters=3)
        total = float(ioc.loglikelihood(None, params))
        separate = sum(float(trial_loglikelihood(env, jnp.asarray(x), params, slice(19, 38),
                                                 linearization=linearization, solve_iters=3))
                       for env, x in zip(envs, xs))
        assert np.isfinite(total)
        assert np.isclose(total, separate, rtol=5e-4), (linearization, total, separate)
        # non-inferred parameters keep the fixed values, inferred ones are replaced
        full = ioc.full_params(HumanKinematicParams(action_cost=3e-4))
        assert full.action_cost == 3e-4 and full.motor_noise_add == 0.2


def test_filtered_kalman_estimate():
    from cari_kinematic import kalman_filter_joint_history

    dt, n = 0.01, 40
    t = np.arange(n) * dt
    q = np.stack([0.5 * t, -1.0 * t, 0.2 + 0 * t], axis=1).astype(np.float32)
    x_filt, P = kalman_filter_joint_history(q, dt)
    # filtered estimate at the last observed frame (not the one-step prediction after it)
    assert np.abs(x_filt[:3] - q[-1]).max() < 2e-3
    assert np.allclose(x_filt[3:], [0.5, -1.0, 0.0], atol=0.1)
    assert np.all(np.linalg.eigvalsh(0.5 * (P + P.T)) > -1e-6)


def test_open_loop_error_and_residual_noise():
    from prophet_ioc.infer.multi_env import ilqr_unrolled, trial_open_loop_error

    T = 6
    params = HumanKinematicParams(running_vel_cost=1e-2)
    env = HumanKinematicReaching(dt=0.05, target=jnp.array([0.3, -0.2, 1.0]))
    X, _ = ilqr_unrolled(env, env.x0, jnp.zeros((T, 19)), params, 8)
    wrist = lambda e, s: e.e(s)
    # the model's own open-loop nominal is predicted exactly; a perturbed trajectory is not
    assert float(trial_open_loop_error(env, X, params, wrist)) < 1e-8
    assert float(trial_open_loop_error(env, X.at[:, 9:13].add(0.1), params, wrist)) > 1e-4
    # the likelihood-only residual noise adds variance: lower density of the exactly predicted transitions
    ll0 = float(trial_loglikelihood(env, X, params, slice(19, 38), solve_iters=8))
    ll1 = float(trial_loglikelihood(env, X, params._replace(residual_noise=0.1), slice(19, 38), solve_iters=8))
    assert np.isfinite(ll0) and ll1 < ll0
    # masks: only the weighted steps / transitions count
    mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    Xp = X.at[3:, 9:13].add(0.3)   # perturbed after the masked prefix only
    assert float(trial_open_loop_error(env, Xp, params, wrist, mask=mask)) < 1e-8
    ll_m = float(trial_loglikelihood(env, Xp, params, slice(19, 38), solve_iters=8, mask=mask))
    ll_m0 = float(trial_loglikelihood(env, X, params, slice(19, 38), solve_iters=8, mask=mask))
    assert np.isclose(ll_m, ll_m0, rtol=1e-4)
