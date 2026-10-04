"""Tests of the learnable HumanKinematicReaching parameters, its pytree registration, the multi-trial IOC
likelihood (prophet_ioc.infer.multi_env) and the handover Kalman filter (cari_kinematic)."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from prophet_ioc.envs.human_kinematic_reaching import (HumanKinematicParams, HumanKinematicReaching, learnable_params,
                                                       params_from_config, stack_envs)
from prophet_ioc.envs.zoh import zoh_coefficients, zoh_noise_cov
from prophet_ioc.infer import MultiTrialLikelihood, trial_loglikelihood


def _random_state(env, rng, scale=0.1):
    return env.x0 + scale * jnp.asarray(rng.standard_normal(env.state_shape[0]), dtype=jnp.float32)


def test_effort_weights_per_group():
    """One effort weight per joint group; the arm groups follow the reaching hand (also traced, reaching_hand any)."""
    p = HumanKinematicParams(w_act_pelvis=1.0, w_act_trunk=2.0, w_act_spine=3.0, w_act_reach_arm=4.0,
                             w_act_passive_arm=5.0, w_act_head=6.0, w_act_legs=7.0)
    expected = {"right": [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4 + [5] * 4 + [6] * 2,
                "left": [1] * 3 + [2] * 3 + [3] * 3 + [5] * 4 + [4] * 4 + [6] * 2}
    for hand in ("right", "left"):
        assert np.allclose(HumanKinematicReaching(reaching_hand=hand).action_weights(p), expected[hand])
        env_any = HumanKinematicReaching(reaching_hand="any")
        env_any.right_hand = jnp.float32(1.0 if hand == "right" else 0.0)
        assert np.allclose(env_any.action_weights(p), expected[hand])
    full = HumanKinematicReaching(mode="full_body").action_weights(p)
    assert full.shape == (27,) and np.allclose(full[19:], 7.0)
    # effort term of the running cost: 0.5 sum_j w_g(j) u_j^2 (everything else zero at the reference, at rest)
    env = HumanKinematicReaching()
    u = jnp.linspace(-1.0, 1.0, 19)
    p0 = p._replace(velocity_floor=0.0, w_lim=0.0)
    assert np.isclose(env._cost(env.x0, u, p0), 0.5 * np.sum(np.array(expected["right"]) * np.array(u) ** 2),
                      rtol=1e-5)


def test_cost_terms_and_dt_scaling():
    """Pelvis displacement from q0, velocity floor, no posture term; dt-scaled running cost."""
    rng = np.random.default_rng(1)
    params = HumanKinematicParams(w_lim=0.0)
    plain = HumanKinematicReaching(dt=0.02)
    scaled = HumanKinematicReaching(dt=0.02, dt_scaled_cost=True, dt_ref=0.05)
    x, u = _random_state(plain, rng), jnp.ones(19, dtype=jnp.float32)
    assert np.allclose(scaled._cost(x, u, params), plain._cost(x, u, params) * 0.02 / 0.05, rtol=1e-6)
    # joints moved away from q0 (no posture term), pelvis at q0, at rest, no action: zero running cost
    x_moved = jnp.concatenate([plain.q0.at[3:].add(0.2), jnp.zeros(19)])
    assert np.isclose(plain._cost(x_moved, jnp.zeros(19), params), 0.0, atol=1e-9)
    # pelvis displacement: 0.5 * base_disp_cost * |dp|^2, removed with the switch
    x_pelvis = jnp.concatenate([plain.q0.at[0].add(0.1), jnp.zeros(19)])
    assert np.isclose(plain._cost(x_pelvis, jnp.zeros(19), params), 0.5 * params.base_disp_cost * 0.01, rtol=1e-4)
    off = params_from_config({"pelvis_displacement_cost": False, "joint_limit_cost": False})
    assert off.base_disp_cost == 0.0 and off.w_lim == 0.0
    assert "base_disp_cost" not in learnable_params({"pelvis_displacement_cost": False})
    assert "base_disp_cost" in learnable_params({}) and "w_act_legs" not in learnable_params({})
    # velocity floor: max(running_vel_cost, velocity_floor)
    qd = jnp.concatenate([plain.q0, 0.5 * jnp.ones(19)])
    c0 = plain._cost(qd, jnp.zeros(19), params._replace(running_vel_cost=0.0))
    assert np.isclose(c0, 0.5 * params.velocity_floor * 19 * 0.25, rtol=1e-4)


def test_joint_limit_penalty():
    """Zero inside the limits, 0.5 * w_lim * excess^2 outside (also for the chest rotation vector norm), PSD Hessian."""
    from prophet_ioc.envs.human_kinematic_reaching import joint_limits
    env = HumanKinematicReaching()
    p = HumanKinematicParams(chest_rot_limit=0.5)
    lo, hi = joint_limits()
    q_in = env.q0
    assert np.all((q_in[6:] > lo[6:]) & (q_in[6:] < hi[6:]))
    assert float(env.joint_limit_penalty(q_in, p)) == 0.0
    q_out = q_in.at[12].set(hi[12] + 0.1).at[17].set(lo[17] - 0.2).at[3:6].set(jnp.array([0.6, 0.0, 0.0]))
    assert np.isclose(env.joint_limit_penalty(q_out, p), 0.5 * (0.1 ** 2 + 0.2 ** 2 + 0.1 ** 2), rtol=1e-4)
    H = jax.hessian(lambda q: env.joint_limit_penalty(q, p))(q_out)
    assert np.all(np.isfinite(H)) and np.linalg.eigvalsh(np.array(H)).min() > -1e-5
    assert np.all(np.isfinite(jax.grad(lambda q: env.joint_limit_penalty(q, p))(q_in.at[3:6].set(0.0))))
    # in the running cost with weight w_lim
    x = jnp.concatenate([q_out, jnp.zeros(19)])
    base = HumanKinematicParams(chest_rot_limit=0.5, w_lim=0.0, velocity_floor=0.0)
    assert np.isclose(env._cost(x, jnp.zeros(19), base._replace(w_lim=2.0)) - env._cost(x, jnp.zeros(19), base),
                      2.0 * float(env.joint_limit_penalty(q_out, p)), rtol=1e-4)


def test_zoh_dynamics_and_noise_covariance():
    """_dynamics = exact ZOH of qdd = u - b qd; V V^T = (sigma_m^2 u^2 + sigma_add^2) M(dt) per joint."""
    from scipy.linalg import expm
    rng = np.random.default_rng(3)
    env = HumanKinematicReaching(dt=0.04)
    x, u = _random_state(env, rng), jnp.asarray(rng.standard_normal(19), dtype=jnp.float32)
    for b in (0.0, 0.7):
        p = HumanKinematicParams(damping=b)
        Phi = expm(np.array([[0, 1, 0], [0, -b, 1], [0, 0, 0]]) * 0.04)
        expected = np.concatenate([Phi[0, 0] * x[:19] + Phi[0, 1] * x[19:] + Phi[0, 2] * u,
                                   Phi[1, 1] * x[19:] + Phi[1, 2] * u])
        assert np.allclose(env._dynamics(x, u, jnp.zeros(76), p), expected, atol=1e-6)
    p = HumanKinematicParams(motor_noise=0.3, motor_noise_add=0.05)
    V = np.array(jax.jacobian(env._dynamics, argnums=2)(x, u, jnp.zeros(76), p), dtype=np.float64)
    S = V @ V.T
    M = np.array(zoh_noise_cov(0.04, xp=np))
    for j in (0, 9, 18):
        idx = [j, 19 + j]
        s2 = 0.3 ** 2 * float(u[j]) ** 2 + 0.05 ** 2
        assert np.allclose(S[np.ix_(idx, idx)], s2 * M, rtol=1e-4, atol=1e-12)
    assert np.allclose(S[0, 1], 0.0) and np.allclose(S[0, 20], 0.0)   # joints independent
    # the position block is no longer singular
    assert np.linalg.eigvalsh(S[:19, :19]).min() > 0


def test_env_pytree_roundtrip_and_stacking():
    env = HumanKinematicReaching(reaching_hand="left", dt=0.03, dt_scaled_cost=True, target=jnp.array([0.2, 0.3, 1.1]))
    leaves, treedef = jax.tree_util.tree_flatten(env)
    env2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert env2.reaching_hand == "left" and env2.dt_scaled_cost and env2.state_shape == (38,)
    assert env2.state_noise_shape == (76,)
    x = env.x0 + 0.1
    assert np.allclose(env2.e(x), env.e(x))
    batch = stack_envs([env, HumanKinematicReaching(reaching_hand="left", dt=0.04, dt_scaled_cost=True)])
    assert batch.dt.shape == (2,) and batch.target.shape == (2, 3)
    assert np.allclose(jax.vmap(lambda e: e.e(e.x0))(batch)[0], env.e(env.x0), atol=1e-6)


def test_multi_trial_likelihood_is_sum_of_trials():
    rng = np.random.default_rng(2)
    T = 5
    params = HumanKinematicParams(running_vel_cost=1e-4, motor_noise_add=0.2, residual_noise=0.1)
    envs, xs = [], []
    for k in range(3):
        env = HumanKinematicReaching(dt=0.05, target=jnp.array([0.3, -0.2, 1.0 + 0.05 * k]))
        envs.append(env)
        xs.append(np.stack([np.asarray(_random_state(env, rng, 0.05)) for _ in range(T + 1)]))
    for linearization in ("solve", "data"):
        ioc = MultiTrialLikelihood([(stack_envs(envs), jnp.asarray(np.stack(xs)))], params,
                                   infer=("w_act_reach_arm",), linearization=linearization, solve_iters=3)
        total = float(ioc.loglikelihood(None, params))
        separate = sum(float(trial_loglikelihood(env, jnp.asarray(x), params, linearization=linearization,
                                                 solve_iters=3))
                       for env, x in zip(envs, xs))
        assert np.isfinite(total)
        assert np.isclose(total, separate, rtol=5e-4), (linearization, total, separate)
        # non-inferred parameters keep the fixed values, inferred ones are replaced
        full = ioc.full_params(HumanKinematicParams(w_act_reach_arm=3e-6))
        assert full.w_act_reach_arm == 3e-6 and full.motor_noise_add == 0.2


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
    params = HumanKinematicParams(running_vel_cost=1e-4, motor_noise_add=0.05)
    env = HumanKinematicReaching(dt=0.05, target=jnp.array([0.3, -0.2, 1.0]))
    X, _ = ilqr_unrolled(env, env.x0, jnp.zeros((T, 19)), params, 8)
    wrist = lambda e, s: e.e(s)
    # the model's own open-loop nominal is predicted exactly; a perturbed trajectory is not
    assert float(trial_open_loop_error(env, X, params, wrist)) < 1e-8
    assert float(trial_open_loop_error(env, X.at[:, 9:13].add(0.1), params, wrist)) > 1e-4
    # the likelihood-only residual noise adds variance: lower density of the exactly predicted transitions
    ll0 = float(trial_loglikelihood(env, X, params, solve_iters=8))
    ll1 = float(trial_loglikelihood(env, X, params._replace(residual_noise=0.5), solve_iters=8))
    assert np.isfinite(ll0) and ll1 < ll0
    # the velocity-block fallback scores fewer components
    llv = float(trial_loglikelihood(env, X, params, slice(19, 38), solve_iters=8))
    assert np.isfinite(llv) and llv != ll0
    # masks: only the weighted steps / transitions count
    mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    Xp = X.at[3:, 9:13].add(0.3)   # perturbed after the masked prefix only
    assert float(trial_open_loop_error(env, Xp, params, wrist, mask=mask)) < 1e-8
    ll_m = float(trial_loglikelihood(env, Xp, params, solve_iters=8, mask=mask))
    ll_m0 = float(trial_loglikelihood(env, X, params, solve_iters=8, mask=mask))
    assert np.isclose(ll_m, ll_m0, rtol=1e-4)
