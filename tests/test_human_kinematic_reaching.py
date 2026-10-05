"""Tests for HumanKinematicReaching environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prophet_ioc.envs.human_kinematic_reaching import (
    HumanKinematicReaching,
    HumanKinematicParams,
    quat_from_rotvec,
    KEYPOINT_NAMES,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_ELBOW,
    KP_RIGHT_WRIST,
)
from prophet_ioc.control import gilqr


def test_quat_from_rotvec():
    """Verify analytical rotation vector to quaternion conversion."""
    w0 = jnp.zeros(3, dtype=jnp.float32)
    q0 = quat_from_rotvec(w0)
    assert np.allclose(q0, [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    wz = jnp.array([0.0, 0.0, 0.5 * np.pi], dtype=jnp.float32)
    qz = quat_from_rotvec(wz)
    expected_qz = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    assert np.allclose(qz, expected_qz, atol=1e-5)
    assert np.allclose(np.linalg.norm(qz), 1.0, atol=1e-6)

    jac = jax.jacobian(quat_from_rotvec)(w0)
    assert not np.any(np.isnan(jac))


def test_upper_body_initialization():
    """Verify Upper Body mode initialization (19 DOFs, 38 state dimensions)."""
    env = HumanKinematicReaching(mode="upper_body")
    assert env.n_dof == 19
    assert env.state_shape == (38,)
    assert env.action_shape == (19,)
    assert env.observation_shape == (38,)

    kpts = env.all_keypoints(env.x0)
    assert kpts.shape == (13, 3)
    assert not np.any(np.isnan(kpts))


def test_full_body_initialization():
    """Verify Full Body mode initialization (27 DOFs, 54 state dimensions)."""
    env = HumanKinematicReaching(mode="full_body")
    assert env.n_dof == 27
    assert env.state_shape == (54,)
    assert env.action_shape == (27,)
    assert env.observation_shape == (54,)

    kpts = env.all_keypoints(env.x0)
    assert kpts.shape == (13, 3)
    assert not np.any(np.isnan(kpts))


def test_rigid_bone_length_conservation():
    """Verify that human anatomical bone lengths are strictly conserved under arbitrary joint configurations."""
    env = HumanKinematicReaching(mode="upper_body")

    key = jax.random.PRNGKey(42)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        q_rand = env.q0 + 0.3 * jax.random.normal(subkey, shape=(env.n_dof,))
        kpts = env.all_keypoints(q_rand)

        r_shoulder = kpts[KP_RIGHT_SHOULDER]
        r_elbow = kpts[KP_RIGHT_ELBOW]
        r_wrist = kpts[KP_RIGHT_WRIST]

        upper_arm_len = float(np.linalg.norm(r_elbow - r_shoulder))
        forearm_len = float(np.linalg.norm(r_wrist - r_elbow))

        assert np.isclose(upper_arm_len, 0.30, atol=1e-4), f"Upper arm stretched: {upper_arm_len} m"
        assert np.isclose(forearm_len, 0.30, atol=1e-4), f"Forearm stretched: {forearm_len} m"


def test_task_jacobians_and_finite_differences():
    """Verify that analytical task Jacobians match numerical finite differences."""
    env = HumanKinematicReaching(mode="upper_body")
    x = env.x0

    J_hand = env.gamma(x)
    assert J_hand.shape == (3, 38)
    assert not np.any(np.isnan(J_hand))

    J_elbow = env.gamma_elbow(x)
    assert J_elbow.shape == (3, 38)
    assert not np.any(np.isnan(J_elbow))

    eps = 1e-4
    for j in [0, 1, 2, 9, 10, 12]:
        dx = np.zeros(38, dtype=np.float32)
        dx[j] = eps
        p_plus = np.array(env.e(x + dx))
        p_minus = np.array(env.e(x - dx))
        num_grad = (p_plus - p_minus) / (2.0 * eps)
        ana_grad = np.array(J_hand[:, j])
        assert np.allclose(num_grad, ana_grad, atol=1e-3), f"Jacobian mismatch at DOF {j}"


def test_ilqg_reaching_convergence():
    """Verify that gILQR successfully drives the upper body to reach a 3D target."""
    env = HumanKinematicReaching(mode="upper_body")
    params = HumanKinematicParams(motor_noise=0.1, obs_noise=1.0)   # default cost weights

    H = 25
    u_init = jnp.zeros((H, env.n_dof), dtype=jnp.float32)

    gains, xbar, ubar = gilqr.solve(p=env, x0=env.x0, U_init=u_init, params=params, max_iter=3)
    assert not np.any(np.isnan(xbar))
    assert not np.any(np.isnan(ubar))

    final_wrist = np.array(env.e(xbar[-1]))
    target = np.array(env.target)
    distance_error = float(np.linalg.norm(final_wrist - target))

    assert distance_error < 0.002, f"Reaching error too large: {distance_error * 1000.0:.2f} mm"


def test_pelvis_root_mode():
    """Verify pelvis-root mode is self-consistent and produces the same FK as chest-root mode."""
    import human_kinematic_model_jax as hkm_lib

    # Default body params: chest_hip_distance = 0.40 m
    body_params = np.array([0.30, 0.40, 0.25, 0.30, 0.30, 0.35, 0.40, 0.40], dtype=np.float32)

    # Chest-root env (legacy)
    env_chest = HumanKinematicReaching(mode="upper_body", body_params=body_params, root_joint="chest")
    # Pelvis-root env (new default)
    env_pelvis = HumanKinematicReaching(mode="upper_body", body_params=body_params, root_joint="pelvis")

    # --- 1. Verify that q0[0:3] is approximately at pelvis height in pelvis-root mode ---
    pelvis_default_height = float(env_pelvis.q0[2])
    assert abs(pelvis_default_height - 0.6) < 1e-5, f"Default pelvis height wrong: {pelvis_default_height}"

    # --- 2. Build a consistent state: chest at [0.1, 0.0, 1.0], upright orientation ---
    chest_pos = np.array([0.1, 0.0, 1.0], dtype=np.float32)
    # Identity chest quaternion
    chest_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    R_chest = np.array(hkm_lib.quat_to_rotmat(chest_quat))
    chest_hip_dist = float(body_params[1])

    # pelvis = chest - chest_hip_dist * R_chest[:, 2]  (with identity R_chest, R_chest[:,2] = [0,0,1])
    pelvis_pos = chest_pos - chest_hip_dist * R_chest[:, 2]  # = [0.1, 0.0, 0.6]

    # Build chest-root state
    q_chest_mode = env_chest.q0.at[0:3].set(jnp.array(chest_pos))
    x_chest = jnp.concatenate([q_chest_mode, jnp.zeros(19)])

    # Build pelvis-root state (same joint angles, q[0:3] = pelvis_pos)
    q_pelvis_mode = env_pelvis.q0.at[0:3].set(jnp.array(pelvis_pos))
    x_pelvis = jnp.concatenate([q_pelvis_mode, jnp.zeros(19)])

    # --- 3. Both modes should yield identical keypoints ---
    kpts_chest = np.array(env_chest.all_keypoints(x_chest))
    kpts_pelvis = np.array(env_pelvis.all_keypoints(x_pelvis))
    assert np.allclose(kpts_chest, kpts_pelvis, atol=1e-4), \
        f"Pelvis-root and chest-root keypoints differ!\n{kpts_chest}\nvs\n{kpts_pelvis}"

    # --- 4. env_pelvis.chest() must return chest position (not pelvis position) ---
    reported_chest = np.array(env_pelvis.chest(x_pelvis))
    assert np.allclose(reported_chest, chest_pos, atol=1e-4), \
        f"chest() in pelvis-root mode returned {reported_chest}, expected {chest_pos}"

    # --- 5. env_pelvis.pelvis() must return the pelvis position ---
    reported_pelvis = np.array(env_pelvis.pelvis(x_pelvis))
    expected_pelvis_fk = 0.5 * (kpts_pelvis[4] + kpts_pelvis[10])  # mean of left/right hips
    assert np.allclose(reported_pelvis, expected_pelvis_fk, atol=1e-4), \
        f"pelvis() mismatch: {reported_pelvis} vs {expected_pelvis_fk}"

    # --- 6. base_disp_cost should penalize pelvis (q[0:3]) not chest in pelvis-root mode ---
    # Verify the pelvis position is indeed lower than the chest position
    q28_pelvis = np.array(env_pelvis.build_q28(q_pelvis_mode))
    chest_from_q28 = q28_pelvis[0:3]
    assert float(chest_from_q28[2]) > float(pelvis_pos[2]) + 0.30, \
        f"Chest {chest_from_q28[2]:.3f} should be well above pelvis {pelvis_pos[2]:.3f}"

    print("\n  ✓ Pelvis-root mode: FK consistent with chest-root, chest()/pelvis() return correct positions.")


def test_pelvis_root_bone_length_conservation():
    """Verify bone lengths are strictly conserved under pelvis-root mode with random joint configurations."""
    env = HumanKinematicReaching(mode="upper_body", root_joint="pelvis")

    key = jax.random.PRNGKey(123)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        # Perturb only joint angles (q[3:]), keep pelvis position fixed
        q_rand = env.q0.at[3:].add(0.3 * jax.random.normal(subkey, shape=(16,)))
        kpts = env.all_keypoints(q_rand)

        r_shoulder = kpts[KP_RIGHT_SHOULDER]
        r_elbow = kpts[KP_RIGHT_ELBOW]
        r_wrist = kpts[KP_RIGHT_WRIST]

        upper_arm_len = float(np.linalg.norm(r_elbow - r_shoulder))
        forearm_len = float(np.linalg.norm(r_wrist - r_elbow))

        assert np.isclose(upper_arm_len, 0.30, atol=1e-4), f"Upper arm stretched: {upper_arm_len} m"
        assert np.isclose(forearm_len, 0.30, atol=1e-4), f"Forearm stretched: {forearm_len} m"


def _fast_path_case():
    """A reaching environment and a trajectory with both wrist goals, the running wrist term and joint limits active."""
    from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams
    env = HumanKinematicReaching(reaching_hand="both", dt=0.05, dt_scaled_cost=True)
    params = HumanKinematicParams(running_target_cost=0.06, velocity_cost=1e-3, running_vel_cost=1e-4,
                                  motor_noise=0.3, motor_noise_add=0.05)
    T = 8
    X = jnp.vstack([env.x0] * (T + 1)) + 0.05 * jax.random.normal(jax.random.PRNGKey(0), (T + 1, env.state_shape[0]))
    X = X.at[3, 3:6].set(jnp.array([0.9, 0.5, 0.3]))      # chest rotation beyond its limit
    X = X.at[:, 12].add(1.5)                               # an arm joint beyond its limit
    U = 0.5 * jax.random.normal(jax.random.PRNGKey(1), (T, env.action_shape[0]))
    return env, params, X, U


def test_fast_quadratization_matches_autodiff():
    """quadratize_cost (structure of the cost) = make_lqr_approx's jacfwd(grad), values and derivatives."""
    from prophet_ioc.control.spec import make_lqr_approx
    env, params, X, U = _fast_path_case()

    def spec(fast, X, p):
        HumanKinematicReaching.fast_quadratization = fast
        return make_lqr_approx(env, p)(X, U)

    try:
        ref, new = spec(False, X, params), spec(True, X, params)
        for name in ref._fields:
            a, b = np.asarray(getattr(ref, name)), np.asarray(getattr(new, name))
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(a).max()), err_msg=name)
        scalar = lambda fast, X, p: sum(jnp.sum(jnp.sin(getattr(spec(fast, X, p), f)))
                                        for f in ("Q", "q", "R", "r", "Qf", "qf"))
        gX0, gp0 = jax.grad(lambda X, p: scalar(False, X, p), argnums=(0, 1))(X, params)
        gX1, gp1 = jax.grad(lambda X, p: scalar(True, X, p), argnums=(0, 1))(X, params)
        np.testing.assert_allclose(gX1, gX0, rtol=1e-4, atol=1e-5 * float(jnp.abs(gX0).max()))
        for k in ("velocity_cost", "base_disp_cost", "running_target_cost", "w_act_reach_arm", "w_lim"):
            np.testing.assert_allclose(getattr(gp1, k), getattr(gp0, k), rtol=1e-3, atol=1e-8, err_msg=k)
    finally:
        HumanKinematicReaching.fast_quadratization = True


def test_joint_signal_noise_backward_matches_generic():
    """glqr.backward_joint_signal_noise (closed-form noise terms) = glqr.backward on the full LQG spec."""
    from prophet_ioc.control import ilqr_unrolled
    env, params, X, U = _fast_path_case()

    def gains(flag):
        HumanKinematicReaching.joint_signal_noise = flag
        return ilqr_unrolled.backward(env, X, U, params)

    try:
        ref, new = gains(False), gains(True)
        for k in ("L", "l", "H"):
            a, b = np.asarray(getattr(ref, k)), np.asarray(getattr(new, k))
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(a).max()), err_msg=k)
        obj = lambda flag, p: jnp.sum(gains(flag).L ** 2) + jnp.sum(gains(flag).l ** 2)
        g0 = jax.grad(lambda p: obj(False, p))(params)
        g1 = jax.grad(lambda p: obj(True, p))(params)
        for k in ("velocity_cost", "running_target_cost", "w_act_reach_arm", "motor_noise"):
            np.testing.assert_allclose(getattr(g1, k), getattr(g0, k), rtol=1e-3, atol=1e-8, err_msg=k)
    finally:
        HumanKinematicReaching.joint_signal_noise = True
