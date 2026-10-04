"""Exact ZOH discretization, max-ent policy, filtered Kalman gains, the online solver with early stopping, the
per-window IOC likelihoods (fully / partially observed, nested) and the RTS smoother of the training states."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from prophet_ioc.belief import kf
from prophet_ioc.control import ilqr_unrolled
from prophet_ioc.control.policy import maxent_noise_factor
from prophet_ioc.data.cari import rts_smooth
from prophet_ioc.envs.human_kinematic_reaching import HumanKinematicParams, HumanKinematicReaching
from prophet_ioc.envs.zoh import zoh_coefficients, zoh_noise_cov, zoh_state_matrices
from prophet_ioc.infer.gaussian import psd_projection
from prophet_ioc.infer.multi_env import simulate_trial, trial_loglikelihood


def _expm_step(dt, b):
    Phi = expm(np.array([[0.0, 1.0, 0.0], [0.0, -b, 1.0], [0.0, 0.0, 0.0]]) * dt)
    return Phi[1, 1], Phi[1, 2], Phi[0, 2], Phi[0, 1]   # e, a1 (u -> qd), a2 (u -> q), a1 (qd -> q)


def test_zoh_matches_matrix_exponential():
    for dt in (0.01, 0.05):
        for b in (0.0, 1e-6, 0.2, 1.5, 3.0, 50.0):   # series branch (b dt < 0.1) and closed form
            e, a1, a2 = (float(v) for v in zoh_coefficients(dt, b, xp=np))
            e_r, a1_r, a2_r, a1q_r = _expm_step(dt, b)
            np.testing.assert_allclose([e, a1, a2, a1], [e_r, a1_r, a2_r, a1q_r], rtol=1e-9, atol=1e-15)
        A, B = zoh_state_matrices(dt, 0.0, 2, xp=np)
        np.testing.assert_allclose(B[:, 0], [dt ** 2 / 2, 0.0, dt, 0.0])
        np.testing.assert_allclose(A[0, 2], dt)


def test_zoh_continuity_at_zero_damping():
    dt = 0.05
    z_edge = 0.1 / dt   # b dt = 0.1: switch between the series and the closed form
    for b0 in (0.0, z_edge):
        lo = np.array(zoh_coefficients(dt, b0 * (1 - 1e-7), xp=np))
        hi = np.array(zoh_coefficients(dt, b0 * (1 + 1e-7) + 1e-12, xp=np))
        np.testing.assert_allclose(lo, hi, rtol=1e-6)
    # float32 and gradients w.r.t. b are finite at b = 0 (no division by zero in either branch)
    g = jax.jacobian(lambda b: jnp.stack(zoh_coefficients(jnp.float32(dt), b)))(jnp.float32(0.0))
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(g, [-dt, -dt ** 2 / 2, -dt ** 3 / 6], rtol=1e-4)   # d/db at 0


def test_zoh_noise_covariance_van_loan():
    """M(dt) = int_0^dt e^{A s} G G^T e^{A^T s} ds for the double integrator (Van Loan's method)."""
    dt = 0.04
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    G = np.array([[0.0], [1.0]])
    F = expm(np.block([[-A, G @ G.T], [np.zeros((2, 2)), A.T]]) * dt)
    Q = F[2:, 2:].T @ F[:2, 2:]
    np.testing.assert_allclose(zoh_noise_cov(dt, xp=np), Q, rtol=1e-10)


def test_maxent_noise_factor():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    H = jnp.asarray(M @ M.T + 4 * np.eye(4))
    Gamma = maxent_noise_factor(H, 1e-3)
    np.testing.assert_allclose(Gamma @ Gamma.T, 1e-3 * np.linalg.inv(H), rtol=1e-4, atol=1e-9)


def test_filtered_gains_exact_observations():
    """Filtered form: with (almost) exact observations of the whole state K -> I; with noisy ones K < I."""
    T, d = 4, 2
    A = jnp.broadcast_to(jnp.array([[1.0, 0.1], [0.0, 1.0]]), (T, d, d))
    V = jnp.broadcast_to(0.1 * jnp.eye(d), (T, d, d))
    F = jnp.broadcast_to(jnp.eye(d), (T, d, d))
    K_exact = kf.forward_filtered(A, V, F, jnp.broadcast_to(1e-6 * jnp.eye(d), (T, d, d)), jnp.eye(d))
    np.testing.assert_allclose(K_exact, np.broadcast_to(np.eye(d), (T, d, d)), atol=1e-6)
    K_noisy = kf.forward_filtered(A, V, F, jnp.broadcast_to(1.0 * jnp.eye(d), (T, d, d)), jnp.eye(d))
    assert np.all(np.linalg.eigvals(np.array(K_noisy[-1])).real < 0.5)


def test_online_solver_early_stopping():
    env = HumanKinematicReaching(dt=0.05, target=jnp.array([0.3, -0.2, 1.0]))
    p = HumanKinematicParams()
    U0 = jnp.zeros((8, 19))
    solve = jax.jit(ilqr_unrolled.solve, static_argnames=("max_iter", "tol"))
    _, X3, _ = solve(env, env.x0, U0, p, max_iter=3)
    X3_ref, _ = ilqr_unrolled.ilqr_unrolled(env, env.x0, U0, p, 3)
    np.testing.assert_allclose(X3, X3_ref, atol=1e-4)            # tol=None: exactly max_iter iterations
    _, X1, _ = solve(env, env.x0, U0, p, max_iter=1)
    _, Xs, _ = solve(env, env.x0, U0, p, max_iter=3, tol=10.0)   # any improvement < 10x the cost: stop after one
    np.testing.assert_allclose(Xs, X1, atol=1e-4)
    _, Xt, _ = solve(env, env.x0, U0, p, max_iter=20, tol=1e-6)
    cost = lambda X: float(env._final_cost(X[-1], p))
    assert cost(Xt) <= cost(X3) + 1e-9


def _synthetic(T=4, seed=0, dtype=jnp.float32):
    env = HumanKinematicReaching(dt=0.06, target=jnp.array([0.35, -0.15, 1.05]))
    env = jax.tree.map(lambda a: jnp.asarray(a, dtype), env)
    p = HumanKinematicParams(running_vel_cost=1e-4, motor_noise_add=0.05)
    x = simulate_trial(env, p, jax.random.PRNGKey(seed), T, solve_iters=4)
    return env, p, x


def test_psd_projection_derivative():
    """The PSD projection of the belief covariance is differentiable through repeated eigenvalues (eigh's derivative
    is NaN there) and its derivative matches finite differences on an indefinite matrix (float64)."""
    with jax.enable_x64(True):
        k = jax.random.split(jax.random.PRNGKey(0), 3)
        A = jax.random.normal(k[0], (6, 6))
        jax.test_util.check_grads(psd_projection, (A + A.T,), order=1, modes=["fwd", "rev"])
        B = jax.random.normal(k[1], (6, 3))
        P = B @ B.T + jnp.eye(6)                                   # eigenvalue 1 repeated 3 times
        dS = jax.random.normal(k[2], (6, 6))
        val, tan = jax.jvp(psd_projection, (P,), (dS + dS.T,))
        np.testing.assert_allclose(val, P, atol=1e-12)
        np.testing.assert_allclose(tan, dS + dS.T, atol=1e-10)   # identity on the PD cone


def test_likelihoods_finite_with_finite_gradients():
    env, p, x = _synthetic()
    names = ("w_act_reach_arm", "running_vel_cost", "velocity_cost")

    def ll(theta, observability):
        params = p._replace(**{k: 10.0 ** theta[i] for i, k in enumerate(names)}, obs_noise=0.1)
        return trial_loglikelihood(env, x, params, solve_iters=3, observability=observability)

    theta = jnp.log10(jnp.array([getattr(p, k) for k in names]))
    for obs in ("full", "partial"):
        val, grad = jax.jit(jax.value_and_grad(lambda th: ll(th, obs)))(theta)
        assert np.isfinite(val) and np.all(np.isfinite(grad)), (obs, val, grad)
        assert np.any(np.abs(grad) > 0)


def test_partial_likelihood_approaches_full():
    """Nested models: the partially observed likelihood (filtered EKF belief) tends to the fully observed one as the
    observation noise vanishes (float64)."""
    with jax.enable_x64(True):
        env, p, x = _synthetic(T=4, seed=1, dtype=jnp.float64)
        f = jax.jit(lambda params, obs: trial_loglikelihood(env, x, params, solve_iters=3, observability=obs),
                    static_argnums=1)
        ll_full = float(f(p, "full"))
        gaps = [abs(float(f(p._replace(obs_noise=s), "partial")) - ll_full) for s in (1e-1, 1e-3, 1e-5)]
    assert np.isfinite(ll_full)
    assert gaps[2] < gaps[1] < gaps[0], gaps
    assert gaps[2] < 1e-3 * max(abs(ll_full), 1.0), (gaps, ll_full)


def test_rts_smoother_recovers_velocities():
    """On trajectories simulated from the model's ZOH random-acceleration dynamics, the RTS smoother recovers the
    velocities far better than finite differences of the noisy positions."""
    rng = np.random.default_rng(4)
    dt, N, m, sa, so = 0.01, 400, 5, 2.0, 0.005
    A, _ = zoh_state_matrices(dt, 0.0, 1, xp=np)
    L = np.linalg.cholesky(sa ** 2 * zoh_noise_cov(dt, xp=np))
    s = np.zeros((m, 2))
    q, qd = [], []
    for _ in range(N):
        q.append(s[:, 0].copy())
        qd.append(s[:, 1].copy())
        s = s @ A.T + rng.standard_normal((m, 2)) @ L.T
    q, qd = np.array(q), np.array(qd)
    y = q + so * rng.standard_normal(q.shape)
    q_s, qd_s = rts_smooth(y, dt, sa, so)
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    err_rts = rms(qd_s - qd)
    err_fd = rms(np.gradient(y, dt, axis=0) - qd)
    assert err_rts < 0.15 * rms(qd), (err_rts, rms(qd))
    assert err_rts < 0.5 * err_fd, (err_rts, err_fd)
    assert rms(q_s - q) < so
    # damping: the same model with b > 0
    q_b, qd_b = rts_smooth(y, dt, sa, so, damping=0.5)
    assert np.all(np.isfinite(qd_b))


def test_synthetic_parameter_recovery():
    """Reaches simulated from the model with known weights: maximizing the fully observed per-window likelihood from a
    start one decade off moves the weights to the true ones (evaluation of the full setting: see README)."""
    from prophet_ioc.envs.human_kinematic_reaching import stack_envs
    from prophet_ioc.infer import MultiTrialLikelihood

    true = HumanKinematicParams(running_vel_cost=1e-4, motor_noise_add=0.05)
    names = ("w_act_reach_arm", "running_vel_cost", "velocity_cost")
    rng = np.random.default_rng(0)
    T, n = 6, 4
    sim = jax.jit(lambda env, key: simulate_trial(env, true, key, T, solve_iters=5))
    envs, xs = [], []
    for k in range(n):
        tgt = np.array([0.35, -0.15, 1.05]) + rng.uniform(-0.12, 0.12, 3)
        envs.append(HumanKinematicReaching(dt=0.07, target=jnp.asarray(tgt, dtype=jnp.float32)))
        xs.append(np.array(sim(envs[-1], jax.random.PRNGKey(k))))
    ioc = MultiTrialLikelihood([(stack_envs(envs), jnp.asarray(np.stack(xs)))], true, names, solve_iters=5)

    def loss(theta, groups):   # data as an argument, not a constant of the compiled function
        return -ioc.loglikelihood(None, true._replace(**{k: 10.0 ** theta[i] for i, k in enumerate(names)}),
                                  groups) / n

    vg = jax.jit(jax.value_and_grad(loss))
    theta_true = np.log10([getattr(true, k) for k in names])
    theta = jnp.asarray(theta_true + np.array([1.0, -1.0, 1.0]))
    m = v = jnp.zeros_like(theta)
    best_val, best = np.inf, None
    for it in range(1, 61):
        val, g = vg(theta, ioc.groups)
        assert np.isfinite(val) and np.all(np.isfinite(g))
        if float(val) < best_val:
            best_val, best = float(val), np.array(theta)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g ** 2
        theta = theta - 0.1 * (m / (1 - 0.9 ** it)) / (jnp.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    err = np.abs(best - theta_true)
    assert np.all(err < 0.3), err                       # from 1 decade to within 0.3 decades of every true weight
    assert best_val <= float(vg(jnp.asarray(theta_true), ioc.groups)[0]) + 1.0
