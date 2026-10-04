import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax import vmap

from prophet_ioc.envs import Env
from prophet_ioc.control import gilqr, ilqr_fixed, glqr
from prophet_ioc.control.policy import create_lqr_policy
from prophet_ioc.infer.base import InverseOptimalControl
from prophet_ioc.infer.utils import estimate_controls


class InverseMaxEntBaseline(InverseOptimalControl):
    r"""Maximum Causal Entropy Inverse Reinforcement Learning Baseline.

    In Maximum Causal Entropy IRL (Ziebart et al., 2008, 2010), demonstrations are assumed
    to be sampled from a stochastic policy whose log-probability is proportional to the
    state-action value function (soft Q-function / Boltzmann policy):
        $$p(u_t \mid x_t) \propto \exp\left( -\frac{1}{\beta} Q(x_t, u_t) \right)$$
    where $\beta$ is the temperature parameter governing demonstrator rationality ($\beta \to 0$
    yields greedy deterministic behavior, while $\beta \to \infty$ yields uniform exploration).

    Under a second-order Laplace approximation of $Q(x_t, u_t)$ around the nominal optimal control
    $u_t^* = \pi_t(x_t)$:
        $$Q(x_t, u_t) \approx Q(x_t, u_t^*) + \frac{1}{2} (u_t - u_t^*)^\top \nabla_{uu}^2 Q(x_t, u_t^*) (u_t - u_t^*)$$
    where $H_t = \nabla_{uu}^2 Q_t = R_t + B_t^\top S_{t+1} B_t$ is the control Hessian matrix
    obtained directly from the Riccati backward pass of iLQR.

    The action likelihood simplifies to a local Gaussian distribution:
        $$u_t \mid x_t \sim \mathcal{N}\left( \pi_t(x_t),\, \beta H_t^{-1} \right)$$
    with log-likelihood:
        $$\log p(u_t \mid x_t) = -\frac{1}{2} (u_t - \pi_t(x_t))^\top \left(\frac{1}{\beta} H_t\right) (u_t - \pi_t(x_t)) + \frac{1}{2}\log\det\left(\frac{1}{\beta} H_t\right) - \frac{m}{2}\log(2\pi)$$

    Key difference with Inverse iLQG:
    - MaxEnt IRL models demonstrator stochasticity as action noise / sub-optimality around an open-loop
      or fully observed state sequence.
    - Inverse iLQG models partial observability: true physical deviations induce internal sensory updates
      and closed-loop feedback reactions through the observer-controller loop ($z_t = [x_t, \hat{x}_t]$).
    """

    def __init__(self, env: Env, maxent_temp: float = 1e-6, max_iter: int = 10, *args, **kwargs):
        self.env = env
        self.solve = gilqr.solve
        self.maxent_temp = maxent_temp
        self.max_iter = max_iter

    def apply_solver(self, x, u, params):
        r"""Solves forward Generalized iLQR from the mean initial state."""
        T = x.shape[1] - 1

        gains, xbar, ubar = self.solve(self.env, x0=x[:, 0].mean(axis=0),
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params, max_iter=self.max_iter)
        policy = create_lqr_policy(gains, xbar, ubar)

        return policy, gains

    def loglikelihood(self, x, params, u=None):
        r"""Evaluates average action log-likelihood under the Riccati Laplace approximation."""
        T = x.shape[1] - 1

        if u is None:
            # Reconstruct nominal controls u_t from demonstrated states x_t via inverse dynamics
            u = vmap(lambda xi: estimate_controls(xi, self.env, params))(x)

        # compute log likelihood of generated samples under the used controller
        def eval_llh(x, u):
            policy, gains = self.apply_solver(x, u, params)

            def eval_llh_t(t, x, u):
                u_policy = policy(t, x)
                # Covariance is beta * H^{-1}, where H = gains.H[t] is the control Hessian Q_uu
                llh = jstats.multivariate_normal.logpdf(u, u_policy, self.maxent_temp * jnp.linalg.inv(gains.H[t]))
                return llh

            llh = vmap(eval_llh_t)(jnp.arange(T), x[:-1], u)
            return llh

        llh = jnp.mean(vmap(eval_llh)(x, u))
        return llh


class FixedInverseMaxEntBaseline(InverseMaxEntBaseline):
    r"""Fast Fixed Linearization MaxEnt IRL Baseline.

    Instead of repeatedly resolving the forward iLQR problem across optimization iterations,
    `FixedInverseMaxEntBaseline` linearizes the nonlinear dynamics and quadratizes the candidate
    cost function directly around the demonstrated trajectory pairs $(X, U)$.

    A single backward Riccati pass (`ilqr_fixed.solve` with `glqr`) yields the control Hessians
    $H_t = Q_{uu}(t)$ and feedback policy $\pi_t(x)$, enabling rapid gradient-based parameter recovery.
    """

    def __init__(self, env: Env, maxent_temp: float = 1e-6, *args, **kwargs):
        super().__init__(env, maxent_temp, *args, **kwargs)
        self.solve = ilqr_fixed.solve

    def apply_solver(self, x, u, params):
        # Linearize candidate cost and dynamics along demonstration trajectories (X, U)
        gains, xbar, ubar = self.solve(self.env, X=x, U=u, params=params, lqr=glqr)
        policy = create_lqr_policy(gains, xbar, ubar)

        return policy, gains
