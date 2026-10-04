from abc import ABC, abstractmethod
from functools import partial
from typing import Any, NamedTuple, Tuple

from jax import jit, random, lax, vmap, numpy as jnp


class Env(ABC):
    # Regularization eps of Q_uu in the (g)LQR backward passes, Ht = H + max(0, eps - lambda_min(H)) I (Li's thesis
    # 5.4.1). It must match the scale of the cost: None keeps the default of the backward pass (lqr / glqr).
    reg_eps = None

    def __init__(self, state_shape, action_shape, observation_shape, state_noise_shape=None, obs_noise_shape=None):
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._observation_shape = observation_shape
        self._state_noise_noise_shape = state_noise_shape or state_shape
        self._obs_noise_shape = obs_noise_shape or observation_shape

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def action_shape(self):
        return self._action_shape

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def state_noise_shape(self):
        return self._state_noise_noise_shape

    @property
    def obs_noise_shape(self):
        return self._obs_noise_shape

    @abstractmethod
    def _dynamics(self, state, action, noise, params) -> jnp.ndarray:  
        ...

    @abstractmethod
    def _observation(self, state, noise, params) -> jnp.ndarray:
        ...

    @abstractmethod
    def _cost(self, state, action, params) -> jnp.ndarray:
        ...

    @abstractmethod
    def _final_cost(self, state, params) -> jnp.ndarray:
        ...

    @abstractmethod
    def _reset(self, noise, params) -> jnp.ndarray:
        ...

    @partial(jit, static_argnums=(0,))
    def reset(self, key, params):
        pro_noise_key, obs_noise_key, new_key = random.split(key, 3)
        new_state = self._reset(random.normal(pro_noise_key, shape=self.state_noise_shape), params)
        env_state = new_state, new_key
        return env_state, self._observation(new_state,
                                            random.normal(obs_noise_key, shape=self.obs_noise_shape), params)

    @partial(jit, static_argnums=(0,))
    def step(self, env_state, action, params):
        state, key = env_state  # unpack state

        cost = self._cost(state, action, params)

        # sample noises
        pro_noise_key, obs_noise_key, new_key = random.split(key, 3)
        pro_noise = random.normal(pro_noise_key, shape=self.state_noise_shape)
        obs_noise = random.normal(obs_noise_key, shape=self.obs_noise_shape)

        # apply dynamics
        new_state = self._dynamics(state, action, pro_noise, params)
        env_state = new_state, new_key

        return env_state, self._observation(new_state, obs_noise, params), cost

    def rollout(self, key, steps, policy, params) -> Tuple[jnp.ndarray, Any, Any, jnp.ndarray]:
        initial_state, initial_obsv = self.reset(key, params)

        def scan_body(carry, t):
            (state, key), obsv = carry

            # sample action
            action_noise_key, new_key = random.split(key, 2)
            acion_noise = random.normal(action_noise_key, shape=self.action_shape)
            action = policy(t, obsv, acion_noise)

            env_state, obsv, cost = self.step((state, new_key), action, params)
            return (env_state, obsv), (env_state, obsv, action, cost)

        (final_state, _), (env_state, obsv, actions, cost) = lax.scan(scan_body, (initial_state, initial_obsv),
                                                                      jnp.arange(steps))

        states = jnp.vstack([initial_state[0], env_state[0]])
        obsvs = jnp.vstack([initial_obsv, obsv])

        costs = jnp.hstack([cost, jnp.array([self._final_cost(final_state[0], params)])])
        return states, obsvs, actions, costs

    def simulate(self, key, steps, trials, policy, params):
        return vmap(lambda key: self.rollout(key, steps, policy, params))(random.split(key, trials))

    def trajectory_cost(self, x, u, params):
        T = x.shape[0] - 1
        final_cost = self._final_cost(x[T], params)
        step_cost = vmap(self._cost, in_axes=(0, 0, None))(x[:-1], u, params).sum()
        return final_cost + step_cost

    @staticmethod
    def get_params_type() -> type:
        ...

    @staticmethod
    def get_params_bounds() -> Tuple[NamedTuple, NamedTuple]:
        ...
