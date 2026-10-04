from abc import abstractmethod
from jax import numpy as jnp

class InverseOptimalControl:

    @abstractmethod
    def loglikelihood(self, x, params) -> jnp.ndarray:
        ...
