from typing import Any, Tuple
import jax.numpy as jnp

from prophet_ioc import Env
from prophet_ioc.control import lqg, make_lqg_approx
from prophet_ioc.control.lqr import Gains


def solve(p: Env,
          X: jnp.array,
          U: jnp.array,
          Sigma0: jnp.ndarray,
          params: Any, lqg_module=lqg) -> Tuple[jnp.ndarray, Gains, jnp.ndarray, jnp.ndarray]:
    lqgspec = make_lqg_approx(p, params)(X, U)
    K, gains = lqg_module.solve(lqgspec, Sigma0=Sigma0)

    return K, gains, X, U
