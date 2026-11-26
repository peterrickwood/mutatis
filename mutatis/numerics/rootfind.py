"""Finding where a function is zero: steady states, and identities solved within a period.

A thin layer over Optimistix's Newton method. Derivatives of the root with respect to anything the function depends on come
from the implicit function theorem, not from differentiating the iterations. Failure is reported as a status, never raised.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float, PyTree

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.status import Status


def find_root(fn: Callable[[PyTree, Any], PyTree], x0: PyTree, args: Any = None, rtol: float = 1e-10, atol: float = 1e-12,
              max_steps: int = 256) -> tuple[PyTree, jax.Array]:
    """A root of fn(x, args), starting from x0, and the outcome status. On failure the last iterate is returned with NOT_CONVERGED."""
    solution = optx.root_find(fn, optx.Newton(rtol=rtol, atol=atol), x0, args=args, max_steps=max_steps, throw=False)
    finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(solution.value)]))
    ok = (solution.result == optx.RESULTS.successful) & finite
    return solution.value, jnp.where(ok, Status.OK, Status.NOT_CONVERGED)
