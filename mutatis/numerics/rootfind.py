"""Finding where a function is zero: steady states, and identities solved within a period.

A thin layer over Optimistix's Newton method. Derivatives of the root with respect to anything the function depends on come
from the implicit function theorem, not from differentiating the iterations: dx = -J^-1 (d fn / d args) dargs, with J the
Jacobian of fn at the root. That rule is written here (`_root_jvp`) rather than left to Optimistix, so that a failure stays a
status: Optimistix's own rule raises an error when the root or its Jacobian is not finite, which stops a sampler whose draw
wanders where the model has no usable steady state. Failure is reported as a status, never raised.
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.status import Status


def find_root(fn: Callable[[PyTree, Any], PyTree], x0: PyTree, args: Any = None, rtol: float = 1e-10, atol: float = 1e-12,
              max_steps: int = 256) -> tuple[PyTree, jax.Array]:
    """A root of fn(x, args), starting from x0, and the outcome status. On failure (no convergence, or a root or Jacobian that is
    not finite) the status is NOT_CONVERGED and the value is a finite stand-in, x0 with any non-finite entry replaced by zero, so
    that what is computed from it stays finite; its derivative is then zero."""
    return _root(fn, rtol, atol, max_steps, x0, args)


def _finite(tree) -> jax.Array:
    return jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]))


def _residual(fn, value):
    """fn as a function of a flat vector, and that vector at `value`."""
    flat, unravel = ravel_pytree(value)
    return (lambda v, a: ravel_pytree(fn(unravel(v), a))[0]), flat, unravel


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _root(fn, rtol, atol, max_steps, x0, args):
    solution = optx.root_find(fn, optx.Newton(rtol=rtol, atol=atol), x0, args=args, max_steps=max_steps, throw=False)
    residual, flat, _ = _residual(fn, solution.value)
    ok = (solution.result == optx.RESULTS.successful) & _finite(solution.value) & jnp.all(jnp.isfinite(jax.jacfwd(residual)(flat, args)))
    stand_in = jax.tree_util.tree_map(lambda v: jnp.where(jnp.isfinite(v), v, 0.0), x0)
    value = jax.tree_util.tree_map(lambda v, s: jnp.where(ok, v, s), solution.value, stand_in)
    return value, jnp.where(ok, Status.OK, Status.NOT_CONVERGED)


@_root.defjvp
def _root_jvp(fn, rtol, atol, max_steps, primals, tangents):
    x0, args = primals
    _, dargs = tangents
    value, status = _root(fn, rtol, atol, max_steps, x0, args)
    residual, flat, unravel = _residual(fn, value)
    ok = status == Status.OK
    J = jax.jacfwd(residual)(flat, args)
    _, dr = jax.jvp(lambda a: residual(flat, a), (args,), (dargs,))
    J = jnp.where(ok, J, jnp.eye(flat.shape[0]))                          # never solve with a Jacobian from a failed search
    dx = jnp.where(ok, -jnp.linalg.solve(J, jnp.where(ok, dr, 0.0)), 0.0)
    return (value, status), (unravel(dx), np.zeros(jnp.shape(status), dtype=jax.dtypes.float0))
