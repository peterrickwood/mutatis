"""Stepping a model forward through time."""

from collections.abc import Callable

import jax
from jaxtyping import PyTree

from mutatis import _jax_config  # noqa: F401


def simulate(step: Callable[[PyTree, PyTree], PyTree], state0: PyTree, inputs: PyTree) -> PyTree:
    """Apply `step(state, input) -> next state` once per period. `inputs` (shocks, outside variables) has time as its leading
    axis. Returns the state after each step, also with time as the leading axis; the starting state is not included."""
    def body(state, period_input):
        following = step(state, period_input)
        return following, following

    _, path = jax.lax.scan(body, state0, inputs)
    return path
