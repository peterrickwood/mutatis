"""Responses to shocks and forecasts, computed by stepping a solved model forward (contract 1 only).

A response is the difference between two paths from the same starting point: one hit by a shock of size one in the first period,
one not. For a linear model it does not depend on the starting point, and equals P^h Q in the solution's terms.
"""

import jax
import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.simulate import simulate
from mutatis.solve.contract import SolvedModel


def impulse_responses(solved: SolvedModel, n_states: int, n_shocks: int, n_outside: int, horizons: int, start=None):
    """Responses of every state variable to each shock, periods 0..horizons: shape (horizons + 1, states, shocks)."""
    start = jnp.zeros(n_states) if start is None else start
    outside = jnp.zeros((horizons + 1, n_outside))
    step = lambda state, inputs: solved.step(state, inputs[0], inputs[1])
    baseline = simulate(step, start, (jnp.zeros((horizons + 1, n_shocks)), outside))

    def one(j):
        shocks = jnp.zeros((horizons + 1, n_shocks)).at[0, j].set(1.0)
        return simulate(step, start, (shocks, outside)) - baseline

    return jnp.moveaxis(jax.vmap(one)(jnp.arange(n_shocks)), 0, -1)


def forecast(solved: SolvedModel, state, n_periods: int, n_shocks: int, outside=None, key=None, n_paths: int = 0):
    """Paths of the state from `state`: with no shocks (n_paths = 0), or `n_paths` paths with shocks drawn from the model."""
    outside = jnp.zeros((n_periods, 0)) if outside is None else jnp.asarray(outside)
    step = lambda s, inputs: solved.step(s, inputs[0], inputs[1])
    if n_paths == 0:
        return simulate(step, state, (jnp.zeros((n_periods, n_shocks)), outside))
    keys = jax.random.split(key, n_paths)
    return jax.vmap(lambda k: simulate(step, state, (solved.sample_shocks(k, n_periods), outside)))(keys)
