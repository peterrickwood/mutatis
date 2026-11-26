"""Recovering the shocks and states implied by the data, for models whose series are all observed exactly (contract 1 only)."""

import jax
import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.solve.contract import SolvedModel


def recover_shocks(solved: SolvedModel, observed, outside, start_state):
    """The shocks and states that reproduce the observed data exactly: (shocks (periods, k), states (periods, n))."""
    form = solved.linear_form()
    ZQ = form.Z @ form.Q

    def period(state, inputs):
        y, z = inputs
        predicted = form.kappa + form.P @ state + form.R @ z
        shocks = jnp.linalg.solve(ZQ, y - form.d - form.Z @ predicted)
        following = predicted + form.Q @ shocks
        return following, (shocks, following)

    _, (shocks, states) = jax.lax.scan(period, start_state, (observed, outside))
    return shocks, states
