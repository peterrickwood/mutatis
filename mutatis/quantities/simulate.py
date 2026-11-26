"""Simulating data from a solved model: shocks, states and observed series, including measurement errors."""

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.system import System
from mutatis.measure.data import Dataset, quarters
from mutatis.numerics.simulate import simulate
from mutatis.solve.contract import SolvedModel


def simulate_data(system: System, solved: SolvedModel, key, n_periods: int, first_date: str = "2000Q1", outside=None, start_state=None) -> Dataset:
    """A dataset drawn from the model. Outside variables, when the model has any, must be given (periods by outside variables).

    The starting state is drawn from the model's starting distribution unless given.
    """
    key_shocks, key_start, key_errors = jax.random.split(key, 3)
    m = len(system.outside_labels)
    if m and outside is None:
        raise ValueError("this model has outside variables: give their values")
    outside = jnp.zeros((n_periods, 0)) if outside is None else jnp.asarray(outside)
    state0 = solved.sample_initial(key_start) if start_state is None else jnp.asarray(start_state)
    shocks = solved.sample_shocks(key_shocks, n_periods)
    states = simulate(lambda s, inputs: solved.step(s, inputs[0], inputs[1]), state0, (shocks, outside))
    observed = np.array(jax.vmap(solved.observe)(states) + solved.sample_errors(key_errors, n_periods))  # a writable copy
    series = {o.series: observed[:, i] for i, o in enumerate(system.observations)}
    for info in system.outside.values():
        if info.calendar is None:
            positions = system.layout.outside[info.key].positions.reshape(-1)
            for position in positions:
                series[info.series[system.outside_labels[position]]] = np.array(outside[:, position])
    return Dataset(quarters(first_date, n_periods), series)
