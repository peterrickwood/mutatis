"""Stage 0 acceptance (slow tier): every toy goes through the same code, and an edit to each gives a comparable second version.

Run: .venv/bin/python -m pytest -m slow tests/test_stage0_acceptance.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.compile.system import compile_model
from mutatis.quantities.simulate import simulate_data
from mutatis.spec import edits as e
from mutatis.versions import run as runs
from mutatis.versions.pipeline import solver
from tests.infer.test_routes import PARAMS
from tests.spec.fixtures import toy

pytestmark = pytest.mark.slow

EDITS = {
    "hidden_level": e.SetPrior("y.sigma_eps", fixed=0.5),
    "two_variable": e.SetPrior("dynamics.B", prior="normal(0, 0.1)"),
    "forward_looking": e.SetExpectations("naive"),
    "error_correction": e.SetPrior("spending.beta1", fixed=1.0),
    "sectors": e.Untie("investment.phi", "m", prior="normal(1, 0.5)"),
}


def dataset_for(name, n_periods=150):
    system = compile_model(toy(name))
    params = {k: jnp.asarray(v, dtype=float) for k, v in PARAMS[name].items()}
    solved = solver(system)({**system.fixed_parameters(), **params})
    outside = np.cumsum(np.random.default_rng(0).normal(size=(n_periods, 1)), axis=0) if system.outside_labels else None
    start = None if solved.initial_gaussian() is not None else np.zeros(system.n_variables)
    return simulate_data(system, solved, jax.random.key(0), n_periods, outside=outside, start_state=start)


def flat_truth(name):
    out = {}
    for key, value in PARAMS[name].items():
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            out[key] = float(value)
        else:
            sets = toy(name).sets
            axes = compile_model(toy(name)).parameters[key].axes
            labels = runs._element_labels(key, axes, sets)
            out.update(zip(labels, value.reshape(-1).tolist()))
    return out


@pytest.mark.parametrize("name", list(EDITS))
def test_toy_through_the_generic_path_and_an_edit(name, tmp_path):
    dataset = dataset_for(name)
    base = runs.run(toy(name), dataset, store=tmp_path, n_particles=1000)
    for label, truth in flat_truth(name).items():
        entry = base.summary["parameters"][label]
        assert abs(entry["mean"] - truth) < 4 * entry["sd"], f"{name}: {label} posterior {entry['mean']:.3f} ({entry['sd']:.3f}) misses the truth {truth}"
    edited = runs.run(toy(name), dataset, store=tmp_path, n_particles=1000, edits=[EDITS[name]])
    text = runs.compare(base, edited)
    assert EDITS[name].op in text
    print(f"\n{text}")
