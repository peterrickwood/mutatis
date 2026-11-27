"""Results stores for the inspector's tests. Runs are stored through the results store's own `save_result`, with draws from the
prior standing in for a posterior, so no sampler runs and every toy is quick."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.compile.system import compile_model
from mutatis.infer.posterior import Posterior
from mutatis.quantities.simulate import simulate_data
from mutatis.spec.edits import apply_edits
from mutatis.versions import run as runs
from mutatis.versions.pipeline import prepare, solver
from tests.infer.test_routes import PARAMS
from tests.spec.fixtures import MODEL_FILES, toy

# Values to simulate each toy's data at: those the route tests use, and the real business cycle toy's from tests/models/rbc.mod.
TRUE = {**PARAMS, "rbc": {"delta": 0.025, "alpha": 0.33, "beta": 0.99, "technology.rho": 0.9, "technology.sigma": 0.01},
        "two_shock_forward_looking": {"demand.rho": 0.6, "demand.S": [[1.0, 0.3], [0.3, 0.5]], "demand.Q": [[1.0, 0.0], [0.0, 1.0]], "price.a": 0.5}}
TOYS = [p.stem for p in MODEL_FILES]


def toy_data(name: str, periods: int = 30):
    model = toy(name)
    system = compile_model(model)
    assert name in TRUE, f"give values to simulate the toy {name!r} at, in tests/web/conftest.py"
    solved = solver(system)({**system.fixed_parameters(), **{k: jnp.asarray(v, dtype=float) for k, v in TRUE[name].items()}})
    outside = np.cumsum(np.random.default_rng(0).normal(size=(periods, len(system.outside_labels))), axis=0) if system.outside_labels else None
    start = None if solved.initial_gaussian() is not None else np.zeros(system.n_variables)
    return simulate_data(system, solved, jax.random.key(0), periods, outside=outside, start_state=start)


def store_draws(folder, model, dataset, draws=None, edits=(), n: int = 300, seed: int = 0):
    """Store a run whose draws are `draws` (unconstrained), or `n` draws from the prior."""
    problem = prepare(apply_edits(model, list(edits)), dataset)
    if draws is None:
        draws = np.asarray(jax.vmap(problem.draw_prior)(jax.random.split(jax.random.key(seed), n)))
    posterior = Posterior(np.asarray(draws), np.full(len(draws), 1 / len(draws)), None, "prior draws")
    settings = {"route": problem.route, "start": problem.start, "method": "prior draws"}
    return runs.save_result(folder, model, list(edits), dataset, problem, posterior, settings, seed)


@pytest.fixture(scope="session")
def toy_store(tmp_path_factory):
    """A store with one run per toy, each holding draws from its prior."""
    store = tmp_path_factory.mktemp("toys")
    for name in TOYS:
        store_draws(store / name, toy(name), toy_data(name))
    return store
