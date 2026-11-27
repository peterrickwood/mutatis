"""Running, storing, loading and comparing versions."""

import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.spec import edits as e
from mutatis.versions import run as runs
from tests.infer.test_routes import PARAMS
from tests.infer.test_regression import FLAT
from mutatis.compile.system import compile_model
from mutatis.quantities.simulate import simulate_data
from mutatis.infer.posterior import Posterior
from mutatis.versions.pipeline import prepare, solver
from tests.spec.fixtures import toy
import jax


def flat_var_and_data():
    model = toy("two_variable")
    for edit in FLAT:
        model = edit.apply(model)
    system = compile_model(model)
    params = {k: jnp.asarray(v, dtype=float) for k, v in PARAMS["two_variable"].items()}
    dataset = simulate_data(system, solver(system)(params), jax.random.key(0), 120, start_state=np.zeros(2))
    return model, dataset


def test_run_store_reload_and_cache(tmp_path):
    model, dataset = flat_var_and_data()
    first = runs.run(model, dataset, store=tmp_path, n_draws=500)
    assert first.run["method"] == "exact_regression" and first.run["route"] == "read_off"
    assert {p.name for p in first.folder.iterdir()} == {"spec.yaml", "run.json", "posterior.npz", "data.npz", "summary.json", "quantities.npz"}
    again = runs.run(model, dataset, store=tmp_path, n_draws=500)
    assert again.folder == first.folder and again.summary == first.summary   # the same inputs give the stored result
    other_seed = runs.run(model, dataset, store=tmp_path, n_draws=500, seed=1)
    assert other_seed.folder != first.folder
    assert runs.load_result(first.folder).model == first.model


def test_a_run_keeps_its_data_and_rebuilds_its_problem(tmp_path):
    """Stored through `save_result` with prior draws in place of a posterior, so no sampler runs."""
    _, dataset = flat_var_and_data()
    model = toy("two_variable")
    for values in dataset.series.values():
        values[[3, 7]] = [np.nan, -np.nan]                 # missing values, of either sign, survive exactly
    problem = prepare(model, dataset)
    draws = np.asarray(jax.vmap(problem.draw_prior)(jax.random.split(jax.random.key(0), 50)))
    posterior = Posterior(draws, np.full(50, 1 / 50), None, "prior")
    settings = {"route": problem.route, "start": problem.start, "method": "prior"}
    result = runs.load_result(runs.save_result(tmp_path / "prior", model, [], dataset, problem, posterior, settings, seed=0).folder)
    stored = runs.load_data(result)
    assert stored.dates == dataset.dates and stored.series.keys() == dataset.series.keys()
    assert all(np.array_equal(stored.series[k], v, equal_nan=True) for k, v in dataset.series.items())
    problem = runs.problem_of(result, stored)
    assert problem.route == "kalman" and problem.start == result.run["settings"]["start"]
    posterior = runs.load_posterior(result)
    assert posterior.particles.shape == (50, problem.size) and np.isclose(posterior.weights.sum(), 1.0)
    np.save(result.folder / "values.npy", np.zeros(1))     # a stray file is harmless; altered data are refused
    with np.load(result.folder / "data.npz") as original:
        np.savez(result.folder / "data.npz", dates=original["dates"], names=original["names"], values=original["values"] + 1.0)
    with pytest.raises(ValueError, match="fingerprint"):
        runs.load_data(result)
    (result.folder / "data.npz").unlink()
    assert runs.load_data(result) is None                  # stored before runs kept their data


def test_compare_lists_the_edits_and_the_changes(tmp_path):
    model, dataset = flat_var_and_data()
    base = runs.run(model, dataset, store=tmp_path, n_draws=500)
    restricted = runs.run(model, dataset, store=tmp_path, n_draws=500, edits=[e.ReplaceEquation("dynamics", 0, "y = c + B @ y(-1) + u")])
    assert restricted.folder == base.folder   # the same equation text: the same model, so the same stored result
    changed = runs.run(model, dataset, store=tmp_path, n_draws=500, edits=[e.SetPrior("dynamics.Sigma", prior="inverse_wishart(df=10, scale=identity)")])
    text = runs.compare(base, changed)
    assert '"set_prior": {"target": "dynamics.Sigma", "prior": "inverse_wishart(df=10, scale=identity)"}' in text
    assert "dynamics.B[a, b]" in text and "not defined" in text
