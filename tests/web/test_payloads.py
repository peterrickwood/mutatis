"""The inspector's payloads, built from stored runs: the specification, priors against posteriors, data against fit, reported
quantities and comparisons. Every toy goes through the same code."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.spec import edits as e
from mutatis.spec.yaml_io import load_model
from mutatis.versions import run as runs
from mutatis.versions.pipeline import prepare
from mutatis.web import estimates
from mutatis.web.compare import comparison
from mutatis.web.describe import describe
from tests.spec.fixtures import toy
from tests.web.conftest import TOYS, TRUE, store_draws, toy_data

ROTATED = """
model: rotated
sets:
  series: [a, b]
  shocks: [s1, s2]
blocks:
  dynamics:
    outputs: ["y[series]"]
    parameters:
      c:     {shape: [series], prior: flat}
      B:     {shape: [series, series], prior: flat}
      Sigma: {shape: [series, series], domain: positive_definite, prior: jeffreys}
      Q:     {shape: [series, shocks], domain: orthogonal, prior: uniform}
    shocks:
      "u[shocks]": "normal(0, 1)"
    equations:
      - "y = c + B @ y(-1) + chol(Sigma) @ Q @ u"
data:
  a: {observes: "y[a]"}
  b: {observes: "y[b]"}
restrictions:
  positive: {quantity: "response(y[a], u[s1], horizons=0)", require: ">= 0"}
"""
REPORT = ("response(y[b], u[s1], horizons=0..2)",)


@pytest.fixture(scope="module")
def rotated(tmp_path_factory):
    """A small rotated system with flat and Jeffreys priors, estimated by the exact formula: with the even prior on the rotation,
    and with no prior on it (ranges)."""
    store, data = tmp_path_factory.mktemp("rotated"), toy_data("two_variable", periods=60)
    base = runs.run(load_model(ROTATED), data, store=store, n_draws=100, report=REPORT)
    return base, runs.run(load_model(ROTATED), data, store=store, n_draws=100, report=REPORT, edits=[e.SetPrior("dynamics.Q", prior="none")])


def _loaded(folder):
    return estimates.load(runs.load_result(folder))


@pytest.mark.parametrize("name", TOYS)
def test_every_toy_gives_complete_payloads(name, toy_store):
    result = runs.load_result(toy_store / name)
    described = describe(result.model)
    assert all(eq["latex"] for b in described["blocks"] for eq in b["equations"])
    assert {n["id"] for n in described["graph"]["nodes"]} == {f"block:{b}" for b in result.model.blocks} | {f"data:{s}" for s in result.model.data}
    loaded = estimates.load(result)
    panels = estimates.parameter_panels(loaded)
    for entry in panels["parameters"]:
        assert entry["kind"] == "prior" and entry["prior_latex"]
        for element in entry["elements"]:
            lower, upper = element["window"]
            assert lower < upper and len(element["posterior"]) == estimates.BINS and element["mean"] is not None
            assert element["prior"]["density"] and all(d >= 0 for d in element["prior"]["density"])
            if type(loaded.problem.system.parameters[entry["key"]].domain).__name__ == "Positive":
                assert lower >= 0
    fit = estimates.fit_panels(loaded)
    system = loaded.problem.system
    assert [s["series"] for s in fit["observed"]] == [o.series for o in system.observations] and fit["draws"] > 0
    for column, series in enumerate(fit["observed"]):
        assert len(series["median"]) == len(series["lower"]) == len(series["data"]) == len(fit["dates"])
        low, middle, high = (np.array(series[k], dtype=float) for k in ("lower", "median", "upper"))
        assert np.all(np.isfinite(middle)) and np.all(low <= middle + 1e-12) and np.all(middle <= high + 1e-12)
        np.testing.assert_array_equal(np.array(series["data"], dtype=float), np.asarray(loaded.problem.observed[:, column]))
    assert estimates.quantity_panels(result) == []


def test_when_the_stored_draws_are_prior_draws_the_posterior_panel_shows_the_prior(tmp_path):
    """The hidden-level toy's half-normal priors: the density on the grid is the half-normal's, and the histogram of 4,000 prior
    draws has, in every bin, the prior's probability of that bin, within five binomial standard errors."""
    folder = store_draws(tmp_path / "prior", toy("hidden_level"), toy_data("hidden_level"), n=4000).folder
    for entry in estimates.parameter_panels(_loaded(folder))["parameters"]:
        (element,) = entry["elements"]
        x = np.array(element["prior"]["x"])
        np.testing.assert_allclose(element["prior"]["density"], ss.halfnorm(scale=1).pdf(x), rtol=1e-10)
        edges = np.array(element["bins"])
        expected = np.diff(ss.halfnorm(scale=1).cdf(edges))
        observed = np.array(element["posterior"]) * np.diff(edges)
        assert np.all(np.abs(observed - expected) <= 5 * np.sqrt(expected * (1 - expected) / 4000) + 1e-9)


def test_a_prior_over_a_whole_matrix_is_drawn_and_shown_on_the_same_bins(toy_store):
    panels = estimates.parameter_panels(_loaded(toy_store / "two_variable"))["parameters"]
    sigma = next(p for p in panels if p["key"] == "dynamics.Sigma")
    assert len(sigma["elements"]) == 4
    for element in sigma["elements"]:
        assert element["prior"]["bins"] == element["bins"]                       # inverse Wishart: no density element by element
        assert sum(d * w for d, w in zip(element["prior"]["density"], np.diff(element["bins"]))) <= 1 + 1e-9
    diagonal = sigma["elements"][0]
    assert diagonal["label"] == "dynamics.Sigma[a, a]" and diagonal["window"][0] >= 0   # a variance cannot be negative


def test_fixed_and_improper_priors(tmp_path, rotated):
    fixed = store_draws(tmp_path / "fixed", toy("hidden_level"), toy_data("hidden_level"), edits=[e.SetPrior("y.sigma_eps", fixed=0.5)]).folder
    entry = next(p for p in estimates.parameter_panels(_loaded(fixed))["parameters"] if p["key"] == "y.sigma_eps")
    assert entry["kind"] == "fixed" and entry["fixed"] == 0.5 and entry["elements"] == []
    panels = {p["key"]: p for p in estimates.parameter_panels(_loaded(rotated[0].folder))["parameters"]}
    assert panels["dynamics.c"]["prior_note"].startswith("flat") and panels["dynamics.Sigma"]["prior_note"].startswith("jeffreys")
    assert all(el.get("prior") is None and len(el["posterior"]) == estimates.BINS for el in panels["dynamics.B"]["elements"])
    rotation = panels["dynamics.Q"]["elements"]                                   # the even prior over rotations, under the restriction
    assert all(el["prior"]["bins"] and -1 <= el["window"][0] < el["window"][1] <= 1 for el in rotation)


def test_no_prior_and_ranges(rotated):
    base, ranges = rotated
    entry = next(p for p in estimates.parameter_panels(_loaded(ranges.folder))["parameters"] if p["key"] == "dynamics.Q")
    assert entry["kind"] == "no_prior" and "ranges" in entry["note"] and entry["elements"] == []
    (conventional,), (ranged,) = estimates.quantity_panels(base), estimates.quantity_panels(ranges)
    assert conventional["kind"] == "draws" and conventional["axis"] == [0, 1, 2] and conventional["axis_label"] == "horizon"
    assert ranged["kind"] == "ranges" and all(lo <= hi for lo, hi in zip(ranged["lower"], ranged["upper"]))
    compared = comparison(base, ranges)
    assert compared["edits"] == [{"set_prior": {"target": "dynamics.Q", "prior": "none"}}] and compared["changed"]["blocks"] == ["dynamics"]
    assert [q["quantity"] for q in compared["quantities"]] == list(REPORT) and compared["quantities"][0]["second"]["kind"] == "ranges"
    assert compared["same_data"] and compared["log_marginal_likelihood"] == [None, None]


def test_a_single_point_posterior_fits_the_direct_one_step_prediction(tmp_path):
    """Every draw at the values the data were simulated at: the median and band are the model's prediction there, c + B y(t-1)."""
    data = toy_data("two_variable")
    problem = prepare(toy("two_variable"), data)
    z = np.asarray(problem.layout.unconstrain({k: jnp.asarray(v, dtype=float) for k, v in TRUE["two_variable"].items()}))
    fit = estimates.fit_panels(_loaded(store_draws(tmp_path / "point", toy("two_variable"), data, draws=np.tile(z, (20, 1))).folder))
    c, B = (np.asarray(TRUE["two_variable"][k]) for k in ("dynamics.c", "dynamics.B"))
    y = np.column_stack([data.series["a"], data.series["b"]])
    direct = c + y[:-1] @ B.T
    assert fit["dates"] == list(data.dates[1:]) and fit["route"] == "read_off"
    for column, series in enumerate(fit["observed"]):
        for key in ("median", "lower", "upper"):
            np.testing.assert_allclose(series[key], direct[:, column], rtol=1e-12)


def test_a_run_whose_shocks_are_sampled(tmp_path):
    """Fat-tailed shocks: the route that samples the shocks with the parameters (and the starting state, from the toy's `initial`
    value). Its stored draws include every period's shocks; the panels read the parameters from them, and the fit predicts from the
    state the sampled shocks imply."""
    edit = e.SetShock("level.eta", "student_t(4, 0, sigma_eta)")
    loaded = _loaded(store_draws(tmp_path / "sampled", toy("hidden_level"), toy_data("hidden_level"), edits=[edit], n=100).folder)
    assert loaded.problem.route == "sample" and loaded.draws.shape[1] > len(loaded.problem.system.parameters)
    panels = estimates.parameter_panels(loaded)["parameters"]
    assert [p["key"] for p in panels] == list(loaded.problem.system.parameters) and all(len(el["posterior"]) == estimates.BINS for p in panels for el in p["elements"])
    fit = estimates.fit_panels(loaded)
    assert fit["route"] == "sample" and "sampled shocks" in fit["basis"] and fit["draws"] == fit["draws_tried"] == 100
    assert all(np.all(np.isfinite(np.array(s["median"], dtype=float))) for s in fit["observed"])


def test_series_that_feed_the_model_are_shown_as_data(toy_store):
    loaded = _loaded(toy_store / "error_correction")
    fit = estimates.fit_panels(loaded)
    (fed,) = fit["fed"]
    first = loaded.problem.data.start.first
    assert fed["series"] == "income" and fed["variable"] == "income"
    np.testing.assert_array_equal(fed["data"], loaded.dataset.series["income"][first:])


def test_comparing_a_version_with_an_edited_one(tmp_path):
    data = toy_data("hidden_level")
    edit = e.SetPrior("level.sigma_eta", prior="half_normal(scale=2)")
    first = runs.load_result(store_draws(tmp_path / "base", toy("hidden_level"), data).folder)
    second = runs.load_result(store_draws(tmp_path / "edited", toy("hidden_level"), data, edits=[edit]).folder)
    compared = comparison(first, second)
    assert compared["edits"] == [edit.to_dict()] and compared["note"] is None
    assert compared["changed"]["blocks"] == ["level"] and compared["changed"]["data"] == [] and compared["same_data"]
    assert [p["label"] for p in compared["parameters"]] == ["level.sigma_eta", "y.sigma_eps"]
    assert all(p["first"]["mean"] is not None and p["second"]["mean"] is not None for p in compared["parameters"])
    other = comparison(first, runs.load_result(store_draws(tmp_path / "other", toy("forward_looking"), toy_data("forward_looking")).folder))
    assert other["edits"] is None and "not versions of one model" in other["note"] and not other["same_data"]


def test_quantity_axes():
    stats = {"mean": [1.0, 2.0, 3.0], "sd": [0.1] * 3, "ci68_lower": [0.9, 1.9, 2.9], "ci68_upper": [1.1, 2.1, 3.1], "draws": 10}
    response = estimates.quantity_panel("response(y[gdp], eps[policy], horizons=2..4)", stats)
    assert response["axis"] == [2, 3, 4] and response["axis_label"] == "horizon" and response["latex"].startswith(r"\operatorname{response}")
    other = estimates.quantity_panel("shock_estimate(eps[policy])", stats)
    assert other["axis"] == [0, 1, 2] and other["axis_label"] == "element"


def test_a_run_stored_before_runs_kept_their_data(toy_store, tmp_path):
    import shutil

    folder = tmp_path / "old"
    shutil.copytree(toy_store / "hidden_level", folder)
    (folder / "data.npz").unlink()
    loaded = _loaded(folder)
    panels = estimates.parameter_panels(loaded)
    assert "before runs kept their data" in panels["note"]
    element = panels["parameters"][0]["elements"][0]
    assert "posterior" not in element and element["mean"] is not None and element["prior"]["density"]   # the summary and the prior remain
    assert "before runs kept their data" in estimates.fit_panels(loaded)["note"]
