"""Stage 2 acceptance (slow tier): the small New Keynesian model of Herbst and Schorfheide (2016), written as a specification
(demos/herbst_schorfheide2016), against the authors' own code and against Dynare. Needs network access the first time (the
data are downloaded per the demonstration's manifest).

Run: .venv/bin/python -m pytest -m slow -s tests/test_stage2_herbst_schorfheide.py

- The log-likelihood and the log prior density at two parameter points equal those of the authors' code (reference/likelihood.csv).
- The first-order solution at the authors' posterior mode equals Dynare's for the same model (reference/dynare_solution.csv).
"""

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.measure.manifest import load_manifest
from mutatis.spec.yaml_io import load_model
from mutatis.versions.pipeline import prepare

pytestmark = pytest.mark.slow
DEMO = Path(__file__).resolve().parents[1] / "demos" / "herbst_schorfheide2016"


@pytest.fixture(scope="module")
def problem():
    return prepare(load_model((DEMO / "model.yaml").read_text()), load_manifest(DEMO / "data.yaml"))


def _points():
    points = {}
    with open(DEMO / "reference" / "likelihood.csv") as handle:
        for row in csv.DictReader(handle):
            points.setdefault(int(row["point"]), {})[row["quantity"]] = float(row["value"])
    return points


@pytest.mark.parametrize("point", [0, 1])
def test_likelihood_and_prior_equal_the_authors_code(problem, point):
    reference = _points()[point]
    values = {k: jnp.asarray(v) for k, v in reference.items() if k not in ("log_likelihood", "log_prior_density")}
    z = problem.layout.unconstrain(values)
    params = {**problem.layout.fixed, **values}
    prior = sum(float(jnp.sum(u.prior.build(params).log_prob(values[u.name]))) for u in problem.parameter_unknowns)
    np.testing.assert_allclose(float(problem.log_likelihood(z)), reference["log_likelihood"], rtol=1e-12)
    np.testing.assert_allclose(prior, reference["log_prior_density"], rtol=1e-12, atol=1e-12)


def test_first_order_solution_equals_dynares(problem):
    """Each variable's coefficient on each state's previous value and on each shock. Observed series are variables in the Dynare
    file and observation equations here, so theirs are compared with Z P and Z Q."""
    values = {k: jnp.asarray(v) for k, v in _points()[0].items() if k not in ("log_likelihood", "log_prior_density")}
    solved = problem.solve({**problem.layout.fixed, **values})
    labels = problem.system.variable_labels
    shocks = [label.split(".", 1)[1] for label in problem.system.shock_labels]
    observed = {o.series: i for i, o in enumerate(problem.system.observations)}
    P, Q, Z = np.asarray(solved.P), np.asarray(solved.Q), np.asarray(solved.Z)
    checked = 0
    with open(DEMO / "reference" / "dynare_solution.csv") as handle:
        for row in csv.DictReader(handle):
            variable, source, theirs = row["variable"], row["input"], float(row["value"])
            matrix = P if source.endswith("(-1)") else Q
            column = labels.index(source[:-4]) if source.endswith("(-1)") else shocks.index(source)
            ours = (Z @ matrix)[observed[variable], column] if variable in observed else matrix[labels.index(variable), column]
            assert abs(ours - theirs) < 1e-10 * max(1.0, abs(theirs)), (variable, source, ours, theirs)
            checked += 1
    assert checked == 56


@pytest.mark.parametrize("point", [0, 1])
def test_the_dynare_file_imported_gives_the_same_likelihood_and_priors(problem, point):
    """model.mod, imported through Dynare's preprocessor (skipped if it is not installed): the same log-likelihood as the
    hand-written specification and the authors' code, and the same priors (the inverse gammas through Dynare's mean and
    standard deviation, so to about 1e-9)."""
    import os
    import shutil
    if not (os.environ.get("MUTATIS_DYNARE_PREPROCESSOR") or shutil.which("dynare-preprocessor")):
        pytest.skip("Dynare's preprocessor is not installed")
    from mutatis.io.dynare import import_dynare

    imported = prepare(import_dynare(DEMO / "model.mod"), load_manifest(DEMO / "data.yaml"))
    reference = _points()[point]
    values = {k.split(".")[-1]: jnp.asarray(v) for k, v in reference.items() if k not in ("log_likelihood", "log_prior_density")}
    z = imported.layout.unconstrain(values)
    np.testing.assert_allclose(float(imported.log_likelihood(z)), reference["log_likelihood"], rtol=1e-12)
    params = {**imported.layout.fixed, **values}
    prior = sum(float(jnp.sum(u.prior.build(params).log_prob(values[u.name]))) for u in imported.parameter_unknowns)
    np.testing.assert_allclose(prior, reference["log_prior_density"], rtol=1e-8, atol=1e-8)


def test_rule_of_thumb_expectations_are_one_edit():
    """With naive expectations (next period expected to be like the last), the model has no expected future values left, so it
    is solved without the forward-looking solve; it still runs through the same path and gives a finite, different likelihood."""
    from mutatis.spec import edits as e

    rational = load_model((DEMO / "model.yaml").read_text())
    naive = e.SetExpectations("naive").apply(rational)
    data = load_manifest(DEMO / "data.yaml")
    plain, edited = prepare(rational, data), prepare(naive, data)
    assert plain.system.has_expectations and not edited.system.has_expectations
    values = {k: jnp.asarray(v) for k, v in _points()[1].items() if k not in ("log_likelihood", "log_prior_density")}
    a, b = (float(p.log_likelihood(p.layout.unconstrain(values))) for p in (plain, edited))
    assert np.isfinite(b) and abs(a - b) > 1.0


def test_posterior_agrees_with_the_authors_sampler(problem):
    """The default sampler (`mcmc`: peak search, a ladder, run until its convergence tests pass) against the authors' random-walk
    Metropolis-Hastings run (reference/posterior.csv): every parameter's mean and standard deviation within 5 standard errors of
    the difference, each side's standard error from its own effective sample size."""
    from mutatis.infer.estimate import estimate

    posterior = estimate(problem, jax.random.key(0), method="mcmc", max_seconds=3600.0, processes=4)
    d = posterior.diagnostics
    assert d["converged"], d["status"]
    params = jax.vmap(problem.parameters)(jnp.asarray(posterior.particles))
    ours_ess = float(d["min_ess"])
    print(f"\n   our run: {d['seconds'] if 'seconds' in d else ''} s, smallest effective sample size {ours_ess:.0f}, rungs {d['rungs']}")
    with open(DEMO / "reference" / "posterior.csv") as handle:
        for row in csv.DictReader(handle):
            x = np.asarray(params[row["parameter"]], dtype=float)
            mean, sd, ess = float(row["mean"]), float(row["sd"]), float(row["effective_draws"])
            se_mean = np.hypot(x.std() / np.sqrt(ours_ess), sd / np.sqrt(ess))
            se_sd = np.hypot(x.std() / np.sqrt(2 * ours_ess), sd / np.sqrt(2 * ess))
            print(f"   {row['parameter']:20s} mean {x.mean():8.4f} against {mean:8.4f} ({(x.mean() - mean) / se_mean:+5.2f} se); "
                  f"sd {x.std():7.4f} against {sd:7.4f} ({(x.std() - sd) / se_sd:+5.2f} se)")
            assert abs(x.mean() - mean) < 5 * se_mean and abs(x.std() - sd) < 5 * se_sd, row["parameter"]
