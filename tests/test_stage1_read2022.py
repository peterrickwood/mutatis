"""Stage 1 acceptance (slow tier): Read (2022), Restrictions (1) to (5), against his published figure data. Needs network access
the first time (the data are downloaded per demos/read2022/data.yaml).

Run: .venv/bin/python -m pytest -m slow -s tests/test_stage1_read2022.py

- Conventional results (a uniform prior on the rotation): every statistic must agree with Read's within sampling error, estimated
  by a bootstrap over our draws and scaled to his 1,000.
- Ranges (no prior on the rotation): our bounds must be exact, checked against an independent optimiser on sampled draws. The
  comparison with Read's published range statistics is printed but not required to pass: his bounds are not always the exact ones
  (decisions log, 28 and 32), so they cannot serve as ground truth here.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from scipy.optimize import minimize

from mutatis.infer.identification import fit_one_column, one_column_extremes
from mutatis.infer.regression import regression_form, regression_posterior
from mutatis.infer.restricted import RestrictedOthers
from mutatis.measure.manifest import load_manifest
from mutatis.spec.edits import apply_edits, load_edit_file
from mutatis.spec.yaml_io import load_model
from mutatis.versions import run as runs
from mutatis.versions.pipeline import prepare

pytestmark = pytest.mark.slow
DEMO = Path(__file__).resolve().parents[1] / "demos" / "read2022"


@pytest.fixture(scope="module")
def inputs():
    return load_model((DEMO / "model.yaml").read_text()), load_manifest(DEMO / "data.yaml"), tuple(yaml.safe_load((DEMO / "report.yaml").read_text()))


def _edits(restriction, ranges):
    edits = load_edit_file(DEMO / "edits" / f"restriction{restriction}.yaml")
    return edits + (load_edit_file(DEMO / "edits" / "ranges.yaml") if ranges else [])


def _print(rows, title):
    bad = [r for r in rows if not r["ok"]]
    print(f"\n{title}: {len(rows)} statistics, {len(rows) - len(bad)} within tolerance, {len(bad)} outside")
    for r in bad:
        print(f"   {r['quantity']}[{r['element']}] {r['statistic']}: ours {r['ours']:.5f}, Read {r['theirs']:.5f}, tolerance {r['tolerance']:.5f}")
    return bad


@pytest.mark.parametrize("restriction", [1, 2, 3, 4, 5])
def test_conventional(restriction, inputs, tmp_path_factory):
    model, dataset, report = inputs
    result = runs.run(model, dataset, store=tmp_path_factory.getbasetemp() / "read2022", edits=_edits(restriction, False), report=report, n_draws=1000)
    rows = runs.compare_with_reference(result, DEMO / "reference" / f"restriction{restriction}.csv")
    assert rows and not _print(rows, f"Restriction ({restriction}), conventional")


@pytest.mark.parametrize("restriction", [1, 2, 3, 4, 5])
def test_ranges_are_exact(restriction, inputs, tmp_path_factory):
    model, dataset, report = inputs
    result = runs.run(model, dataset, store=tmp_path_factory.getbasetemp() / "read2022", edits=_edits(restriction, True), report=report, n_draws=1000)
    _print(runs.compare_with_reference(result, DEMO / "reference" / f"restriction{restriction}.csv"), f"Restriction ({restriction}), ranges, against Read (not required)")
    # The requirement: on a sample of draws, no independent optimiser finds an allowed value outside our bounds, and ours are attained.
    problem = prepare(apply_edits(model, _edits(restriction, True)), dataset)
    posterior = regression_posterior(problem, jax.random.key(restriction), n_draws=5, report=report)
    others = RestrictedOthers(problem, regression_form(problem).others, report)
    rotation = others.no_prior[0].name
    worst = 0.0
    for i in range(len(posterior.particles)):
        values, _ = problem.layout.constrain(jnp.asarray(posterior.particles[i]))
        form = fit_one_column(lambda Q: others.restriction_values({**values, rotation: Q}), lambda Q: others.report_values({**values, rotation: Q}), 4, 0, jax.random.key(i))
        lower, upper, _, _, _ = one_column_extremes(form)
        rng = np.random.default_rng(i)
        for row in rng.choice(len(form.C), 12, replace=False):
            c = form.C[row]
            for sign in (1.0, -1.0):
                for start in rng.normal(size=(20, 4)):
                    res = minimize(lambda q: sign * (c @ q), start / np.linalg.norm(start), method="SLSQP",
                                   constraints=[{"type": "ineq", "fun": lambda q: form.S @ q}, {"type": "eq", "fun": lambda q: q @ q - 1}],
                                   options={"ftol": 1e-12, "maxiter": 300})
                    if res.success and np.all(form.S @ res.x >= -1e-9) and abs(res.x @ res.x - 1) < 1e-8:
                        value = c @ res.x
                        worst = max(worst, lower[row] - value, value - upper[row])
    print(f"   largest value an independent optimiser found outside our bounds: {worst:.2e}")
    assert worst < 1e-8
