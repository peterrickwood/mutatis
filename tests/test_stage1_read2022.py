"""Stage 1 acceptance (slow tier): Read (2022), Restrictions (1) to (6), against his published figure data. Needs network access
the first time (the data are downloaded per demos/read2022/data.yaml).

Run: .venv/bin/python -m pytest -m slow -s tests/test_stage1_read2022.py

- Conventional results (a uniform prior on the rotation): every statistic must agree with Read's within sampling error, estimated
  by a bootstrap over our draws and scaled to his 1,000.
- Ranges (no prior on the rotation): our bounds must be exact, checked against an independent optimiser on sampled draws. The
  comparison with Read's published range statistics is printed but not required to pass: his bounds are not always the exact ones
  (decisions log, 28 and 32), so they cannot serve as ground truth here. For Restriction (6) his bounds come from random
  rotations, so ours, being exact, must contain his.
- Restriction (6): Read's published figure 6 comes from replication code that pairs the proxy with the VAR's values from 36
  quarters earlier (decisions log, 46). It is compared with our results for the same pairing, given as data (the proxy moved 36
  quarters earlier); the model itself, in demos/read2022, pairs them as the paper describes.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from scipy.optimize import minimize

from mutatis.infer.identification import fit_one_column, linear_in_rotation, one_column_extremes
from mutatis.infer.regression import regression_form, regression_posterior
from mutatis.infer.restricted import RestrictedOthers
from mutatis.measure.data import Dataset
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


def _as_read_code_pairs_the_proxy(dataset):
    """Read's mainfile_proxy.m shortens the proxy to its 100 values and then indexes the VAR's rows with a mask of that length,
    so his first-stage regression pairs the proxy for 1994Q1-2018Q4 with the VAR's values for the first 100 periods explained,
    1985Q1-2009Q4. The same pairing as data: the proxy moved earlier by the number of explained periods before it starts."""
    proxy = np.asarray(dataset.series["PROXY"], dtype=float)
    shift = int(np.flatnonzero(~np.isnan(proxy))[0]) - 4          # explained periods start after the 4 lags
    moved = np.full_like(proxy, np.nan)
    moved[:-shift] = proxy[shift:]
    return Dataset(dataset.dates, {**dataset.series, "PROXY": moved})


def _data(restriction, dataset):
    return _as_read_code_pairs_the_proxy(dataset) if restriction == 6 else dataset


def _print(rows, title):
    bad = [r for r in rows if not r["ok"]]
    print(f"\n{title}: {len(rows)} statistics, {len(rows) - len(bad)} within tolerance, {len(bad)} outside")
    for r in bad:
        print(f"   {r['quantity']}[{r['element']}] {r['statistic']}: ours {r['ours']:.5f}, Read {r['theirs']:.5f}, tolerance {r['tolerance']:.5f}")
    return bad


@pytest.mark.parametrize("restriction", [1, 2, 3, 4, 5, 6])
def test_conventional(restriction, inputs, tmp_path_factory):
    model, dataset, report = inputs
    # Read tries up to 100,000 rotations per draw before calling its allowed set empty (opt.maxDraw); it matters only for (6).
    result = runs.run(model, _data(restriction, dataset), store=tmp_path_factory.getbasetemp() / "read2022", edits=_edits(restriction, False), report=report,
                      n_draws=1000, max_tries=100_000)
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


def test_ranges_restriction6_are_exact_and_contain_reads(inputs, tmp_path_factory):
    """Restriction (6)'s proxy restrictions involve every column of the rotation, so its ranges come from the search over rotations.
    Required: on a sample of draws, an independent optimiser on an exact reduction finds no allowed value outside our bounds; and
    our sets of posterior means contain Read's, found from random rotations, within sampling error."""
    model, dataset, report = inputs
    data = _as_read_code_pairs_the_proxy(dataset)
    result = runs.run(model, data, store=tmp_path_factory.getbasetemp() / "read2022", edits=_edits(6, True), report=report, n_draws=1000)
    rows = runs.compare_with_reference(result, DEMO / "reference" / "restriction6.csv")
    _print(rows, "Restriction (6), ranges, against Read (only containment required)")
    sets = [r for r in rows if r["statistic"] in ("set_mean_lower", "set_mean_upper")]
    outside = [r for r in sets if (r["ours"] > r["theirs"] + r["tolerance"]) if r["statistic"] == "set_mean_lower"]
    outside += [r for r in sets if (r["ours"] < r["theirs"] - r["tolerance"]) if r["statistic"] == "set_mean_upper"]
    for r in outside:
        print(f"   NOT CONTAINED {r['quantity']}[{r['element']}] {r['statistic']}: ours {r['ours']:.5f}, Read {r['theirs']:.5f}")
    assert sets and not outside

    # Exactness. With Restriction (5)'s restrictions on the first column (S1 q1 >= 0) and the proxy's, a'q1 >= |a'qj| for every
    # other column with a = (d chol(Sigma))', an allowed rotation exists exactly when S1 q1 >= 0 and a'q1 >= |a| / sqrt(4) for
    # its first column (test_identification.py), so the bounds of a quantity c'q1 are those of a problem in q1 alone.
    problem = prepare(apply_edits(model, _edits(6, True)), data)
    posterior = regression_posterior(problem, jax.random.key(6), n_draws=5, report=report)
    others = RestrictedOthers(problem, regression_form(problem).others, report)
    rotation = others.no_prior[0].name
    n = 4
    first = np.arange(n) * n                     # where the first column sits in Q.reshape(-1)
    worst, checked = 0.0, 0
    for i in range(len(posterior.particles)):
        values, _ = problem.layout.constrain(jnp.asarray(posterior.particles[i]))
        S, s0 = linear_in_rotation(lambda Q: others.restriction_values({**values, rotation: Q}), n, jax.random.key(i))
        C, c0 = linear_in_rotation(lambda Q: others.report_values({**values, rotation: Q}), n, jax.random.key(100 + i))
        assert np.abs(s0).max() < 1e-9 and np.abs(c0).max() < 1e-9 and np.abs(np.delete(C, first, axis=1)).max() < 1e-9
        one_column = np.abs(np.delete(S, first, axis=1)).max(axis=1) < 1e-9
        S1 = S[one_column][:, first]
        a = np.asarray(jnp.linalg.cholesky(values["economy.Sigma"]).T @ values["proxy.d"].reshape(-1))
        assert (~one_column).sum() == 2 * (n - 1)            # the proxy rows: a'(q1 - qj) >= 0 and a'(q1 + qj) >= 0
        lower = np.concatenate([posterior.diagnostics["bounds"][q][0][i] for q in report])
        upper = np.concatenate([posterior.diagnostics["bounds"][q][1][i] for q in report])
        rng = np.random.default_rng(i)
        constraints = [{"type": "ineq", "fun": lambda q: S1 @ q}, {"type": "ineq", "fun": lambda q: a @ q - np.linalg.norm(a) / np.sqrt(n)},
                       {"type": "eq", "fun": lambda q: q @ q - 1}]
        for row in rng.choice(len(C), 12, replace=False):
            c = C[row][first]
            for sign in (1.0, -1.0):
                for start in rng.normal(size=(20, n)):
                    res = minimize(lambda q: sign * (c @ q), start / np.linalg.norm(start), method="SLSQP", constraints=constraints,
                                   options={"ftol": 1e-12, "maxiter": 300})
                    if res.success and all(np.all(np.atleast_1d(k["fun"](res.x)) >= -1e-9) for k in constraints[:2]) and abs(res.x @ res.x - 1) < 1e-8:
                        value = c @ res.x
                        worst = max(worst, lower[row] - value, value - upper[row])
                        checked += 1
    print(f"   {checked} optimiser solutions checked; largest value outside our bounds: {worst:.2e}")
    assert checked > 100 and worst < 1e-8
