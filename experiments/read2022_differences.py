"""Diagnosis of the differences from Read (2022) found by tests/test_stage1_read2022.py.

1. Conventional intervals: how much do our own 68% interval ends move between seeds (Restriction (3))?
2. Ranges: are our exact bounds right where they differ from Read's (Restriction (2))? An independent optimiser (SciPy's SLSQP
   from 100 starting points) minimises and maximises the same responses over allowed rotation columns.

Run: PYTHONPATH=. MUTATIS_CACHE=... .venv/bin/python -u experiments/read2022_differences.py
"""

import jax
import jax.numpy as jnp
import numpy as np
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

DEMO = "demos/read2022"


def main():
    model = load_model(open(f"{DEMO}/model.yaml").read())
    dataset = load_manifest(f"{DEMO}/data.yaml")
    report = tuple(yaml.safe_load(open(f"{DEMO}/report.yaml")))
    cash, cpi = report[0], report[2]

    print("1. Restriction (3), conventional: our 68% interval ends across 5 seeds (1,000 draws each)")
    ends = []
    for seed in range(5):
        result = runs.run(model, dataset, store="/tmp/claude-1000/read_diag", edits=load_edit_file(f"{DEMO}/edits/restriction3.yaml"), report=report, n_draws=1000, seed=seed)
        q = result.summary["quantities"]
        ends.append([q[cash]["ci68_lower"][1], q[cash]["ci68_upper"][1], q[cpi]["ci68_lower"][13]])
        formula = [1.51 * q[cash]["sd"][1] / np.sqrt(1000), 1.51 * q[cash]["sd"][1] / np.sqrt(1000), 1.51 * q[cpi]["sd"][13] / np.sqrt(1000)]
    ends = np.array(ends)
    print("   values by seed (cash h1 lower, cash h1 upper, cpi h13 lower):\n", np.round(ends, 4))
    print("   spread across seeds (sd):", np.round(ends.std(axis=0, ddof=1), 4), "  the formula's standard error:", np.round(formula, 4))

    print("\n2. Restriction (2), ranges: our exact bounds against SLSQP on 15 draws")
    edited = apply_edits(model, load_edit_file(f"{DEMO}/edits/restriction2.yaml") + load_edit_file(f"{DEMO}/edits/ranges.yaml"))
    problem = prepare(edited, dataset)
    twi = report[3]
    posterior = regression_posterior(problem, jax.random.key(0), n_draws=15, report=(twi,))
    others = RestrictedOthers(problem, regression_form(problem).others, (twi,))
    rotation = others.no_prior[0].name
    worst = 0.0
    for i in range(len(posterior.particles)):
        values, _ = problem.layout.constrain(jnp.asarray(posterior.particles[i]))
        restrict = lambda Q: others.restriction_values({**values, rotation: Q})
        quantity = lambda Q: others.report_values({**values, rotation: Q})
        form = fit_one_column(restrict, quantity, 4, 0, jax.random.key(i))
        lower, upper, _, _, _ = one_column_extremes(form)
        rng = np.random.default_rng(i)
        found_lower, found_upper = np.full(len(lower), np.inf), np.full(len(lower), -np.inf)
        for start in rng.normal(size=(100, 4)):
            for h in (1, 2, 3, 4, 10, 11):
                c = form.C[h]
                for sign in (1.0, -1.0):
                    res = minimize(lambda q: sign * (c @ q), start / np.linalg.norm(start), method="SLSQP",
                                   constraints=[{"type": "ineq", "fun": lambda q: form.S @ q}, {"type": "eq", "fun": lambda q: q @ q - 1}],
                                   options={"ftol": 1e-12, "maxiter": 300})
                    if res.success and np.all(form.S @ res.x >= -1e-9) and abs(res.x @ res.x - 1) < 1e-8:
                        v = c @ res.x
                        found_lower[h], found_upper[h] = min(found_lower[h], v), max(found_upper[h], v)
        for h in (1, 2, 3, 4, 10, 11):
            gap = max(lower[h] - found_lower[h], found_upper[h] - upper[h])  # > 0 would mean SLSQP found a value outside our bounds
            worst = max(worst, gap)
        if i < 4:
            print(f"   draw {i}: TWI h2 exact [{lower[2]:.4f}, {upper[2]:.4f}] SLSQP [{found_lower[2]:.4f}, {found_upper[2]:.4f}];"
                  f" h4 exact [{lower[4]:.4f}, {upper[4]:.4f}] SLSQP [{found_lower[4]:.4f}, {found_upper[4]:.4f}]")
    print(f"   largest amount by which SLSQP found a value outside our bounds, over 15 draws and 6 horizons: {worst:.2e}")


if __name__ == "__main__":
    main()
