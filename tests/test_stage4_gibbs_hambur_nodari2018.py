"""Stage 4, first part (slow tier): the housing model of Gibbs, Hambur and Nodari (2018), imported from the authors' own Dynare
code (demos/gibbs_hambur_nodari2018/dynare.yaml), against Dynare. Needs network access the first time (the 145 MB archive is
downloaded per the demonstration's files) and Dynare's preprocessor (skipped if it is not installed).

Run: .venv/bin/python -m pytest -m slow -s tests/test_stage4_gibbs_hambur_nodari2018.py

- Every fixed parameter value worked out from the file equals the one the authors' run stored.
- At the authors' stored posterior mode, the log-likelihood and the log prior density equal Dynare's (reference/likelihood.csv).
"""

import csv
import os
import shutil
import zipfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from mutatis.measure.manifest import fetch, load_manifest
from mutatis.versions.pipeline import prepare

pytestmark = pytest.mark.slow
DEMO = Path(__file__).resolve().parents[1] / "demos" / "gibbs_hambur_nodari2018"


@pytest.fixture(scope="module")
def imported():
    if not (os.environ.get("MUTATIS_DYNARE_PREPROCESSOR") or shutil.which("dynare-preprocessor")):
        pytest.skip("Dynare's preprocessor is not installed")
    from mutatis.io.dynare import import_dynare_source, stored_parameter_values

    description = yaml.safe_load((DEMO / "dynare.yaml").read_text())
    archive = fetch(description["source"]["url"], description["source"]["sha256"])
    with zipfile.ZipFile(archive) as files:
        stored = stored_parameter_values(files.read(description["values"]))
        mode_file = files.read("Domestic Sector/multisector_mode.mat")
    return import_dynare_source(DEMO / "dynare.yaml"), stored, mode_file


def test_fixed_values_equal_the_stored_ones(imported):
    """Worked out from the file in its order (with the 31 values its MATLAB steady state computes taken from the stored results),
    every fixed parameter equals the value in the authors' stored M_.params."""
    model, stored, _ = imported
    fixed = {k: p.fixed for k, p in model.parameters.items() if p.prior is None}
    assert len(model.parameters) == 189 + 21 and len(fixed) == 210 - 58    # the 21 estimated shock sizes are parameters here
    for name, value in fixed.items():
        assert abs(value - stored[name]) <= 1e-12 * max(1.0, abs(stored[name])), (name, value, stored[name])


def test_likelihood_and_prior_at_the_mode_equal_dynares(imported):
    from io import BytesIO

    from scipy.io import loadmat

    model, _, mode_file = imported
    problem = prepare(model, load_manifest(DEMO / "data.yaml"))
    assert problem.system.n_variables == 150 and len(problem.data.dates) == 99
    mode = loadmat(BytesIO(mode_file), squeeze_me=True)
    values = {}
    for name, x in zip((str(n).strip() for n in mode["parameter_names"]), mode["xparam1"]):
        values[f"stderr_{name}" if f"stderr_{name}" in model.parameters else name] = jnp.asarray(float(x))
    assert len(values) == 58 and set(values) == {u.name for u in problem.parameter_unknowns}
    with open(DEMO / "reference" / "likelihood.csv") as handle:
        reference = {row["quantity"]: float(row["value"]) for row in csv.DictReader(handle)}
    params = {**problem.layout.fixed, **values}
    prior = sum(float(jnp.sum(u.prior.build(params).log_prob(values[u.name]))) for u in problem.parameter_unknowns)
    likelihood = float(problem.log_likelihood(problem.layout.unconstrain(values)))
    print(f"\n   log likelihood {likelihood:.10f} against Dynare's {reference['log_likelihood']:.10f}; log prior {prior:.10f} against {reference['log_prior_density']:.10f}")
    np.testing.assert_allclose(likelihood, reference["log_likelihood"], rtol=1e-12)
    np.testing.assert_allclose(prior, reference["log_prior_density"], rtol=1e-12)
