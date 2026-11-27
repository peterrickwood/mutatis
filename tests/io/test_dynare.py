"""Importing Dynare model files through Dynare's own preprocessor (skipped when the preprocessor is not installed; see
mutatis/io/dynare.py for how to point to it)."""

import csv
import os
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.compile.system import compile_model
from mutatis.numerics.status import Status
from mutatis.solve.linear import solve_linear
from mutatis.spec.yaml_io import load_model

if not (os.environ.get("MUTATIS_DYNARE_PREPROCESSOR") or shutil.which("dynare-preprocessor")):
    pytest.skip("Dynare's preprocessor is not installed", allow_module_level=True)

from mutatis.io.dynare import _inverse_gamma_sd, import_dynare  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"


def _solution(system, values):
    solved = solve_linear(system, {**system.fixed_parameters(), **{k: jnp.asarray(v) for k, v in values.items()}})
    assert int(solved.status) == int(Status.OK)
    P, Q, labels = np.asarray(solved.P), np.asarray(solved.Q), system.variable_labels
    rest = np.linalg.solve(np.eye(len(labels)) - P, np.asarray(solved.kappa))
    return P, Q, rest, labels


def test_the_imported_rbc_model_solves_as_dynare_does_and_as_the_hand_written_one():
    model = import_dynare(MODELS / "rbc.mod")
    assert set(model.blocks["rbc"].outputs) == {"c", "k", "y", "i", "a"} and model.parameters["beta"].fixed == 0.99
    P, Q, rest, labels = _solution(compile_model(model), {})
    with open(MODELS / "rbc_dynare.csv") as handle:
        for row in csv.DictReader(handle):
            variable, source, theirs = labels.index(row["variable"]), row["input"], float(row["value"])
            ours = rest[variable] if source == "steady_state" else P[variable, labels.index(source[:-4])] if source.endswith("(-1)") else Q[variable, 0]
            assert abs(ours - theirs) < 1e-9 * max(1.0, abs(theirs)), (row, ours)
    hand = compile_model(load_model((MODELS / "rbc.yaml").read_text()))
    P2, Q2, rest2, labels2 = _solution(hand, {"delta": 0.025, "alpha": 0.33, "beta": 0.99, "technology.rho": 0.95, "technology.sigma": 0.01})
    order = [labels2.index(v) for v in labels]
    np.testing.assert_allclose(P2[np.ix_(order, order)], P, atol=1e-12)
    np.testing.assert_allclose(Q2[order], Q, atol=1e-12)
    np.testing.assert_allclose(rest2[order], rest, rtol=1e-12)


def test_dynare_priors_become_the_same_distributions():
    """Dynare states the inverse gamma for a standard deviation by its mean and standard deviation; the import finds (s, nu)."""
    from mutatis.dist import InverseGammaSD
    import jax

    s, nu = _inverse_gamma_sd(0.5013256549262001, 0.26208)     # about s = 0.4, nu = 4 (Herbst and Schorfheide's sigma_R prior)
    draws = np.asarray(InverseGammaSD(s, nu).sample(jax.random.key(0), (400_000,)))
    assert abs(draws.mean() - 0.5013256549262001) < 5 * draws.std() / np.sqrt(len(draws))
    assert abs(s - 0.4) < 1e-3 and abs(nu - 4) < 1e-2
