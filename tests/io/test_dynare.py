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


def test_a_file_with_includes_local_definitions_estimated_shock_sizes_and_values_computed_elsewhere(tmp_path):
    """The features a full-size published model uses (stage 4): an included file; a model-local variable (#); a two-period lag;
    a declared shock no equation uses; a parameter whose value MATLAB code computes (given here in `values`), and one worked out
    from it; an estimated shock standard deviation; initval entries for shocks; and statements after `estimation`, which do
    not count. The import must solve as the same model written by hand."""
    (tmp_path / "declarations.mod").write_text("var y x; varexo e u unused; parameters rho phi theta scale;\n")
    (tmp_path / "main.mod").write_text("""
@#include "declarations.mod"
rho = 0.6;
[values, ~] = steady_state_helper();
phi = values(1);
theta = phi / 2;
scale = 100;
model(linear);
# slope = theta / scale;
y = rho * y(-1) + 0.1 * y(-2) + slope * x + e / scale;
x = 0.5 * x(+1) + u;
end;
initval; y = 0; x = 0; e = 0; u = 0; end;
shocks; var e; stderr 2; var u; stderr 1; end;
varobs y;
estimated_params; rho, beta_pdf, 0.5, 0.2; stderr e, gamma_pdf, 1, 0.5; end;
estimation(datafile=nothing, mode_compute=0);
shocks; var u; stderr 7; end;
""")
    model = import_dynare(tmp_path / "main.mod", values={"phi": 0.8})
    block = model.blocks["main"]
    assert set(block.shocks) == {"e", "u"} and block.shocks["e"].distribution == "normal(0, stderr_e)" and block.shocks["u"].distribution == "normal(0, 1)"
    assert set(block.definitions) == {"slope"} and block.definitions["slope"].strip("()") == "theta / scale" and set(block.steady_state) == {"y", "x"}
    assert model.parameters["phi"].fixed == 0.8 and model.parameters["theta"].fixed == 0.4
    assert model.parameters["rho"].prior.startswith("beta(") and model.parameters["stderr_e"].prior.startswith("gamma(")
    with pytest.raises(Exception, match="computed outside Dynare's language"):
        import_dynare(tmp_path / "main.mod")
    hand = load_model("""
model: hand
time: quarterly
parameters: {rho: {fixed: 0.6}, stderr_e: {fixed: 2.0}}
blocks:
  main:
    outputs: [y, x]
    shocks: {e: "normal(0, stderr_e)", u: "normal(0, 1)"}
    equations:
      - y = rho * y(-1) + 0.1 * y(-2) + 0.4 / 100 * x + e / 100
      - x = 0.5 * E(x(+1)) + u
data: {y: {observes: y}}
""")
    P, Q, _, labels = _solution(compile_model(model), {"rho": 0.6, "stderr_e": 2.0})
    P2, Q2, _, labels2 = _solution(compile_model(hand), {})
    assert sorted(labels) == sorted(labels2)
    order = [labels2.index(v) for v in labels]
    np.testing.assert_allclose(P2[np.ix_(order, order)], P, atol=1e-14)
    np.testing.assert_allclose(Q2[order], Q, atol=1e-14)


def test_stored_parameter_values_are_read_from_a_dynare_results_file(tmp_path):
    from scipy.io import savemat

    from mutatis.io.dynare import stored_parameter_values

    savemat(tmp_path / "model_results.mat", {"M_": {"param_names": np.array(["alpha", "beta_long"], dtype=object), "params": np.array([0.33, 0.99])}})
    assert stored_parameter_values(str(tmp_path / "model_results.mat")) == {"alpha": 0.33, "beta_long": 0.99}
    assert stored_parameter_values((tmp_path / "model_results.mat").read_bytes()) == {"alpha": 0.33, "beta_long": 0.99}
