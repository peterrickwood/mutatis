"""Nonlinear systems, solved to first order around their steady state: closed-form cases."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.compile.system import compile_model
from mutatis.numerics.status import Status
from mutatis.solve.linear import SolveError, solve_linear
from mutatis.spec.yaml_io import load_model

GROWTH = """
model: capital
time: quarterly
blocks:
  capital:
    outputs: [k]
    parameters:
      delta: {prior: "beta(mean=0.1, sd=0.02)"}
      s:     {prior: "beta(mean=0.2, sd=0.05)"}
      alpha: {prior: "beta(mean=0.33, sd=0.05)"}
      sigma: {prior: "half_normal(scale=1)"}
    shocks:
      e: "normal(0, sigma)"
    steady_state:
      k: "(s / delta) ^ (1 / (1 - alpha)) * 0.8"     # deliberately not exact: Newton's method finishes the job
    equations:
      - "k = (1 - delta) * k(-1) + s * k(-1) ^ alpha + e"
data:
  K: {observes: "log(k)", error: "normal(0, 0.01)"}
"""
PARAMS = {"capital.delta": 0.1, "capital.s": 0.2, "capital.alpha": 0.33, "capital.sigma": 0.05}


def test_a_backward_looking_nonlinear_model_rests_at_its_steady_state_and_moves_as_its_derivative_says():
    """k = (1 - delta) k(-1) + s k(-1)^alpha + e rests at k* = (s / delta)^(1 / (1 - alpha)); around it, k moves with
    persistence 1 - delta (1 - alpha) and the shock one for one. The observation log(k) is linearised there too."""
    system = compile_model(load_model(GROWTH))
    assert not system.linear
    solved = solve_linear(system, {k: jnp.asarray(v) for k, v in PARAMS.items()})
    k_star = (0.2 / 0.1) ** (1 / (1 - 0.33))
    persistence = 1 - 0.1 * (1 - 0.33)
    assert int(solved.status) == int(Status.OK)
    np.testing.assert_allclose(np.asarray(solved.P), [[persistence]], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(solved.Q), [[1.0]], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(solved.kappa), [k_star * (1 - persistence)], rtol=1e-10)   # rests at k*
    np.testing.assert_allclose(np.asarray(solved.Z), [[1 / k_star]], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(solved.d), [np.log(k_star) - 1.0], rtol=1e-10)              # log(k*) - Z k*


def test_derivatives_pass_through_the_steady_state():
    """d k* / d s = k* / (s (1 - alpha)), from the implicit function theorem through Newton's method."""
    system = compile_model(load_model(GROWTH))
    rest = lambda s: solve_linear(system, {**{k: jnp.asarray(v) for k, v in PARAMS.items()}, "capital.s": s}).kappa[0] / (0.1 * (1 - 0.33))
    k_star = (0.2 / 0.1) ** (1 / (1 - 0.33))
    np.testing.assert_allclose(float(jax.grad(rest)(jnp.asarray(0.2))), k_star / (0.2 * (1 - 0.33)), rtol=1e-8)


def test_a_nonlinear_model_needs_steady_state_values():
    system = compile_model(load_model(GROWTH.replace('    steady_state:\n      k: "(s / delta) ^ (1 / (1 - alpha)) * 0.8"     # deliberately not exact: Newton\'s method finishes the job\n', "")))
    with pytest.raises(SolveError, match="steady_state"):
        solve_linear(system, {k: jnp.asarray(v) for k, v in PARAMS.items()})


def test_a_forward_looking_nonlinear_model_matches_dynare():
    """A basic real business cycle model (tests/models/rbc.yaml; the same model in Dynare's language, rbc.mod): steady state and
    first-order solution against Dynare 6.5's (rbc_dynare.csv; oo_.dr.ys, ghx, ghu at the values in rbc.mod)."""
    import csv
    from pathlib import Path

    models = Path(__file__).resolve().parents[1] / "models"
    system = compile_model(load_model((models / "rbc.yaml").read_text()))
    values = {"delta": 0.025, "alpha": 0.33, "beta": 0.99, "technology.rho": 0.95, "technology.sigma": 0.01}
    solved = solve_linear(system, {k: jnp.asarray(v) for k, v in values.items()})
    assert int(solved.status) == int(Status.OK)
    labels = system.variable_labels
    P, Q = np.asarray(solved.P), np.asarray(solved.Q)
    rest = np.linalg.solve(np.eye(len(labels)) - P, np.asarray(solved.kappa))
    checked = 0
    with open(models / "rbc_dynare.csv") as handle:
        for row in csv.DictReader(handle):
            variable, source, theirs = labels.index(row["variable"]), row["input"], float(row["value"])
            if source == "steady_state":
                ours = rest[variable]
            elif source.endswith("(-1)"):
                ours = P[variable, labels.index(source[:-4])]
            else:
                ours = Q[variable, 0]
            assert abs(ours - theirs) < 1e-9 * max(1.0, abs(theirs)), (row, ours)
            checked += 1
    assert checked == 20
