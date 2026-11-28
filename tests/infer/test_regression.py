"""The exact regression formula: when it applies, and that its draws have the textbook moments."""

import jax
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.compile.system import compile_model
from mutatis.infer.estimate import estimate
from mutatis.infer.problem import build_problem
from mutatis.infer.regression import NotApplicable, regression_form
from mutatis.measure.data import arrange
from mutatis.quantities.simulate import simulate_data
from mutatis.spec import edits as e
from mutatis.spec.yaml_io import load_model
from mutatis.versions.pipeline import solver
from tests.helpers import assert_mean_matches
from tests.infer.test_routes import setup
from tests.spec.fixtures import toy

FLAT = [e.SetPrior("dynamics.c", prior="flat"), e.SetPrior("dynamics.B", prior="flat"), e.SetPrior("dynamics.Sigma", prior="jeffreys")]


def flat_var(edits=FLAT, n_periods=200):
    model = toy("two_variable")
    for edit in edits:
        model = edit.apply(model)
    _, system, params, solve, data = setup("two_variable", n_periods=n_periods, model=model)
    return build_problem(system, solve, data), data


def test_detects_a_var_and_matches_least_squares():
    problem, data = flat_var()
    form = regression_form(problem)
    assert len(form.blocks) == 1
    block = form.blocks[0]
    assert block.block == "dynamics" and block.covariance == "dynamics.Sigma" and block.regressors.shape == (199, 3)
    y = data.observed
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    ols = np.linalg.lstsq(X, y[1:], rcond=None)[0]
    np.testing.assert_allclose(np.linalg.lstsq(block.regressors, block.responses, rcond=None)[0], ols, rtol=1e-10)


def test_draws_have_the_textbook_moments():
    problem, data = flat_var()
    posterior = estimate(problem, jax.random.key(0), n_draws=4000)
    assert posterior.method == "exact_regression" and posterior.log_marginal_likelihood is None
    values = jax.vmap(problem.parameters)(posterior.particles)
    block = posterior.diagnostics["blocks"]["dynamics"]
    B_hat, S, k = block["least_squares"], block["residual_cross_product"], block["regressors"]
    periods, n = 199, 2
    assert_mean_matches(np.asarray(values["dynamics.Sigma"]).reshape(-1, 4), S.ravel() / (periods - k - n - 1))   # inverse Wishart mean
    assert_mean_matches(np.asarray(values["dynamics.c"]), B_hat[0])
    assert_mean_matches(np.asarray(values["dynamics.B"]).reshape(-1, 4), B_hat[1:].T.ravel())  # row i of B is equation i


def test_proper_coefficient_priors_are_not_this_case():
    problem, _ = flat_var(edits=[e.SetPrior("dynamics.Sigma", prior="jeffreys")])
    with pytest.raises(NotApplicable, match="flat priors"):
        regression_form(problem)


def test_errors_must_be_linear_in_the_coefficients():
    nonlinear = FLAT + [e.ReplaceEquation("dynamics", 0, "y = c + B @ B @ y(-1) + u")]
    problem, _ = flat_var(edits=nonlinear)
    with pytest.raises(NotApplicable, match="not a linear function"):
        regression_form(problem)


# ------------------------------------------------------------------------------------- several blocks, with their own samples

# The two-variable VAR with a second block: a survey answer z regressed on the VAR's current and lagged values, as Read's proxy
# is. Its own shock and covariance; observed only in some periods.
SURVEY = """
model: var-with-survey
time: quarterly
sets:
  series: [a, b]
  surveys: [s]
blocks:
  dynamics:
    outputs: ["y[series]"]
    parameters:
      c:     {shape: [series], prior: flat}
      B:     {shape: [series, series], prior: flat}
      Sigma: {shape: [series, series], domain: positive_definite, prior: jeffreys}
    shocks:
      "u[series]": "multivariate_normal(0, Sigma)"
    equations:
      - "y = c + B @ y(-1) + u"
  survey:
    inputs:  ["y[series]"]
    outputs: ["z[surveys]"]
    parameters:
      k:     {shape: [surveys], prior: flat}
      A:     {shape: [surveys, series], prior: flat}
      G:     {shape: [surveys, series], prior: flat}
      Omega: {shape: [surveys, surveys], domain: positive_definite, prior: jeffreys}
    shocks:
      "v[surveys]": "multivariate_normal(0, Omega)"
    equations:
      - "z = k + A @ y + G @ y(-1) + v"
data:
  a: {observes: "y[a]"}
  b: {observes: "y[b]"}
  Z: {observes: "z[s]"}
"""
SURVEY_PARAMS = {"dynamics.c": [0.1, -0.2], "dynamics.B": [[0.5, 0.1], [0.0, 0.3]], "dynamics.Sigma": [[1.0, 0.3], [0.3, 0.5]],
                 "survey.k": [0.3], "survey.A": [[0.8, -0.4]], "survey.G": [[0.2, 0.1]], "survey.Omega": [[0.25]]}

# A different family: equation-by-equation estimation (as MARTIN's), two single-equation blocks in a chain, each on its own sample.
CHAIN = """
model: two-equations
time: quarterly
sets:
  one: [x]
blocks:
  income:
    inputs:  ["r[one]"]
    outputs: ["inc[one]"]
    parameters:
      a:   {shape: [one], prior: flat}
      b:   {shape: [one, one], prior: flat}
      g:   {shape: [one, one], prior: flat}
      S:   {shape: [one, one], domain: positive_definite, prior: jeffreys}
    shocks:
      "e[one]": "multivariate_normal(0, S)"
    equations:
      - "inc = a + b @ inc(-1) + g @ r(-1) + e"
  spending:
    inputs:  ["inc[one]"]
    outputs: ["spend[one]"]
    parameters:
      a:   {shape: [one], prior: flat}
      h:   {shape: [one, one], prior: flat}
      S:   {shape: [one, one], domain: positive_definite, prior: jeffreys}
    shocks:
      "e[one]": "multivariate_normal(0, S)"
    equations:
      - "spend = a + h @ inc + e"
data:
  INC:   {observes: "inc[x]"}
  SPEND: {observes: "spend[x]"}
  R:     {feeds: "r[x]"}
"""
CHAIN_PARAMS = {"income.a": [0.5], "income.b": [[0.6]], "income.g": [[-0.3]], "income.S": [[0.4]],
                "spending.a": [1.0], "spending.h": [[0.7]], "spending.S": [[0.09]]}


def _two_blocks(text, params, missing_series, missing_periods, n_periods=160, seed=0):
    model = load_model(text)
    system = compile_model(model)
    solve = solver(system)
    values = {k: jax.numpy.asarray(v, dtype=float) for k, v in params.items()}
    outside = np.random.default_rng(seed).normal(size=(n_periods, len(system.outside_labels))) if system.outside_labels else None
    dataset = simulate_data(system, solve(values), jax.random.key(seed), n_periods, outside=outside, start_state=np.zeros(system.n_variables))
    dataset.series[missing_series][missing_periods] = np.nan
    return system, solve, values, arrange(system, dataset, model.data)


@pytest.mark.parametrize("text, params, series, block", [(SURVEY, SURVEY_PARAMS, "Z", "survey"), (CHAIN, CHAIN_PARAMS, "SPEND", "spending")])
def test_blocks_with_their_own_samples_give_the_exact_likelihood(text, params, series, block):
    """With a series missing in some periods, the product of the blocks' error densities, each over its own periods, equals the
    likelihood the Kalman filter computes from the whole model with those values missing: the factorisation is exact."""
    missing = np.zeros(160, dtype=bool)
    missing[50:90] = True
    system, solve, values, data = _two_blocks(text, params, series, missing)
    problem = build_problem(system, solve, data, route="kalman", start="data")
    form = regression_form(problem)
    assert sorted(b.block for b in form.blocks) == sorted(system.outputs_by_block)
    own = next(b for b in form.blocks if b.block == block)
    other = next(b for b in form.blocks if b.block != block)
    assert len(own.periods) == 159 - 40 and len(other.periods) == 159   # 159 periods explained after one presample period

    log_likelihood = 0.0
    for b in form.blocks:
        beta = np.zeros((b.regressors.shape[1], b.responses.shape[1]))
        for name, position, row, regressor in b.coefficients:
            beta[regressor, row] = np.asarray(values[name]).reshape(-1)[position]
        errors = b.responses - b.regressors @ beta
        log_likelihood += ss.multivariate_normal(np.zeros(errors.shape[1]), np.asarray(values[b.covariance])).logpdf(errors).sum()
    kalman = float(problem.log_likelihood(problem.layout.unconstrain(values)))
    assert abs(log_likelihood - kalman) < 1e-8 * abs(kalman), (log_likelihood, kalman)


@pytest.mark.parametrize("text, params, series, block", [(SURVEY, SURVEY_PARAMS, "Z", "survey"), (CHAIN, CHAIN_PARAMS, "SPEND", "spending")])
def test_each_block_is_its_own_least_squares_problem(text, params, series, block):
    """Each block's draws centre on its own least-squares fit over its own periods, computed here directly from the data."""
    missing = np.zeros(160, dtype=bool)
    missing[:30] = True                                  # missing at the start, presample included: no starting state from data
    system, solve, values, data = _two_blocks(text, params, series, missing)
    problem = build_problem(system, solve, data)
    posterior = estimate(problem, jax.random.key(0), n_draws=1000)
    assert posterior.method == "exact_regression"
    summary = posterior.diagnostics["blocks"][block]
    assert summary["periods"] == 160 - 30
    form = {b.block: b for b in regression_form(problem).blocks}[block]
    np.testing.assert_allclose(summary["least_squares"], np.linalg.lstsq(form.regressors, form.responses, rcond=None)[0], rtol=1e-10)
    drawn = jax.vmap(problem.parameters)(posterior.particles)
    covariance = np.asarray(drawn[form.covariance]).reshape(-1)
    T, k = summary["periods"], summary["regressors"]
    assert_mean_matches(covariance[:, None], summary["residual_cross_product"].ravel() / (T - k - 2))   # inverse Wishart mean, one equation


def test_a_series_with_missing_values_may_not_be_used_lagged():
    lagged = SURVEY.replace('"z = k + A @ y + G @ y(-1) + v"', '"z = k + A @ y + G @ y(-1) + 0.5 * z(-1) + v"')
    missing = np.zeros(160, dtype=bool)
    missing[50:90] = True
    system, solve, _, data = _two_blocks(lagged, SURVEY_PARAMS, "Z", missing)
    with pytest.raises(NotApplicable, match="missing values"):
        regression_form(build_problem(system, solve, data, route="kalman", start="data"))
