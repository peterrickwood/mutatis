"""The exact regression formula: when it applies, and that its draws have the textbook moments."""

import jax
import numpy as np
import pytest

from mutatis.infer.estimate import estimate
from mutatis.infer.problem import build_problem
from mutatis.infer.regression import NotApplicable, regression_form
from mutatis.spec import edits as e
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
    assert form.covariance == "dynamics.Sigma" and form.regressors.shape == (199, 3)
    y = data.observed
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    ols = np.linalg.lstsq(X, y[1:], rcond=None)[0]
    np.testing.assert_allclose(np.linalg.lstsq(form.regressors, form.responses, rcond=None)[0], ols, rtol=1e-10)


def test_draws_have_the_textbook_moments():
    problem, data = flat_var()
    posterior = estimate(problem, jax.random.key(0), n_draws=4000)
    assert posterior.method == "exact_regression" and posterior.log_marginal_likelihood is None
    values = jax.vmap(problem.parameters)(posterior.particles)
    B_hat, S, k = posterior.diagnostics["least_squares"], posterior.diagnostics["residual_cross_product"], posterior.diagnostics["regressors"]
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
