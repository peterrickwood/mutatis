"""One-step-ahead predictions of the data, by each route, against direct calculations."""

import jax.numpy as jnp
import numpy as np

from mutatis.infer.problem import _initial_from_standard, build_problem
from mutatis.infer.states import hidden_states, one_step_predictions
from mutatis.spec import edits as e
from tests.infer.test_routes import setup
from tests.spec.fixtures import toy


def test_read_off_predicts_from_last_periods_data_and_the_kalman_filter_agrees():
    _, system, params, solve, data = setup("two_variable")
    y = data.observed
    c, B = (np.asarray(params[k]) for k in ("dynamics.c", "dynamics.B"))
    direct = c + y[:-1] @ B.T
    read_off = build_problem(system, solve, data, route="read_off")
    np.testing.assert_allclose(one_step_predictions(read_off, read_off.layout.unconstrain(params)), direct, rtol=1e-12)
    kalman = build_problem(system, solve, data, route="kalman", start="data")   # every series exact: the filter knows the state
    np.testing.assert_allclose(one_step_predictions(kalman, kalman.layout.unconstrain(params)), direct, rtol=1e-10)


def test_kalman_prediction_of_a_hidden_level_equals_a_direct_recursion_with_missing_data():
    """mu(t) = mu(t-1) + eta, y = mu + noise: the prediction of y(t) is the filtered level at t-1. A missing value skips the update."""
    _, system, params, solve, data = setup("hidden_level", n_periods=25, missing=0.2)
    problem = build_problem(system, solve, data, route="kalman")
    q, h = float(params["level.sigma_eta"]) ** 2, float(params["y.sigma_eps"]) ** 2
    level, variance, expected = 0.0, 100.0, []        # initial: mu ~ normal(0, 10), the period before the first observation
    for y in data.observed[:, 0]:
        variance += q
        expected.append(level)
        if not np.isnan(y):
            gain = variance / (variance + h)
            level, variance = level + gain * (y - level), (1 - gain) * variance
    np.testing.assert_allclose(one_step_predictions(problem, problem.layout.unconstrain(params))[:, 0], expected, rtol=1e-10, atol=1e-12)


def test_sampled_shocks_predict_from_the_state_they_imply_one_period_earlier():
    _, system, params, solve, data = setup("hidden_level", n_periods=12)
    problem = build_problem(system, solve, data, route="sample", start="model")
    rng = np.random.default_rng(3)
    values = {**params, "shocks": jnp.asarray(rng.normal(size=(12, 1))), "initial": jnp.asarray(rng.normal(size=1))}
    z = problem.layout.unconstrain(values)
    predicted = np.asarray(one_step_predictions(problem, z))[:, 0]
    states = np.asarray(hidden_states(problem, z))[:, 0]
    start = float(_initial_from_standard(solve(params), values["initial"])[0])
    np.testing.assert_allclose(predicted, np.concatenate([[start], states[:-1]]), rtol=1e-12)   # y is mu itself, plus noise


def test_no_usable_solution_gives_nan():
    model = e.SetPrior("price.a", prior="normal(0.5, 1)").apply(toy("forward_looking"))   # a may now exceed one: many solutions
    _, system, params, solve, data = setup("forward_looking", model=model)
    problem = build_problem(system, solve, data, route="kalman")
    good = one_step_predictions(problem, problem.layout.unconstrain(params))
    bad = one_step_predictions(problem, problem.layout.unconstrain({**params, "price.a": jnp.asarray(1.5)}))
    assert np.all(np.isfinite(np.asarray(good))) and np.all(np.isnan(np.asarray(bad)))
