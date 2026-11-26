"""Responses, forecasts and hidden states."""

import jax
import jax.numpy as jnp
import numpy as np

from mutatis.infer.problem import build_problem
from mutatis.infer.states import hidden_states
from mutatis.quantities.responses import forecast, impulse_responses
from tests.infer.test_routes import setup


def test_responses_of_the_forward_looking_toy():
    _, system, params, solve, _ = setup("forward_looking")
    solved = solve(params)
    responses = impulse_responses(solved, system.n_variables, len(system.shock_labels), 0, horizons=5)
    k = 1 / (1 - 0.5 * 0.8)
    np.testing.assert_allclose(responses[:, 0, 0], 0.8 ** np.arange(6), rtol=1e-12)        # demand
    np.testing.assert_allclose(responses[:, 1, 0], k * 0.8 ** np.arange(6), rtol=1e-12)    # price = demand / (1 - a rho)
    linear = np.stack([np.linalg.matrix_power(np.asarray(solved.P), h) @ np.asarray(solved.Q) for h in range(6)])
    np.testing.assert_allclose(responses, linear, rtol=1e-12, atol=1e-14)


def test_forecast_without_shocks_follows_the_solution():
    _, system, params, solve, _ = setup("two_variable")
    solved = solve(params)
    path = forecast(solved, jnp.array([1.0, -1.0]), 3, 2)
    expected, x = [], np.array([1.0, -1.0])
    for _ in range(3):
        x = np.asarray(solved.kappa) + np.asarray(solved.P) @ x
        expected.append(x)
    np.testing.assert_allclose(path, np.array(expected), rtol=1e-12)
    paths = forecast(solved, jnp.zeros(2), 4, 2, key=jax.random.key(0), n_paths=7)
    assert paths.shape == (7, 4, 2)


def test_hidden_states_by_each_route():
    _, system, params, solve, data = setup("two_variable")
    problem = build_problem(system, solve, data)
    states = hidden_states(problem, problem.layout.unconstrain(params))
    np.testing.assert_allclose(states, data.observed[1:], rtol=1e-12)  # every series observed exactly: the states are the data
    _, system, params, solve, data = setup("hidden_level", n_periods=15)
    kalman = build_problem(system, solve, data, route="kalman")
    smoothed = hidden_states(kalman, kalman.layout.unconstrain(params))
    assert smoothed.shape == (15, 1) and np.all(np.isfinite(np.asarray(smoothed)))
