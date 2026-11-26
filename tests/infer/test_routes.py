"""The three ways of handling shocks agree with each other and with direct calculations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.compile.system import compile_model
from mutatis.infer.problem import ProblemError, build_problem, choose_route
from mutatis.measure.data import arrange
from mutatis.quantities.simulate import simulate_data
from mutatis.spec import edits as e
from mutatis.versions.pipeline import solver
from tests.spec.fixtures import toy

PARAMS = {
    "hidden_level": {"level.sigma_eta": 0.3, "y.sigma_eps": 0.5},
    "forward_looking": {"demand.rho": 0.8, "demand.sigma_x": 0.7, "price.a": 0.5},
    "two_variable": {"dynamics.c": [0.1, -0.2], "dynamics.B": [[0.5, 0.1], [0.0, 0.3]], "dynamics.Sigma": [[1.0, 0.3], [0.3, 0.5]]},
    "error_correction": {"spending.alpha0": 0.1, "spending.gamma": -0.2, "spending.beta1": 0.9, "spending.alpha1": 0.3, "spending.sigma": 0.4},
    "sectors": {"capital.delta": 0.1, "investment.phi": 1.2, "investment.sigma_u": 0.5, "rate.rho": 0.6, "rate.sigma_e": 0.3},
}


def setup(name, n_periods=30, model=None, missing=0.0, seed=0):
    model = model or toy(name)
    system = compile_model(model)
    params = {k: jnp.asarray(v, dtype=float) for k, v in PARAMS[name].items()}
    solve = solver(system)
    outside = None
    if system.outside_labels:
        outside = np.cumsum(np.random.default_rng(seed).normal(size=(n_periods, len(system.outside_labels))), axis=0)
    start = None if solve(params).initial_gaussian() is not None else np.zeros(system.n_variables)
    dataset = simulate_data(system, solve(params), jax.random.key(seed), n_periods, outside=outside, start_state=start)
    assert all(np.all(np.isfinite(v)) for v in dataset.series.values()), "simulated data must be finite before any is removed"
    if missing:
        rng = np.random.default_rng(seed + 1)
        for values in dataset.series.values():
            values[rng.uniform(size=len(values)) < missing] = np.nan
    return model, system, params, solve, arrange(system, dataset, model.data)


def theta(problem, params):
    """The unconstrained vector for given parameter values (problems with parameters only)."""
    return problem.layout.unconstrain(params)


@pytest.mark.parametrize("name, expected", [("two_variable", "read_off"), ("error_correction", "read_off"), ("hidden_level", "kalman"),
                                            ("forward_looking", "kalman"), ("sectors", "kalman")])
def test_route_chosen_automatically(name, expected):
    _, system, _, _, data = setup(name)
    assert choose_route(system, data) == expected


def test_fat_tails_switch_to_sampling_the_shocks():
    model = e.SetShock("demand.e", "student_t(4, 0, sigma_x)").apply(toy("forward_looking"))
    _, system, _, _, data = setup("forward_looking", model=e.SetShock("demand.e", "normal(0, sigma_x)").apply(model))
    assert choose_route(compile_model(model), data) == "sample"


def test_var_read_off_equals_kalman_equals_direct_calculation():
    _, system, params, solve, data = setup("two_variable")
    read_off = build_problem(system, solve, data, route="read_off")
    kalman = build_problem(system, solve, data, route="kalman", start="data")
    z = theta(read_off, params)
    y = data.observed
    c, B, Sigma = (np.asarray(params[k]) for k in ("dynamics.c", "dynamics.B", "dynamics.Sigma"))
    errors = y[1:] - c - y[:-1] @ B.T
    direct = ss.multivariate_normal(np.zeros(2), Sigma).logpdf(errors).sum()
    np.testing.assert_allclose(read_off.log_likelihood(z), direct, rtol=1e-10)
    np.testing.assert_allclose(kalman.log_likelihood(z), direct, rtol=1e-10)


def test_error_correction_read_off_equals_direct_calculation():
    _, system, params, solve, data = setup("error_correction")
    problem = build_problem(system, solve, data)
    spend, income = data.observed[:, 0], data.outside[:, 0]
    p = {k.split(".")[1]: float(v) for k, v in params.items()}
    errors = np.diff(spend) - (p["alpha0"] + p["gamma"] * (spend[:-1] - p["beta1"] * income[:-1]) + p["alpha1"] * np.diff(income))
    np.testing.assert_allclose(problem.log_likelihood(theta(problem, params)), ss.norm(0, p["sigma"]).logpdf(errors).sum(), rtol=1e-10)
    kalman = build_problem(system, solve, data, route="kalman", start="data")
    np.testing.assert_allclose(kalman.log_likelihood(theta(kalman, params)), ss.norm(0, p["sigma"]).logpdf(errors).sum(), rtol=1e-10)


@pytest.mark.medium
@pytest.mark.parametrize("name, missing", [("hidden_level", 0.0), ("hidden_level", 0.2), ("forward_looking", 0.0), ("sectors", 0.1)])
def test_sampling_the_shocks_integrates_to_the_kalman_likelihood(name, missing):
    # With normal noise, the log density with shocks as unknowns is quadratic in them, so integrating them out is exact:
    # log of the integral = value at the peak + (d/2) log(2 pi) - (1/2) log det(-Hessian). That must equal the Kalman likelihood.
    _, system, params, solve, data = setup(name, n_periods=20, missing=missing)
    kalman = build_problem(system, solve, data, route="kalman", start="model")
    sampled = build_problem(system, solve, data, route="sample", start="model")
    z_theta = theta(kalman, params)
    d = sampled.size - z_theta.size
    total = lambda u: sampled.log_posterior(jnp.concatenate([z_theta, u]))
    u0 = jnp.zeros(d)
    H = jax.hessian(total)(u0)
    peak = u0 - jnp.linalg.solve(H, jax.grad(total)(u0))  # one Newton step is exact for a quadratic
    log_integral = total(peak) + 0.5 * d * np.log(2 * np.pi) - 0.5 * jnp.linalg.slogdet(-H)[1]
    np.testing.assert_allclose(log_integral - kalman.log_prior(z_theta), kalman.log_likelihood(z_theta), rtol=1e-8)


def test_prior_draws_have_the_right_shape_and_finite_densities():
    _, system, params, solve, data = setup("forward_looking")
    for route in ("kalman", "sample"):
        problem = build_problem(system, solve, data, route=route, start="model")
        z = problem.draw_prior(jax.random.key(1))
        assert z.shape == (problem.size,)
        assert np.isfinite(float(problem.log_prior(z)))


def test_starting_from_the_data_needs_exact_observations():
    _, system, _, solve, data = setup("hidden_level")
    with pytest.raises(ProblemError, match="cannot be read from the data"):
        build_problem(system, solve, data, start="data")
