"""Solving compiled linear systems: the step-forward rule, the observation rule, noise and the starting distribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
import scipy.stats as ss

from mutatis.compile.system import compile_model
from mutatis.numerics.status import Status
from mutatis.solve.linear import SolveError, solve_linear
from mutatis.spec import edits as e
from tests.helpers import assert_covariance_matches
from tests.spec.fixtures import toy

FORWARD = {"demand.rho": jnp.array(0.8), "demand.sigma_x": jnp.array(0.7), "price.a": jnp.array(0.5)}


def test_forward_looking_toy():
    solved = solve_linear(compile_model(toy("forward_looking")), FORWARD)
    k = 1 / (1 - 0.5 * 0.8)
    assert int(solved.status) == Status.OK
    np.testing.assert_allclose(solved.P, [[0.8, 0], [0.8 * k, 0]], rtol=1e-12)
    np.testing.assert_allclose(solved.Q, [[1], [k]], rtol=1e-12)
    np.testing.assert_allclose(solved.kappa, 0, atol=1e-14)
    np.testing.assert_allclose(solved.Z, np.eye(2))
    noise = solved.gaussian_noise()
    np.testing.assert_allclose(noise.shock_cov, [[0.49]])
    np.testing.assert_allclose(noise.error_var, [0.01, 0.01])
    mean, cov = solved.initial_gaussian()  # the long-run distribution
    np.testing.assert_allclose(cov, scipy.linalg.solve_discrete_lyapunov(np.asarray(solved.P), np.asarray(solved.Q @ noise.shock_cov @ solved.Q.T)), rtol=1e-9)
    np.testing.assert_allclose(mean, 0, atol=1e-14)


def test_no_unique_solution_is_a_status():
    solved = solve_linear(compile_model(toy("forward_looking")), {**FORWARD, "price.a": jnp.array(1.5)})
    assert int(solved.status) == Status.MANY_STABLE_SOLUTIONS


def test_outside_variable_enters_through_R():
    system = compile_model(toy("error_correction"))
    params = {f"spending.{k}": jnp.array(v) for k, v in dict(alpha0=0.1, gamma=-0.2, beta1=0.9, alpha1=0.3, sigma=1.0).items()}
    solved = solve_linear(system, params)
    np.testing.assert_allclose(solved.R[:, 0], [0.3, 1.0])            # spend moves by alpha1 with income; the copy equals income
    np.testing.assert_allclose(solved.P, [[1 - 0.2, 0.2 * 0.9 - 0.3], [0, 0]], atol=1e-14)
    np.testing.assert_allclose(solved.kappa, [0.1, 0])
    assert solved.initial_gaussian() is None  # an outside variable has no long-run distribution of its own: start from the data
    assert solved.exact == (True,)


def test_vector_shocks_and_their_probability():
    system = compile_model(toy("two_variable"))
    Sigma = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    solved = solve_linear(system, {"dynamics.c": jnp.zeros(2), "dynamics.B": 0.5 * jnp.eye(2), "dynamics.Sigma": Sigma})
    u = jnp.array([0.2, -0.4])
    np.testing.assert_allclose(solved.shock_log_prob(u), ss.multivariate_normal([0, 0], np.asarray(Sigma)).logpdf(np.asarray(u)), rtol=1e-12)
    np.testing.assert_allclose(solved.gaussian_noise().shock_cov, Sigma)
    draws = solved.sample_shocks(jax.random.key(0), 20_000)
    assert draws.shape == (20_000, 2)
    assert_covariance_matches(draws, [0.0, 0.0], Sigma)


def test_initial_priors_for_a_drifting_level():
    system = compile_model(toy("hidden_level"))
    solved = solve_linear(system, {"level.sigma_eta": jnp.array(0.3), "y.sigma_eps": jnp.array(0.5)})
    mean, cov = solved.initial_gaussian()
    np.testing.assert_allclose(mean, [0.0])
    np.testing.assert_allclose(cov, [[100.0]])
    np.testing.assert_allclose(solved.error_log_prob(jnp.array([1.0]), jnp.array([0.4])), ss.norm(0, 0.5).logpdf(0.6), rtol=1e-12)
    np.testing.assert_allclose(solved.error_log_prob(jnp.array([jnp.nan]), jnp.array([0.4])), 0.0)  # missing data contributes nothing


def test_fat_tailed_shocks_switch_off_gaussian_noise():
    model = e.SetShock("demand.e", "student_t(4, 0, sigma_x)").apply(toy("forward_looking"))
    solved = solve_linear(compile_model(model), FORWARD)
    assert solved.gaussian_noise() is None
    assert solved.initial_gaussian() is None
    np.testing.assert_allclose(solved.shock_log_prob(jnp.array([0.3])), ss.t(4, 0, 0.7).logpdf(0.3), rtol=1e-12)


def test_limits_are_reported():
    nonlinear = e.ReplaceEquation("demand", 0, "x = rho * x(-1)^2 + e").apply(toy("forward_looking"))
    with pytest.raises(SolveError, match="not linear"):
        solve_linear(compile_model(nonlinear), FORWARD)


def test_many_parameter_values_at_once_with_derivatives():
    system = compile_model(toy("forward_looking"))
    a = jnp.linspace(0.1, 0.9, 9)
    solve = lambda a: solve_linear(system, {**FORWARD, "price.a": a}).P[1, 0]
    np.testing.assert_allclose(jax.vmap(solve)(a), 0.8 / (1 - a * 0.8), rtol=1e-12)
    np.testing.assert_allclose(jax.vmap(jax.grad(solve))(a), 0.64 / (1 - a * 0.8) ** 2, rtol=1e-10)
