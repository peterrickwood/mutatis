"""Restrictions truncate the prior, for every sampler: the second consumer of stage 1's restrictions on quantities, on the
forward-looking toy (a different family from the VAR they were built for)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.compile.system import compile_model
from mutatis.infer.estimate import estimate
from mutatis.infer.problem import build_problem
from mutatis.spec import edits as e
from mutatis.spec.model import Restriction
from mutatis.versions.pipeline import solver
from tests.infer.test_regression import FLAT, flat_var
from tests.infer.test_routes import setup
from tests.spec.fixtures import toy

# The price's response on impact to a demand shock of one unit is 1 / (1 - a rho) (p = x / (1 - a rho)).
LIMIT = 1.7
RESTRICTED = e.AddRestriction("price_responds_strongly", Restriction("response(p, e, horizons=0)", f">= {LIMIT}"))


def _problems():
    model = toy("forward_looking")
    _, system, _, solve, data = setup("forward_looking", n_periods=60)
    restricted_system = compile_model(RESTRICTED.apply(model))
    return build_problem(system, solve, data), build_problem(restricted_system, solver(restricted_system), data)


def _impact(problem, z):
    params = problem.parameters(z)
    return 1.0 / (1.0 - params["price.a"] * params["demand.rho"])


def test_restrictions_make_the_log_prior_minus_infinity_exactly_where_they_fail():
    plain, restricted = _problems()
    z = jax.vmap(plain.draw_prior)(jax.random.split(jax.random.key(0), 400))
    holds = np.asarray(jax.vmap(lambda u: _impact(plain, u))(z)) >= LIMIT
    assert 0.2 < holds.mean() < 0.8, "the restriction should cut the prior, not keep or drop nearly all of it"
    plain_prior, restricted_prior = (np.asarray(jax.vmap(p.log_prior)(z)) for p in (plain, restricted))
    np.testing.assert_array_equal(restricted_prior[holds], plain_prior[holds])
    assert np.all(np.isneginf(restricted_prior[~holds]))
    np.testing.assert_array_equal(np.asarray(jax.vmap(restricted.log_likelihood)(z)), np.asarray(jax.vmap(plain.log_likelihood)(z)))


def test_prior_draws_satisfy_the_restrictions_and_follow_the_truncated_prior():
    plain, restricted = _problems()
    drawn = jax.vmap(restricted.draw_prior)(jax.random.split(jax.random.key(1), 2000))
    impact = np.asarray(jax.vmap(lambda u: _impact(restricted, u))(drawn))
    assert np.all(impact >= LIMIT)
    # Against the plain prior's draws kept where the restriction holds: the same distribution, so the same mean within 5 standard errors.
    reference = np.asarray(jax.vmap(lambda u: _impact(plain, u))(jax.vmap(plain.draw_prior)(jax.random.split(jax.random.key(2), 8000))))
    reference = reference[reference >= LIMIT]
    se = np.sqrt(impact.var() / len(impact) + reference.var() / len(reference))
    assert abs(impact.mean() - reference.mean()) < 5 * se, (impact.mean(), reference.mean(), se)


def test_exact_regression_drops_the_draws_that_break_a_restriction_on_the_coefficients():
    """A VAR without a rotation: the restriction involves only estimated coefficients, so the formula keeps exactly the draws
    where it holds. The response of y[a] to its own error one period on is B[a, a]."""
    rule = e.AddRestriction("persistent", Restriction("response(y[a], u[a], horizons=1)", ">= 0.5"))
    plain, _ = flat_var()
    restricted, _ = flat_var(edits=[*FLAT, rule])
    posterior = estimate(restricted, jax.random.key(0), n_draws=1000)
    B = np.asarray(jax.vmap(restricted.parameters)(jnp.asarray(posterior.particles))["dynamics.B"])
    assert np.all(B[:, 0, 0] >= 0.5)
    unrestricted = estimate(plain, jax.random.key(0), n_draws=1000)
    share = np.mean(np.asarray(jax.vmap(plain.parameters)(jnp.asarray(unrestricted.particles))["dynamics.B"])[:, 0, 0] >= 0.5)
    assert abs(posterior.diagnostics["plausibility"] - share) < 5 * np.sqrt(share * (1 - share) / 1000) + 1e-9


@pytest.mark.medium
def test_the_restricted_posterior_is_the_plain_posterior_cut_down():
    """By SMC on both problems: the restricted posterior's mean of the impact response equals the plain posterior's mean over the
    region where the restriction holds, within the two runs' Monte Carlo errors."""
    plain, restricted = _problems()
    results = {}
    for name, problem in (("plain", plain), ("restricted", restricted)):
        posterior = estimate(problem, jax.random.key(3), method="smc")
        impact = np.asarray(jax.vmap(lambda u: _impact(problem, u))(jnp.asarray(posterior.particles)))
        w = np.asarray(posterior.weights)
        if name == "plain":
            w = w * (impact >= LIMIT)
        w = w / w.sum()
        mean = float(w @ impact)
        results[name] = (mean, np.sqrt(w**2 @ (impact - mean) ** 2), impact, w)
    (m1, se1, _, _), (m2, se2, impact, w) = results["plain"], results["restricted"]
    assert np.all(impact[w > 0] >= LIMIT)
    assert abs(m1 - m2) < 5 * np.hypot(se1, se2) + 1e-3, (m1, m2, se1, se2)


def test_ranges_with_a_sampler_on_a_forward_looking_model():
    """Ranges outside the regression formula (stage 1's bounds, second consumer): a forward-looking model whose two shocks are
    mixed by a rotation with no prior. The price's response to the first shock at horizon h is rho^h / (1 - a rho) times the
    first entry of chol(S) q1, and the restriction asks that entry to be at least 0; so over allowed rotations the response runs
    from 0 to rho^h sqrt(S[d1, d1]) / (1 - a rho), for every posterior draw."""
    from mutatis.versions.pipeline import solver as make_solver

    model = e.SetPrior("demand.Q", prior="none").apply(toy("two_shock_forward_looking"))
    system = compile_model(model)
    solve = make_solver(system)
    true = {"demand.rho": 0.6, "demand.S": [[1.0, 0.3], [0.3, 0.5]], "demand.Q": np.eye(2), "price.a": 0.5}
    from mutatis.measure.data import arrange
    from mutatis.quantities.simulate import simulate_data
    dataset = simulate_data(system, solve({k: jnp.asarray(v, dtype=float) for k, v in true.items()}), jax.random.key(4), 80)
    problem = build_problem(system, solve, arrange(system, dataset, model.data))
    report = ("response(p, e[s1], horizons=0..3)",)
    posterior = estimate(problem, jax.random.key(5), method="smc", n_draws=40, report=report, n_particles=500)
    assert posterior.method == "smc + ranges" and posterior.diagnostics["plausibility"] == 1.0
    lower, upper = posterior.diagnostics["bounds"][report[0]]
    params = jax.vmap(problem.parameters)(jnp.asarray(posterior.particles))
    rho, a, S11 = (np.asarray(params[k]) for k in ("demand.rho", "price.a", "demand.S"))
    expected = rho[:, None] ** np.arange(4) * np.sqrt(S11[:, 0, 0])[:, None] / (1 - a * rho)[:, None]
    np.testing.assert_allclose(upper, expected, rtol=1e-9)
    np.testing.assert_allclose(lower, 0.0, atol=1e-9)
