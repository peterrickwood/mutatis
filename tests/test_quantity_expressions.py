"""Quantities written in the expression language, against direct calculations."""

import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.quantities.expressions import Context, QuantityError, evaluate_quantity, restriction_holds
from mutatis.quantities.recover import recover_shocks
from tests.infer.test_routes import setup


def context(name):
    _, system, params, solve, data = setup(name, n_periods=40)
    first = data.start.first if data.start else 0
    series = {o.series: data.observed[first:, i] for i, o in enumerate(system.observations)}
    start = jnp.asarray(data.start.state) if data.start else None
    return Context(system, solve(params), jnp.asarray(data.observed[first:]), jnp.asarray(data.outside[first:]), start, series), params


def test_responses_of_the_forward_looking_toy():
    ctx, _ = context("forward_looking")
    k = 1 / (1 - 0.5 * 0.8)
    np.testing.assert_allclose(evaluate_quantity("response(p, e, horizons=0..3)", ctx), k * 0.8 ** np.arange(4), rtol=1e-12)
    np.testing.assert_allclose(evaluate_quantity("response(x, e, horizons=2)", ctx), [0.64])
    assert bool(restriction_holds("response(p, e, horizons=0..3)", ">= 0", ctx))
    assert not bool(restriction_holds("response(p, e, horizons=0..3)", "< 1", ctx))


def test_shock_equation_and_estimates_for_the_var():
    ctx, params = context("two_variable")
    # In the toy VAR the shocks u enter one-for-one, so the equation defining u[a] has coefficient 1 on y[a] and 0 on y[b].
    np.testing.assert_allclose(evaluate_quantity("shock_equation(u[a], y[a])", ctx), 1.0, atol=1e-12)
    np.testing.assert_allclose(evaluate_quantity("shock_equation(u[a], y[b])", ctx), 0.0, atol=1e-12)
    shocks, _ = recover_shocks(ctx.solved, ctx.observed, ctx.outside, ctx.start_state)
    np.testing.assert_allclose(evaluate_quantity("shock_estimate(u[b])", ctx), shocks[:, 1])
    a, b = np.asarray(ctx.series["a"]), np.asarray(shocks[:, 0])
    np.testing.assert_allclose(evaluate_quantity("correlation(data(a), shock_estimate(u[a]))", ctx), np.corrcoef(a, b)[0, 1], rtol=1e-10)
    largest = evaluate_quantity("max(j in series: abs(correlation(data(a), shock_estimate(u[j]))))", ctx)
    np.testing.assert_allclose(largest, max(abs(np.corrcoef(a, np.asarray(shocks[:, j]))[0, 1]) for j in range(2)), rtol=1e-10)


def test_unknown_names_are_reported():
    ctx, _ = context("two_variable")
    with pytest.raises(QuantityError, match="not a variable"):
        evaluate_quantity("response(y[c], u[a], horizons=0)", ctx)
    with pytest.raises(QuantityError, match="matches 0 shocks"):
        evaluate_quantity("response(y[a], eps[a], horizons=0)", ctx)
