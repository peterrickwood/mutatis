"""Risk experiment 2 from PLAN.md: the forward-looking solve on many parameter values at once, with derivatives.

Medium tier: run with `pytest -m medium -s tests/numerics/test_linear_re_at_scale.py` to see the timings.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.numerics.linear_re import solve_forward_looking
from mutatis.numerics.status import Status
from tests.numerics import reference
from tests.numerics.test_linear_re import toy

pytestmark = pytest.mark.medium


def _timed(fn, *args):
    jax.block_until_ready(fn(*args))  # compile
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - start


def test_ten_thousand_draws_across_the_edge_of_the_stable_region():
    # The toy has a unique non-exploding solution exactly when |a| < 1 and |rho| < 1. Draws cover both sides of both edges.
    keys = jax.random.split(jax.random.key(0))
    a = jax.random.uniform(keys[0], (10_000,), minval=-1.5, maxval=1.5)
    rho = jax.random.uniform(keys[1], (10_000,), minval=-1.2, maxval=1.2)
    solve = jax.jit(jax.vmap(lambda a, rho: solve_forward_looking(*toy(a, rho))))
    gradient = jax.jit(jax.vmap(jax.grad(lambda a, rho: solve_forward_looking(*toy(a, rho)).P[1, 0], argnums=(0, 1))))
    solutions, seconds_solve = _timed(solve, a, rho)
    (da, drho), seconds_gradient = _timed(gradient, a, rho)
    print(f"\n10,000 draws: solve {seconds_solve:.2f} s, solve and derivatives {seconds_gradient:.2f} s")

    a, rho = np.asarray(a), np.asarray(rho)
    determinate = (np.abs(a) < 1) & (np.abs(rho) < 1)
    status = np.asarray(solutions.status)
    print("status counts:", {Status(s).name: int(np.sum(status == s)) for s in np.unique(status)})
    np.testing.assert_array_equal(status == Status.OK, determinate)  # no false successes, no false failures
    k = 1 / (1 - a * rho)
    np.testing.assert_allclose(np.asarray(solutions.P)[determinate, 1, 0], (rho * k)[determinate], rtol=1e-9)
    # Derivatives of rho / (1 - a rho): with respect to a, rho^2 k^2; with respect to rho, k^2.
    np.testing.assert_allclose(np.asarray(da)[determinate], (rho**2 * k**2)[determinate], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(drho)[determinate], (k**2)[determinate], rtol=1e-8)
    assert np.all(np.isfinite(np.asarray(da))) and np.all(np.isfinite(np.asarray(drho)))
    np.testing.assert_array_equal(np.asarray(da)[~determinate], 0.0)


@pytest.mark.parametrize("n", [10, 20, 40])
def test_cost_grows_with_model_size(n):
    rng = np.random.default_rng(n)
    A, B, C, D, P_true = map(jnp.asarray, reference.model_with_known_solution(n, rng.uniform(-0.95, 0.95, n), rng.uniform(1.05, 3.0, n), rng))
    solve = jax.jit(lambda A: solve_forward_looking(A, B, C, D).P)
    gradient = jax.jit(jax.grad(lambda A: jnp.sum(solve_forward_looking(A, B, C, D).P)))
    P, seconds_solve = _timed(solve, A)
    _, seconds_gradient = _timed(gradient, A)
    print(f"\n{n} variables: solve {1000 * seconds_solve:.1f} ms, derivatives {1000 * seconds_gradient:.1f} ms")
    np.testing.assert_allclose(P, P_true, rtol=1e-7, atol=1e-9)
