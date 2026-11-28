"""Root finding and simulation."""

import jax
import jax.numpy as jnp
import numpy as np

from mutatis.numerics.kalman import stationary_covariance
from mutatis.numerics.rootfind import find_root
from mutatis.numerics.simulate import simulate
from mutatis.numerics.status import Status


def test_root_and_its_derivative():
    root = lambda a: find_root(lambda x, a: x**2 - a, jnp.array(1.0), a)
    x, status = root(2.0)
    assert int(status) == Status.OK
    np.testing.assert_allclose(x, np.sqrt(2.0), rtol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda a: root(a)[0])(2.0), 1 / (2 * np.sqrt(2.0)), rtol=1e-10)  # implicit function theorem


def test_system_of_equations():
    fn = lambda v, _: jnp.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[0] - v[1]])
    x, status = find_root(fn, jnp.array([1.0, 0.5]))
    assert int(status) == Status.OK
    np.testing.assert_allclose(x, [np.sqrt(2.0), np.sqrt(2.0)], rtol=1e-12)


def test_failure_is_a_status_not_an_exception():
    x, status = find_root(lambda x, _: x**2 + 1.0, jnp.array(0.5))  # no real root
    assert int(status) == Status.NOT_CONVERGED
    statuses = jax.vmap(lambda a: find_root(lambda x, a: x**2 - a, jnp.array(1.0), a)[1])(jnp.array([4.0, -1.0, 9.0]))
    np.testing.assert_array_equal(statuses, [Status.OK, Status.NOT_CONVERGED, Status.OK])


def test_simulate_matches_a_plain_loop_with_two_inputs():
    rng = np.random.default_rng(0)
    P, Q, R = rng.normal(size=(2, 2)) * 0.5, rng.normal(size=(2, 1)), rng.normal(size=(2, 3))
    shocks, outside = rng.normal(size=(10, 1)), rng.normal(size=(10, 3))
    path = simulate(lambda s, u: P @ s + Q @ u[0] + R @ u[1], jnp.zeros(2), (jnp.asarray(shocks), jnp.asarray(outside)))
    expected, s = [], np.zeros(2)
    for t in range(10):
        s = P @ s + Q @ shocks[t] + R @ outside[t]
        expected.append(s)
    np.testing.assert_allclose(path, np.array(expected), rtol=1e-12)


def test_long_simulation_settles_at_the_stationary_covariance():
    # Two methods that share no code: a long simulated path, and the doubling formula. Tolerance from the sampling error.
    T, n_periods = jnp.array([[0.5, 0.2], [-0.1, 0.3]]), 200_000
    shocks = jax.random.normal(jax.random.key(0), (n_periods, 2))
    path = np.asarray(simulate(lambda s, e: T @ s + e, jnp.zeros(2), shocks))[1000:]
    X, _ = stationary_covariance(T, jnp.eye(2))
    # The path is autocorrelated, so draws are not independent. The standard error comes from batch means: the spread of the
    # covariance across 100 long stretches of the path, which are nearly independent of each other.
    batches = np.array([np.cov(chunk.T) for chunk in np.array_split(path, 100)])
    standard_error = batches.std(axis=0, ddof=1) / np.sqrt(len(batches))
    np.testing.assert_array_less(np.abs(batches.mean(axis=0) - np.asarray(X)), 5 * standard_error)


def test_second_derivative_of_a_root():
    """Derivatives of the root pass through the implicit function theorem at every order: x = sqrt(a)."""
    root = lambda a: find_root(lambda x, a: x**2 - a, jnp.array(1.0), a)[0]
    np.testing.assert_allclose(jax.hessian(root)(2.0), -0.25 * 2.0**-1.5, rtol=1e-9)
    np.testing.assert_allclose(jax.jacfwd(jax.grad(root))(2.0), -0.25 * 2.0**-1.5, rtol=1e-9)


def test_a_failed_search_has_a_finite_stand_in_and_a_zero_derivative():
    """Where the search fails or the root is not finite, the status says so; the value is the start (non-finite entries as 0) and
    its derivative is zero, so a sampler's gradient through it does not raise (it once did, inside Optimistix's own rule)."""
    fn = lambda x, a: jnp.exp(x) - a          # no root for a <= 0: Newton runs off to minus infinity
    value, status = find_root(fn, jnp.array(1.0), -1.0)
    assert int(status) == Status.NOT_CONVERGED and float(value) == 1.0
    assert float(jax.grad(lambda a: find_root(fn, jnp.array(1.0), a)[0])(-1.0)) == 0.0
    value, status = find_root(lambda x, a: x**2 - a, jnp.array([jnp.inf, 2.0]), 4.0)
    assert int(status) == Status.NOT_CONVERGED and np.all(np.asarray(value) == [0.0, 2.0])
    grads = jax.vmap(jax.grad(lambda a: find_root(fn, jnp.array(1.0), a)[0]))(jnp.array([2.0, -1.0]))
    np.testing.assert_allclose(grads, [0.5, 0.0], rtol=1e-10)
