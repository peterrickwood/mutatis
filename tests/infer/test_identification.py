"""Ranges over rotations: exact bounds against formulas and brute-force search, the one-column check, restricted draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.infer.identification import OneColumn, draw_restricted, linear_in_one_column, one_column_bounds, one_column_extremes
from mutatis.dist import UniformOrthogonal


def test_bounds_in_two_dimensions_match_formulas():
    form = OneColumn(0, np.eye(2), np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]))  # q in the first quadrant
    lower, upper, empty = one_column_bounds(form)
    assert not empty
    np.testing.assert_allclose(upper, [np.sqrt(2), -1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(lower, [1.0, -np.sqrt(2), -1.0], atol=1e-10)


def test_empty_cone():
    _, _, empty = one_column_bounds(OneColumn(0, np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), np.ones((1, 2))))
    assert empty


@pytest.mark.parametrize("seed", range(8))
def test_bounds_contain_and_match_a_dense_search(seed):
    rng = np.random.default_rng(seed)
    n = 3 + seed % 2
    S = rng.normal(size=(rng.integers(1, 5), n))
    C = rng.normal(size=(3, n))
    lower, upper, empty = one_column_bounds(OneColumn(0, S, C))
    q = rng.normal(size=(400_000, n))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    allowed = q[np.all(q @ S.T >= 0, axis=1)]
    if empty:
        assert len(allowed) == 0
        return
    values = allowed @ C.T
    # A search can only find values inside the true range: the bounds must contain everything it finds.
    assert np.all(values.min(axis=0) >= lower - 1e-9) and np.all(values.max(axis=0) <= upper + 1e-9)
    # And each bound must be attained: by a unit vector that satisfies the restrictions and gives exactly that value. Together
    # these pin the bounds down exactly; a search alone cannot, because the extremes sit in the cone's corners, where few of its
    # points fall when the allowed region is small.
    _, _, argmin, argmax, _ = one_column_extremes(OneColumn(0, S, C))
    for vectors, bound in ((argmin, lower), (argmax, upper)):
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-12)
        assert np.all(vectors @ S.T >= -1e-9)
        np.testing.assert_allclose(np.einsum("ij,ij->i", vectors, C), bound, atol=1e-10)


def test_one_column_check():
    n = 3
    A, B = jnp.asarray(np.random.default_rng(0).normal(size=(2, n))), jnp.asarray(np.random.default_rng(1).normal(size=(4, n)))
    form = linear_in_one_column(lambda Q: A @ Q[:, 1], lambda Q: B @ Q[:, 1], n, jax.random.key(0))
    assert form is not None and form.column == 1
    np.testing.assert_allclose(form.S, A) and np.testing.assert_allclose(form.C, B)
    assert linear_in_one_column(lambda Q: A @ Q[:, 1], lambda Q: B @ Q[:, 0], n, jax.random.key(0)) is None       # two columns
    assert linear_in_one_column(lambda Q: (A @ Q[:, 0]) ** 2, lambda Q: B @ Q[:, 0], n, jax.random.key(0)) is None  # not linear
    # Linear in one column only because Q is orthogonal: a row of the inverse of M Q is Q' M^-1, so row 0 depends on column 0 alone.
    M = jnp.asarray(np.random.default_rng(2).normal(size=(n, n)) + 3 * np.eye(n))
    form = linear_in_one_column(lambda Q: jnp.linalg.inv(M @ Q)[0], lambda Q: (M @ Q)[:, 0], n, jax.random.key(1))
    assert form is not None and form.column == 0
    np.testing.assert_allclose(form.S, np.linalg.inv(np.asarray(M)).T, atol=1e-9)


def test_restricted_draws():
    n = 4
    satisfied = lambda Q: Q[0, 0] >= 0.8
    draw = draw_restricted(lambda k, size: UniformOrthogonal(n).sample(k, (size,)), satisfied, jax.random.key(0))
    assert draw is not None and float(draw[0, 0]) >= 0.8
    assert draw_restricted(lambda k, size: UniformOrthogonal(n).sample(k, (size,)), lambda Q: Q[0, 0] > 1.5, jax.random.key(0), max_tries=1000) is None
