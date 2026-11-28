"""Ranges over rotations: exact bounds against formulas and brute-force search, the one-column check, restricted draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.infer.identification import OneColumn, draw_restricted, linear_in_one_column, one_column_bounds, one_column_extremes, rotation_bounds
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


@pytest.mark.parametrize("seed", range(4))
def test_rotation_search_reaches_the_exact_bounds_where_both_apply(seed):
    # Risk 4 in PLAN.md: does the general search stop at a local best? Compare with the exact one-column method.
    rng = np.random.default_rng(10 + seed)
    n = 4
    S, C = jnp.asarray(rng.normal(size=(3, n))), jnp.asarray(rng.normal(size=(2, n)))
    lower, upper, empty = one_column_bounds(OneColumn(0, np.asarray(S), np.asarray(C)))
    found_lower, found_upper, argmin, argmax, found_empty = rotation_bounds(lambda Q: S @ Q[:, 0], lambda Q: C @ Q[:, 0], n, jax.random.key(seed))
    assert empty == found_empty
    if not empty:
        assert np.all(found_lower >= lower - 1e-9) and np.all(found_upper <= upper + 1e-9)   # never outside the true range
        np.testing.assert_allclose(found_lower, lower, atol=1e-8)
        np.testing.assert_allclose(found_upper, upper, atol=1e-8)
        for Q, value, i in [(Q, v, i) for i, (Q, v) in enumerate(zip(argmax, found_upper))] + [(Q, v, i) for i, (Q, v) in enumerate(zip(argmin, found_lower))]:
            np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-12)                       # attained at an allowed rotation
            assert np.all(np.asarray(S) @ Q[:, 0] >= -1e-12) and abs(float(np.asarray(C)[i] @ Q[:, 0]) - value) < 1e-12


def _proxy_bound_by_reduction(S1, a, c, n, rng):
    """The exact largest c'q1 when the restrictions are S1 q1 >= 0 plus Read's proxy restrictions, a'q1 >= |a'qj| for every
    other column j. An orthonormal completion q2..qn with |a'qj| <= a'q1 exists exactly when a'q1 >= |a| / sqrt(n) (the other
    columns share a's remaining length, |a|^2 - (a'q1)^2, at best equally), so the problem is convex in q1 alone: largest c'q
    over the cone {S1 q >= 0, a'q >= |a| |q| / sqrt(n)} and |q| <= 1, solved here from many starts by SLSQP."""
    from scipy.optimize import minimize

    cone = [{"type": "ineq", "fun": lambda q: S1 @ q}, {"type": "ineq", "fun": lambda q: a @ q - np.linalg.norm(a) * np.linalg.norm(q) / np.sqrt(n)},
            {"type": "ineq", "fun": lambda q: 1.0 - q @ q}]
    best = -np.inf
    for start in rng.normal(size=(8, n)):   # a convex problem: a few starts guard against a failed solve
        result = minimize(lambda q: -(c @ q), 0.5 * start / np.linalg.norm(start), method="SLSQP", constraints=cone, options={"ftol": 1e-14, "maxiter": 500})
        if result.success and all(np.all(np.atleast_1d(con["fun"](result.x)) >= -1e-10) for con in cone):
            best = max(best, c @ result.x)
    return best


@pytest.mark.parametrize("seed", range(4))
def test_rotation_search_with_restrictions_on_several_columns_matches_the_exact_reduction(seed):
    """Read's proxy restrictions involve every column of the rotation, so the one-column method cannot apply; the search must
    still reach the exact bounds, found here by an independent reduction to a convex problem."""
    rng = np.random.default_rng(20 + seed)
    n = 4
    S1, a = rng.normal(size=(2, n)), rng.normal(size=n)
    C = np.stack([a + 0.5 * rng.normal(size=n), -a + 0.5 * rng.normal(size=n)])
    restrict = lambda Q: jnp.concatenate([jnp.asarray(S1) @ Q[:, 0], jnp.asarray(a) @ Q[:, :1].repeat(n - 1, axis=1) - jnp.asarray(a) @ Q[:, 1:],
                                          jnp.asarray(a) @ Q[:, :1].repeat(n - 1, axis=1) + jnp.asarray(a) @ Q[:, 1:], (jnp.asarray(a) @ Q[:, 0])[None]])
    quantities = lambda Q: jnp.asarray(C) @ Q[:, 0]
    lower, upper, _, argmax, empty = rotation_bounds(restrict, quantities, n, jax.random.key(seed))
    # Nonempty exactly when some q1 allowed by S1 is within 60 degrees of a in 4 dimensions (cos >= 1/sqrt(n)).
    closest = _proxy_bound_by_reduction(S1, np.zeros(n), a / np.linalg.norm(a), n, rng)
    assert empty == (closest < 1 / np.sqrt(n)), (empty, closest)
    if empty:
        return
    compared = 0
    for i in range(2):
        for sign, found in ((1.0, upper[i]), (-1.0, -lower[i])):
            exact = _proxy_bound_by_reduction(S1, a, sign * C[i], n, rng)
            if exact > 1e-6:     # the reduction is exact where the bound is positive (then attained on the unit sphere)
                assert found <= exact + 1e-8, (found, exact)                                   # never outside the true range
                assert abs(found - exact) < 1e-7, (found, exact)
                compared += 1
    assert compared >= 2
    assert np.all(np.asarray(restrict(jnp.asarray(argmax[0]))) >= -1e-12)


@pytest.mark.parametrize("seed", [11, 45])
def test_the_softened_search_reaches_extremes_where_many_restrictions_meet(seed):
    """Cases found by searching random instances (decision 53): with proxy restrictions, an extreme can lie where every one of
    them binds at once (a'q1 = |a| / sqrt(n)), which the barrier search, staying inside, is pushed away from. Alone it falls
    short here by more than 1e-3; with the softened search the bounds equal the exact reduction's."""
    rng = np.random.default_rng(1000 + seed)
    n = 4
    S1, a, C = rng.normal(size=(5, n)), rng.normal(size=n), rng.normal(size=(6, n))
    restrict = lambda Q: jnp.concatenate([jnp.asarray(S1) @ Q[:, 0], jnp.asarray(a) @ Q[:, :1].repeat(n - 1, axis=1) - jnp.asarray(a) @ Q[:, 1:],
                                          jnp.asarray(a) @ Q[:, :1].repeat(n - 1, axis=1) + jnp.asarray(a) @ Q[:, 1:], (jnp.asarray(a) @ Q[:, 0])[None]])
    quantities = lambda Q: jnp.asarray(C) @ Q[:, 0]
    barrier_only = rotation_bounds(restrict, quantities, n, jax.random.key(seed), soft_starts=0)
    lower, upper, argmin, argmax, empty = rotation_bounds(restrict, quantities, n, jax.random.key(seed))
    assert not empty
    shortfall, compared = 0.0, 0
    for i in range(len(C)):
        for sign, found, alone in ((1.0, upper[i], barrier_only[1][i]), (-1.0, -lower[i], -barrier_only[0][i])):
            exact = _proxy_bound_by_reduction(S1, a, sign * C[i], n, rng)
            if exact > 1e-6:     # the reduction is exact where the bound is positive
                assert found <= exact + 1e-9 and abs(found - exact) < 1e-8, (i, sign, found, exact)
                shortfall, compared = max(shortfall, exact - alone), compared + 1
    assert compared >= 6 and shortfall > 1e-3
    for Q in (*argmin, *argmax):
        assert np.all(np.asarray(restrict(jnp.asarray(Q))) >= -1e-12)


def test_rotation_search_with_a_nonlinear_quantity():
    """A quantity that is not linear in the rotation goes through the functions themselves: Q[0,0]^2 over all rotations is 0 to 1."""
    lower, upper, _, _, empty = rotation_bounds(lambda Q: jnp.zeros(0), lambda Q: jnp.stack([Q[0, 0] ** 2]), 3, jax.random.key(0), pool=500)
    assert not empty and abs(lower[0]) < 1e-9 and abs(upper[0] - 1.0) < 1e-9


def test_rotation_search_finds_a_tiny_allowed_set_and_reports_an_empty_one():
    """Q[0,0] >= 0.9999 holds for about 1 rotation in a million in 4 dimensions, so no random start is allowed: the first phase
    must find one. Q[0,0] >= 0.9 together with Q[0,0] <= -0.9 allows nothing."""
    quantities = lambda Q: Q[:, 0]
    lower, upper, _, _, empty = rotation_bounds(lambda Q: jnp.stack([Q[0, 0] - 0.9999]), quantities, 4, jax.random.key(1), pool=2000)
    assert not empty and abs(upper[0] - 1.0) < 1e-9 and abs(lower[0] - 0.9999) < 1e-9
    *_, empty = rotation_bounds(lambda Q: jnp.stack([Q[0, 0] - 0.9, -Q[0, 0] - 0.9]), quantities, 4, jax.random.key(1), pool=2000)
    assert empty
