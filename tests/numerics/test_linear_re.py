"""The linear solves: known answers, an independent SciPy reference, the existence and uniqueness check, and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.numerics.linear_re import cyclic_reduction, roots, solve_backward, solve_forward_looking, solve_sylvester
from mutatis.numerics.status import Status
from tests.numerics import reference


def toy(a, rho):
    """The forward-looking toy: x = rho x(-1) + e and p = x + a E p(+1), with variables [x, p].

    Its solution is p = x / (1 - a rho), so P = [[rho, 0], [rho / (1 - a rho), 0]] and Q = [[1], [1 / (1 - a rho)]].
    """
    A = jnp.array([[0.0, 0.0], [0.0, -a]])
    B = jnp.array([[1.0, 0.0], [-1.0, 1.0]])
    C = jnp.array([[-rho, 0.0], [0.0, 0.0]])
    D = jnp.array([[-1.0], [0.0]])
    return A, B, C, D


def toy_solution(a, rho):
    return np.array([[rho, 0.0], [rho / (1 - a * rho), 0.0]]), np.array([[1.0], [1 / (1 - a * rho)]])


@pytest.mark.parametrize("a, rho", [(0.5, 0.8), (0.99, 0.9), (0.2, -0.7), (0.0, 0.5), (-0.9, 0.3)])
def test_toy_known_answer(a, rho):
    P, Q, status = solve_forward_looking(*toy(a, rho))
    P_true, Q_true = toy_solution(a, rho)
    assert int(status) == Status.OK
    np.testing.assert_allclose(P, P_true, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Q, Q_true, rtol=1e-12, atol=1e-14)


def test_toy_with_a_random_walk():
    # x = x(-1) + e has a root exactly on the unit circle. Then E x(t+k) = x(t), so p = x / (1 - a).
    a = 0.6
    P, Q, status = solve_forward_looking(*toy(a, 1.0))
    assert int(status) == Status.OK
    np.testing.assert_allclose(P, [[1.0, 0.0], [1 / (1 - a), 0.0]], rtol=1e-10)


@pytest.mark.parametrize("a, rho, expected", [
    (1.5, 0.5, Status.MANY_STABLE_SOLUTIONS),   # the price equation's own root 1/a lies inside the unit circle
    (-1.5, 0.3, Status.MANY_STABLE_SOLUTIONS),  # the same, with 1/a negative
    (0.5, 1.2, Status.NO_STABLE_SOLUTION),      # demand explodes and the price cannot undo it
])
def test_toy_existence_and_uniqueness(a, rho, expected):
    P, Q, status = solve_forward_looking(*toy(a, rho))
    assert int(status) == expected
    assert float(jnp.abs(P).sum() + jnp.abs(Q).sum()) == 0.0


@pytest.mark.parametrize("n", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", range(4))
def test_random_models_with_known_solution(n, seed):
    rng = np.random.default_rng(seed)
    inside, outside = rng.uniform(-0.95, 0.95, n), rng.choice([-1, 1], n) * rng.uniform(1.1, 3.0, n)
    A, B, C, D, P_true = reference.model_with_known_solution(n, inside, outside, rng)
    P, Q, status = solve_forward_looking(*map(jnp.asarray, (A, B, C, D)))
    assert int(status) == Status.OK
    np.testing.assert_allclose(P, P_true, rtol=1e-8, atol=1e-10)
    P_ref, Q_ref = reference.stable_solution(A, B, C, D)
    np.testing.assert_allclose(P, P_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Q, Q_ref, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("seed", range(12))
def test_classification_agrees_with_counting_roots(seed):
    # Move one root across the unit circle and check the status matches a count of SciPy's roots.
    rng = np.random.default_rng(100 + seed)
    n = 3
    inside, outside = rng.uniform(-0.9, 0.9, n), rng.choice([-1, 1], n) * rng.uniform(1.2, 2.5, n)
    case = seed % 3
    if case == 1:
        outside[0] = 0.5 * np.sign(outside[0])     # one extra root inside: n + 1 roots inside
    elif case == 2:
        inside[0] = 1.1 * np.sign(inside[0])       # one root moves out: n - 1 roots inside
    A, B, C, D, _ = reference.model_with_known_solution(n, inside, outside, rng)
    values, _ = reference.roots_of_quadratic(A, B, C)
    count = int(np.sum(np.isfinite(values) & (np.abs(values) < 1.0)))
    expected = Status.OK if count == n else Status.MANY_STABLE_SOLUTIONS if count > n else Status.NO_STABLE_SOLUTION
    _, _, status = solve_forward_looking(*map(jnp.asarray, (A, B, C, D)))
    assert int(status) == expected


def test_static_equation_and_singular_lead_matrix():
    # Three variables: x = rho x(-1) + e, p = x + a E p(+1), and a static identity y = x + p. A has two zero rows.
    a, rho = 0.4, 0.7
    A = jnp.zeros((3, 3)).at[1, 1].set(-a)
    B = jnp.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0]])
    C = jnp.zeros((3, 3)).at[0, 0].set(-rho)
    D = jnp.array([[-1.0], [0.0], [0.0]])
    P, Q, status = solve_forward_looking(A, B, C, D)
    assert int(status) == Status.OK
    k = 1 / (1 - a * rho)
    np.testing.assert_allclose(P[:, 0], [rho, rho * k, rho * (1 + k)], rtol=1e-12)
    np.testing.assert_allclose(Q[:, 0], [1.0, k, 1 + k], rtol=1e-12)


def test_backward_solve():
    B = jnp.array([[1.0, 0.0], [-0.5, 1.0]])
    C = jnp.array([[-1.2, 0.0], [0.0, -0.3]])
    D = jnp.eye(2)
    P, Q, status = solve_backward(B, C, D)
    assert int(status) == Status.OK  # explosive (1.2) but unique: allowed for a model without expectations
    np.testing.assert_allclose(B @ P, -C, atol=1e-14)
    np.testing.assert_allclose(B @ Q, -D, atol=1e-14)
    _, _, singular = solve_backward(jnp.zeros((2, 2)), C, D)
    assert int(singular) == Status.SINGULAR


def test_wrong_answers_from_the_reduction_are_caught():
    # With a clear gap between the 2nd and 3rd roots (0.5, 0.6 | 2, 3) the reduction finds the solution.
    rng = np.random.default_rng(7)
    A, B, C, D, P_true = map(jnp.asarray, reference.model_with_known_solution(2, np.array([0.5, 0.6]), np.array([2.0, 3.0]), rng))
    X, settled = cyclic_reduction(C, B, A)
    assert bool(settled)
    np.testing.assert_allclose(X, P_true, rtol=1e-10)
    # With roots 0.5, 0.8 | -0.8, 3 the 2nd and 3rd are the same size. The reduction can then stop on a matrix that does not solve
    # the equation at all, so the solve must not trust "settled" alone: the residual check and the root count must catch it.
    A, B, C, D, _ = map(jnp.asarray, reference.model_with_known_solution(2, np.array([0.5, 0.8]), np.array([-0.8, 3.0]), rng))
    X, settled = cyclic_reduction(C, B, A)
    residual = float(jnp.abs(A @ X @ X + B @ X + C).max())
    assert (not bool(settled)) or residual > 1.0
    assert int(solve_forward_looking(A, B, C, D).status) == Status.MANY_STABLE_SOLUTIONS


@pytest.mark.parametrize("seed", range(6))
def test_roots_match_scipy(seed):
    rng = np.random.default_rng(300 + seed)
    n = 3
    A, B, C = (rng.normal(size=(n, n)) for _ in range(3))
    if seed % 2:
        A[0] = 0.0  # a singular A gives roots at infinity
    mine = np.asarray(roots(*map(jnp.asarray, (A, B, C))))
    theirs, _ = reference.roots_of_quadratic(A, B, C)
    finite = np.isfinite(theirs)
    assert np.sum(np.abs(mine) >= 1e6) == np.sum(~finite)
    unmatched = list(mine[np.abs(mine) < 1e6])
    for root in theirs[finite]:  # pair each SciPy root with the nearest remaining root of ours
        nearest = int(np.argmin(np.abs(np.array(unmatched) - root)))
        np.testing.assert_allclose(unmatched.pop(nearest), root, rtol=1e-8)


@pytest.mark.parametrize("n", [1, 3, 8, 20])
@pytest.mark.parametrize("gap", [0.5, 1e-2, 1e-4])
def test_sylvester_solve_matches_the_dense_reference(n, gap):
    # The derivative's equation M X + A X P = R, with M = A P + B from a model whose largest inside root and smallest outside
    # root are 1 - gap and 1 + gap: the closer they are, the more doublings and the worse the equation's conditioning.
    rng = np.random.default_rng(n)
    inside = rng.uniform(-0.9, 0.9, n)
    outside_inverse = rng.choice([-1, 1], n) / rng.uniform(1.1, 3.0, n)
    inside[0], outside_inverse[0] = 1 - gap, -1 / (1 + gap)
    A, B, C, _, P = reference.structured_model(rng, inside, outside_inverse)
    M, R = A @ P + B, rng.normal(size=(n, n))
    expected = np.linalg.solve(reference.sylvester_matrix(M, A, P), R.ravel()).reshape(n, n)
    X = solve_sylvester(*map(jnp.asarray, (M, A, P, R)))
    # Rounding errors are magnified by the condition of the dense system (for the reference) and by the series' amplification
    # (for doubling), so the two may differ by a small multiple of machine precision times the larger of the two.
    magnifier = max(np.linalg.cond(reference.sylvester_matrix(M, A, P)), reference.series_amplification(M, A, P))
    np.testing.assert_allclose(X, expected, rtol=0, atol=10 * np.finfo(float).eps * magnifier * np.max(np.abs(expected)))
    norm = lambda Y: np.linalg.norm(Y, np.inf)
    scale = norm(M) * norm(X) + norm(A) * norm(X) * norm(P) + norm(R)
    assert norm(M @ X + A @ X @ P - R) < 1e-14 * scale  # the residual, relative to the sizes involved: a backward error


# Derivatives -----------------------------------------------------------------------------------------------------------------

def test_derivative_matches_closed_form():
    # d/da of rho / (1 - a rho) is rho^2 / (1 - a rho)^2.
    rho = 0.8
    dP21 = jax.grad(lambda a: solve_forward_looking(*toy(a, rho)).P[1, 0])(0.5)
    np.testing.assert_allclose(dP21, rho**2 / (1 - 0.5 * rho) ** 2, rtol=1e-10)


@pytest.mark.parametrize("seed", range(3))
def test_derivatives_match_finite_differences(seed):
    rng = np.random.default_rng(200 + seed)
    n = 3
    A, B, C, D, _ = map(jnp.asarray, reference.model_with_known_solution(n, rng.uniform(-0.9, 0.9, n), rng.uniform(1.2, 2.5, n), rng))
    W1, W2 = jnp.asarray(rng.normal(size=(n, n))), jnp.asarray(rng.normal(size=(n, D.shape[1])))

    def summary(A, B, C, D):
        solution = solve_forward_looking(A, B, C, D)
        return jnp.sum(W1 * solution.P) + jnp.sum(W2 * solution.Q)

    reverse = jax.grad(summary, argnums=(0, 1, 2, 3))(A, B, C, D)
    forward = jax.jacfwd(summary, argnums=(0, 1, 2, 3))(A, B, C, D)
    h = 1e-6
    for which, (g_reverse, g_forward) in enumerate(zip(reverse, forward)):
        args = [A, B, C, D]
        direction = jnp.asarray(rng.normal(size=args[which].shape))
        plus, minus = list(args), list(args)
        plus[which], minus[which] = args[which] + h * direction, args[which] - h * direction
        numeric = (summary(*plus) - summary(*minus)) / (2 * h)
        np.testing.assert_allclose(jnp.sum(g_reverse * direction), numeric, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(g_forward, g_reverse, rtol=1e-10, atol=1e-12)


def test_many_parameter_values_at_once_with_failures_mixed_in():
    # The price equation's coefficient a runs across the boundary at a = 1, where the model stops having a unique answer.
    a = jnp.linspace(0.1, 1.9, 41)
    rho = 0.6
    solve_one = lambda a: solve_forward_looking(*toy(a, rho))
    solutions = jax.vmap(solve_one)(a)
    np.testing.assert_array_equal(solutions.status, np.where(np.asarray(a) < 1, Status.OK, Status.MANY_STABLE_SOLUTIONS))
    ok = np.asarray(a) < 1
    np.testing.assert_allclose(solutions.P[ok, 1, 0], rho / (1 - np.asarray(a)[ok] * rho), rtol=1e-12)
    gradients = jax.vmap(jax.grad(lambda a: solve_one(a).P[1, 0]))(a)
    assert bool(jnp.all(jnp.isfinite(gradients)))       # failed values give zero derivatives, never NaN
    np.testing.assert_array_equal(gradients[~ok], 0.0)
