"""The forward-looking solve at realistic sizes: against an independent method (SciPy's ordered QZ), derivatives against finite
differences, and risk experiment 2 from PLAN.md (many parameter values at once, with derivatives).

Medium tier: run with `pytest -m medium -s tests/numerics/test_linear_re_at_scale.py` to see the timings.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.numerics.linear_re import RESIDUAL_TOLERANCE, UNIT_ROOT_MARGIN, solve_forward_looking
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


# Against an independent method, at realistic sizes ----------------------------------------------------------------------------

KINDS = ["complex_roots", "static_and_unlagged", "unit_root", "gap1e-2", "gap1e-3", "gap1e-4", "one_extra_inside", "one_inside_missing"]


def placed_roots(n, kind, rng):
    """Where the roots go. `inside`: eigenvalues of the solution P. `outside_inverse`: minus the inverse of each other root."""
    inside = rng.uniform(-0.9, 0.9, n)
    outside_inverse = rng.choice([-1, 1], n) / rng.uniform(1.1, 3.0, n)
    if kind == "complex_roots":  # every other pair of inside roots becomes a complex pair of the same size
        inside = inside.astype(complex)
        for i in range(0, n - 1, 4):
            angle = rng.uniform(0.2, 2.8)
            inside[i], inside[i + 1] = abs(inside[i]) * np.exp(1j * angle), abs(inside[i]) * np.exp(-1j * angle)
    elif kind == "static_and_unlagged":  # a third of the equations static (roots at infinity), a third of the variables unlagged (zero roots)
        inside[: n // 3] = 0.0
        outside_inverse[n - n // 3:] = 0.0
    elif kind == "unit_root":
        inside[0] = 1.0
    elif kind.startswith("gap"):  # the largest inside root and the smallest outside root are 1 - gap and 1 + gap
        gap = float(kind[3:])
        inside[0], outside_inverse[0] = 1 - gap, -1 / (1 + gap)
    elif kind == "one_extra_inside":
        outside_inverse[0] = -2.0  # a root at 0.5
    elif kind == "one_inside_missing":
        inside[0] = 1.3
    return inside, outside_inverse


@pytest.mark.parametrize("n", [10, 40, 80])
@pytest.mark.parametrize("kind", KINDS)
def test_agrees_with_ordered_qz(n, kind):
    rng = np.random.default_rng(1000 * n + KINDS.index(kind))
    A, B, C, D, _ = reference.structured_model(rng, *placed_roots(n, kind, rng), k=3)
    P_ref, _, count = reference.qz_solution(A, B, C, D, UNIT_ROOT_MARGIN)
    expected = Status.OK if count == n else Status.MANY_STABLE_SOLUTIONS if count > n else Status.NO_STABLE_SOLUTION
    P, Q, status = jax.jit(solve_forward_looking)(*map(jnp.asarray, (A, B, C, D)))
    assert int(status) == expected
    if expected != Status.OK:
        return
    norm = lambda X: np.max(np.sum(np.abs(X), axis=0))
    residual = norm(A @ P @ P + B @ P + C) / max(norm(A), norm(B), norm(C))
    difference = np.max(np.abs(P - P_ref)) / np.max(np.abs(P_ref))
    print(f"\nn={n} {kind}: relative difference from QZ {difference:.1e}, relative residual {residual:.1e}")
    assert residual < RESIDUAL_TOLERANCE / 100
    if kind != "gap1e-4":
        assert difference < 1e-8
    # At a gap of 1e-4 the answer itself is sensitive to rounding: in decision 35's check against the exact answer (Newton's
    # method with residuals in 40-place arithmetic), both this solve and QZ were 1e-9 to 3e-8 away from it at n = 10 to 80.
    # Only the residual, which an algorithm can control, is required there.


# Derivatives at realistic sizes ------------------------------------------------------------------------------------------------

def root_preserving_path(n, kind, rng):
    """A smooth path theta -> (A, B, C, D) along which every root stays exactly where it is.

    Finite differences in a random direction of A, B and C can move a root across the unit circle, where the solution stops
    existing (this happened for roots within 1e-2 of the circle with relative steps of 1e-6). Here the matrices are built as in
    `reference.structured_model` from eigenvector matrices and M, and only those move along the path, so the roots do not.
    """
    inside, outside_inverse = placed_roots(n, kind, rng)
    V, V2, M, D = rng.normal(size=(n, n)), rng.normal(size=(n, n)), rng.normal(size=(n, n)) + np.sqrt(n) * np.eye(n), rng.normal(size=(n, 3))
    dV, dV2, dM, dD = (rng.normal(size=X.shape) * np.linalg.norm(X) / np.sqrt(X.size) for X in (V, V2, M, D))
    J, J2 = jnp.diag(inside), jnp.diag(outside_inverse)

    def matrices(theta):
        Vt, V2t, Mt = V + theta * dV, V2 + theta * dV2, M + theta * dM
        P = Vt @ J @ jnp.linalg.inv(Vt)
        A = Mt @ V2t @ J2 @ jnp.linalg.inv(V2t)
        return A, Mt - A @ P, -Mt @ P, D + theta * dD

    return matrices


@pytest.mark.parametrize("n", [10, 40])
@pytest.mark.parametrize("kind", ["static_and_unlagged", "unit_root", "gap1e-2"])
def test_derivatives_match_finite_differences_at_scale(n, kind):
    rng = np.random.default_rng(7 * n + len(kind))
    matrices = root_preserving_path(n, kind, rng)
    W1, W2 = jnp.asarray(rng.normal(size=(n, n))), jnp.asarray(rng.normal(size=(n, 3)))

    def summary(theta):
        solution = solve_forward_looking(*matrices(theta))
        return jnp.sum(W1 * solution.P) + jnp.sum(W2 * solution.Q), solution.status

    value = jax.jit(lambda theta: summary(theta)[0])
    h = 1e-5
    assert all(int(jax.jit(lambda t: summary(t)[1])(t)) == Status.OK for t in (0.0, h, -h, h / 2, -h / 2))
    reverse = float(jax.jit(jax.grad(value))(0.0))
    forward = float(jax.jit(lambda t: jax.jvp(value, (t,), (1.0,))[1])(0.0))
    central = lambda step: float(value(step) - value(-step)) / (2 * step)
    fine, coarse = central(h / 2), central(h)
    numeric = (4 * fine - coarse) / 3  # Richardson extrapolation removes the h^2 term
    scale = max(abs(numeric), abs(reverse))
    print(f"\nn={n} {kind}: reverse {reverse:.10g}, forward {forward:.10g}, finite differences {numeric:.10g} (h and h/2 differ by {abs(fine - coarse) / scale:.1e})")
    # The finite difference is uncertain by about its change between h and h/2 (truncation) plus the solve's rounding error
    # divided by h (about 1e-12 / 1e-5).
    assert abs(reverse - numeric) / scale < 10 * abs(fine - coarse) / scale + 1e-7
    assert abs(forward - reverse) / scale < 1e-6  # two different linear solves: the transposed one serves reverse mode
