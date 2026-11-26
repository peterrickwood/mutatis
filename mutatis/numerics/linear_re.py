"""Solving linear models, with and without expectations of the future.

A linear model is written as

    A E[x(t+1)] + B x(t) + C x(t-1) + D e(t) = 0

where x holds every variable and e every shock. Its solution is a rule for stepping forward,

    x(t) = P x(t-1) + Q e(t).

Without expectation terms (A = 0) the rule follows directly: P = -B^-1 C and Q = -B^-1 D (`solve_backward`).

With expectation terms, P must solve the matrix equation A P^2 + B P + C = 0, and the solution wanted is the one that does not
explode. The equation has 2n roots (n is the number of variables, and roots may be infinite); P carries the n smallest.
Exactly one non-exploding solution exists when n roots are inside the unit circle and n are outside: the Blanchard-Kahn
conditions. `solve_forward_looking` counts the roots (`roots`) to check those conditions, and finds P by cyclic reduction, a
fast iterative method built from matrix products and linear solves (Bini, Latouche and Meini; also used by Dynare).

Derivatives with respect to A, B and C come from differentiating the equation P satisfies (the implicit function theorem), not
from differentiating the iterations. That gives a linear matrix equation for the derivative (`solve_sylvester`), solved by
doubling with n x n products only.

Roots within `UNIT_ROOT_MARGIN` of the unit circle count as not exploding, so models with a random walk can be solved.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.status import Status

UNIT_ROOT_MARGIN = 1e-6
RESIDUAL_TOLERANCE = 1e-8


class LinearSolution(NamedTuple):
    P: Float[Array, "n n"]
    Q: Float[Array, "n k"]
    status: Int[Array, ""]


def solve_backward(B: Float[Array, "n n"], C: Float[Array, "n n"], D: Float[Array, "n k"]) -> LinearSolution:
    """Solve B x(t) + C x(t-1) + D e(t) = 0 for x(t) = P x(t-1) + Q e(t). The answer is unique whether or not it explodes."""
    lu = jax.scipy.linalg.lu_factor(B)
    P, Q = -jax.scipy.linalg.lu_solve(lu, C), -jax.scipy.linalg.lu_solve(lu, D)
    finite = jnp.all(jnp.isfinite(P)) & jnp.all(jnp.isfinite(Q)) & (jnp.min(jnp.abs(jnp.diagonal(lu[0]))) > 0)
    status = jnp.where(finite, Status.OK, Status.SINGULAR)
    return _sanitised(P, Q, status)


def solve_forward_looking(A: Float[Array, "n n"], B: Float[Array, "n n"], C: Float[Array, "n n"], D: Float[Array, "n k"],
                          tolerance: float = 1e-13, max_iterations: int = 100) -> LinearSolution:
    """Solve A E[x(t+1)] + B x(t) + C x(t-1) + D e(t) = 0 for the one solution x(t) = P x(t-1) + Q e(t) that does not explode.

    If there is no such solution, or more than one, `status` says so and P and Q are zero.
    """
    P, status = _solve_quadratic(A, B, C, tolerance, max_iterations)
    status = status.astype(int)
    M = A @ P + B
    Q = -jnp.linalg.solve(jnp.where(status == Status.OK, M, jnp.eye(M.shape[0])), D)
    return _sanitised(P, Q, status)


def _sanitised(P, Q, status):
    ok = status == Status.OK
    return LinearSolution(jnp.where(ok, P, 0.0), jnp.where(ok, Q, 0.0), status)


def cyclic_reduction(A0: Float[Array, "n n"], A1: Float[Array, "n n"], A2: Float[Array, "n n"], tolerance: float = 1e-13,
                     max_iterations: int = 100) -> tuple[Float[Array, "n n"], jax.Array]:
    """The solution X of A0 + A1 X + A2 X^2 = 0 whose eigenvalues are the n smallest roots, and whether the iteration settled.

    The iteration settles when the n smallest roots are strictly smaller in size than the others.
    """
    scale = jnp.maximum(jnp.maximum(_norm(A0), _norm(A1)), _norm(A2))

    def step(carry):
        a0, a1, a2, a_hat, iteration, _ = carry
        s0, s2 = jnp.linalg.solve(a1, a0), jnp.linalg.solve(a1, a2)
        change = a2 @ s0
        a1_next = a1 - a0 @ s2 - change
        a_hat_next = a_hat - change
        a0_next, a2_next = -a0 @ s0, -a2 @ s2
        # The answer is built from a_hat, so stop when a_hat stops changing. Its change shrinks like (r_n / r_n+1)^(2^k), where r_n
        # and r_n+1 are the sizes of the n-th and (n+1)-th roots. Stopping when a0 or a2 alone is small would stop too early when
        # one shrinks while the other grows.
        gap = _norm(change) / jnp.maximum(_norm(a_hat_next), scale * jnp.finfo(A0.dtype).eps)
        return a0_next, a1_next, a2_next, a_hat_next, iteration + 1, gap

    def keep_going(carry):
        *_, iteration, gap = carry
        return (gap > tolerance) & (iteration < max_iterations) & jnp.isfinite(gap)

    initial = (A0, A1, A2, A1, jnp.array(0), jnp.array(jnp.finfo(A0.dtype).max, dtype=A0.dtype))  # finite, so the loop starts
    *_, a_hat, iteration, gap = jax.lax.while_loop(keep_going, step, initial)
    X = -jnp.linalg.solve(a_hat, A0)
    settled = (gap <= tolerance) & jnp.all(jnp.isfinite(X))
    return X, settled


def _norm(X):
    return jnp.max(jnp.sum(jnp.abs(X), axis=0))  # the matrix 1-norm


def roots(A: Float[Array, "n n"], B: Float[Array, "n n"], C: Float[Array, "n n"]) -> jax.Array:
    """All 2n roots of det(r^2 A + r B + C) = 0, with roots at infinity returned as very large numbers. Not differentiated.

    The roots are those of the pencil G - r F, where F = [[0, A], [I, 0]] and G = [[-C, -B], [0, I]]. JAX has no routine for a
    pencil whose F may be singular, so the shift-and-invert trick is used: with a shift s that is not a root, the eigenvalues m
    of (G - s F)^-1 F are m = 1 / (r - s), so r = s + 1 / m, and a root at infinity has m = 0.
    """
    A, B, C = (jax.lax.stop_gradient(M) for M in (A, B, C))
    n = A.shape[0]
    zero, eye = jnp.zeros((n, n)), jnp.eye(n)
    F = jnp.block([[zero, A], [eye, zero]])
    G = jnp.block([[-C, -B], [zero, eye]])
    m = jnp.linalg.eigvals(jnp.linalg.solve(G - _SHIFT * F, F.astype(complex)))
    return _SHIFT + 1.0 / jnp.where(m == 0, jnp.finfo(A.dtype).tiny, m)


_SHIFT = 0.5 + 0.75j  # any number that is not a root; a complex one cannot coincide with a real root


def _classify(A, B, C, tolerance, max_iterations):
    n = A.shape[0]
    P, settled = cyclic_reduction(C, B, A, tolerance, max_iterations)
    residual = _norm(A @ P @ P + B @ P + C) / jnp.maximum(jnp.maximum(_norm(A), _norm(B)), _norm(C))
    inside = jnp.sum(jnp.abs(roots(A, B, C)) < 1.0 + UNIT_ROOT_MARGIN)
    found = settled & (residual < RESIDUAL_TOLERANCE)
    status = jnp.where(inside < n, Status.NO_STABLE_SOLUTION,
                       jnp.where(inside > n, Status.MANY_STABLE_SOLUTIONS, jnp.where(found, Status.OK, Status.NOT_CONVERGED)))
    return jnp.where(status == Status.OK, P, 0.0), status


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4))
def _solve_quadratic(A, B, C, tolerance, max_iterations):
    P, status = _classify(A, B, C, tolerance, max_iterations)
    return P, status.astype(A.dtype)


@_solve_quadratic.defjvp
def _solve_quadratic_jvp(tolerance, max_iterations, primals, tangents):
    # Differentiating A P^2 + B P + C = 0 gives  (A P + B) dP + A dP P = -(dA P^2 + dB P + dC),  a linear equation for dP.
    A, B, C = primals
    dA, dB, dC = tangents
    P, status = _solve_quadratic(A, B, C, tolerance, max_iterations)
    ok = status == Status.OK
    n = P.shape[0]
    # Where the solve failed, P is zero and the tangent must be zero, never NaN: the equation becomes X = 0.
    M, A = jnp.where(ok, A @ P + B, jnp.eye(n)), jnp.where(ok, A, 0.0)
    lu = jax.scipy.linalg.lu_factor(M)
    right = jnp.where(ok, -(dA @ P @ P + dB @ P + dC), 0.0)
    dP = jax.lax.custom_linear_solve(lambda X: M @ X + A @ X @ P, right,
                                     lambda _, R: _sylvester_by_doubling(lu, A, P, R, transposed=False),
                                     lambda _, R: _sylvester_by_doubling(lu, A, P, R, transposed=True))
    return (P, status), (dP, jnp.zeros_like(status))


def solve_sylvester(M: Float[Array, "n n"], A: Float[Array, "n n"], P: Float[Array, "n n"], R: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """The X with M X + A X P = R, when M is invertible and every eigenvalue of M^-1 A times every eigenvalue of P is less than
    one in size. That holds for the equation the derivative of a non-exploding solution satisfies (M = A P + B): the eigenvalues
    of M^-1 A are minus the inverses of the n exploding roots, and those of P are the n others, so the condition is the gap
    between the two groups of roots that the solution needs anyway.

    Rewritten as X = G + F X P with G = M^-1 R and F = -M^-1 A, the answer is the sum of F^j G P^j over j = 0, 1, 2, ...
    Doubling sums it in blocks that double in length, squaring F and P at each step, so k steps cover 2^k terms using only
    products of n x n matrices: the cost grows with n^3 per step, where forming the equation as one n^2 x n^2 system grows
    with n^6. The number of steps grows with the log of 1 / (1 - gap ratio): about 18 when the roots are 1e-4 apart around 1.
    """
    return _sylvester_by_doubling(jax.scipy.linalg.lu_factor(M), A, P, R, transposed=False)


def _sylvester_by_doubling(lu, A, P, R, transposed: bool):
    # transposed: solve M' X + A' X P' = R, the transposed equation that reverse-mode derivatives need.
    trans = 1 if transposed else 0
    F = -jax.scipy.linalg.lu_solve(lu, A.T if transposed else A, trans=trans)
    X, _ = sum_by_doubling(F, jax.scipy.linalg.lu_solve(lu, R, trans=trans), P.T if transposed else P)
    return X


def sum_by_doubling(F: Float[Array, "n n"], G: Float[Array, "n m"], P: Float[Array, "m m"],
                    max_doublings: int = 64) -> tuple[Float[Array, "n m"], Float[Array, ""]]:
    """The sum of F^j G P^j over j = 0, 1, 2, ..., which solves X = G + F X P, and the relative size of the last block added.

    Step k adds the block of terms from 2^k to 2^(k+1) - 1 as F_k X P_k, then squares F_k and P_k, so k steps cover 2^k terms
    with products of whole matrices only. Stops when a block changes the sum by less than rounding (relative, in the 1-norm),
    when it is not finite, or after `max_doublings`. The sum converges when every eigenvalue of F times every eigenvalue of P
    is less than one in size; a last block that is still large, or not finite, says it did not.
    """
    eps = jnp.finfo(G.dtype).eps

    def step(carry):
        X, F, P, _, k = carry
        increment = F @ X @ P
        X = X + increment
        change = _norm(increment) / jnp.maximum(_norm(X), jnp.finfo(X.dtype).tiny)
        return X, F @ F, P @ P, change, k + 1

    def keep_going(carry):
        *_, change, k = carry
        return (change > eps) & (k < max_doublings)

    X, _, _, change, _ = jax.lax.while_loop(keep_going, step, (G, F, P, jnp.array(jnp.inf, G.dtype), jnp.array(0)))
    return X, change
