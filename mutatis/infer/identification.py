"""Unknowns the data cannot inform: the rotation of a structural VAR, for example.

When the likelihood does not depend on some unknowns, their posterior is their prior, restricted to what the restrictions allow.
Two things are then possible:

- with a prior (the even distribution over rotations, say): draw from it, keeping only draws that satisfy the restrictions
  (`draw_restricted`), giving the conventional single-prior result;
- with no prior: for each posterior draw of everything else, the smallest and largest value of each reported quantity over all
  values the restrictions allow (`bounds`), giving ranges (Giacomini and Kitagawa, 2021).

Bounds use an exact method when every restriction and quantity depends on a single column q of a rotation, linearly: then the
allowed q form a cone {q : S q >= 0} on the unit sphere, and the largest value of c'q there is the length of c's projection onto
the cone, when that projection is not zero (a small non-negative least-squares problem). Whether this applies is checked on the
model, numerically. Restrictions that involve several columns, or nonlinear quantities, need a general optimiser (not built yet).
"""

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import nnls

from mutatis import _jax_config  # noqa: F401
from mutatis.dist import UniformOrthogonal


class IdentificationError(ValueError):
    pass


@dataclass(frozen=True)
class OneColumn:
    """Restrictions and quantities as linear functions of column `column` of the rotation: S q >= 0, quantities = C q."""

    column: int
    S: np.ndarray     # (restriction rows, n)
    C: np.ndarray     # (quantity rows, n)


def linear_in_one_column(restrictions, quantities, n: int, key, extra: int = 6) -> OneColumn | None:
    """If every restriction row and quantity row is c'q for one column q of the rotation, those c vectors; else None.

    `restrictions` and `quantities` map a rotation (n, n) to a flat vector; a restriction holds where its value is >= 0. Both
    are evaluated at n + `extra` random rotations, and fitted, for each column in turn, as linear functions of that column alone
    by least squares. The fit must be exact, and must hold again at fresh rotations. Evaluating only at rotations matters: a
    function such as a row of the inverse of chol(Sigma) Q is linear in one column of Q only because Q is orthogonal, so
    derivatives with respect to unrestricted changes in Q would wrongly show it depending on every column.
    """
    fit_key, check_key = jax.random.split(key)
    fit, check = (UniformOrthogonal(n).sample(k, (n + extra,)) for k in (fit_key, check_key))
    values = {name: (np.asarray(jax.vmap(fn)(fit)), np.asarray(jax.vmap(fn)(check))) for name, fn in (("S", restrictions), ("C", quantities))}
    found = []
    for column in range(n):
        q_fit, q_check = np.asarray(fit[:, :, column]), np.asarray(check[:, :, column])
        coefficients = {}
        for name, (at_fit, at_check) in values.items():
            if at_fit.shape[1] == 0:
                coefficients[name] = np.zeros((0, n))
                continue
            solution = np.linalg.lstsq(q_fit, at_fit, rcond=None)[0].T
            scale = max(1.0, float(np.abs(at_fit).max()))
            if not (np.allclose(q_fit @ solution.T, at_fit, atol=1e-9 * scale) and np.allclose(q_check @ solution.T, at_check, atol=1e-9 * scale)):
                break
            coefficients[name] = solution
        else:
            found.append(OneColumn(column, coefficients["S"], coefficients["C"]))
    # Quantities that do not depend on the rotation at all fit every column with zero coefficients; prefer a column that matters.
    useful = [f for f in found if np.abs(f.S).max(initial=0) > 0 or np.abs(f.C).max(initial=0) > 0]
    return (useful or found)[0] if len(useful) == 1 or (not useful and found) else None


def fit_one_column(restrictions, quantities, n: int, column: int, key, extra: int = 2) -> OneColumn:
    """The c vectors for a column already known to work (see `linear_in_one_column`), from n + `extra` random rotations."""
    rotations = UniformOrthogonal(n).sample(key, (n + extra,))
    q = np.asarray(rotations[:, :, column])
    S = np.linalg.lstsq(q, np.asarray(jax.vmap(restrictions)(rotations)), rcond=None)[0].T
    C = np.linalg.lstsq(q, np.asarray(jax.vmap(quantities)(rotations)), rcond=None)[0].T
    return OneColumn(column, S, C)


def _projection(c: np.ndarray, S: np.ndarray) -> np.ndarray:
    """The projection of c onto the cone {q : S q >= 0}, using the polar cone {-S' l : l >= 0} (Moreau's decomposition)."""
    if S.shape[0] == 0:
        return c
    weights, _ = nnls(-S.T, c)
    return c + S.T @ weights


def _rays(S: np.ndarray, tolerance: float = 1e-10) -> list[np.ndarray]:
    """Unit vectors spanning the edges of the cone {q : S q >= 0} in n dimensions, and its directions along which S q = 0."""
    n = S.shape[1]
    out = []
    null = np.linalg.svd(S)[2][np.linalg.matrix_rank(S, tol=tolerance):] if S.shape[0] else np.eye(n)
    for direction in null:
        out += [direction, -direction]
    for rows in itertools.combinations(range(S.shape[0]), n - 1):
        sub = S[list(rows)]
        if np.linalg.matrix_rank(sub, tol=tolerance) != n - 1:
            continue
        ray = np.linalg.svd(sub)[2][-1]
        for candidate in (ray, -ray):
            if np.all(S @ candidate >= -tolerance):
                out.append(candidate / np.linalg.norm(candidate))
    return out


def one_column_bounds(form: OneColumn, tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray, bool]:
    """Smallest and largest value of every quantity over unit vectors in the cone. Third value: whether the cone is empty."""
    lower, upper, _, _, empty = one_column_extremes(form, tolerance)
    return lower, upper, empty


def one_column_extremes(form: OneColumn, tolerance: float = 1e-10):
    """As `one_column_bounds`, plus the unit vectors that attain each bound: (lower, upper, argmin, argmax, empty)."""
    rays = _rays(form.S, tolerance)
    k, n = form.C.shape
    if not rays:
        return np.full(k, np.nan), np.full(k, np.nan), np.full((k, n), np.nan), np.full((k, n), np.nan), True
    upper, lower, argmax, argmin = np.empty(k), np.empty(k), np.empty((k, n)), np.empty((k, n))
    for i, c in enumerate(form.C):
        upper[i], argmax[i] = _largest(c, form.S, rays, tolerance)
        value, argmin[i] = _largest(-c, form.S, rays, tolerance)
        lower[i] = -value
    return lower, upper, argmin, argmax, False


def _largest(c, S, rays, tolerance) -> tuple[float, np.ndarray]:
    projected = _projection(c, S)
    length = float(np.linalg.norm(projected))
    if length > tolerance:
        return length, projected / length
    values = [float(c @ r) for r in rays]  # c points away from the cone: the best is along one of its edges
    best = int(np.argmax(values))
    return values[best], rays[best]


def draw_restricted(sample_prior, satisfied, key, max_tries: int = 10_000, batch: int = 500):
    """A draw from the prior that satisfies the restrictions, by rejection; None if none is found within `max_tries` draws."""
    tries = 0
    while tries < max_tries:
        key, sub = jax.random.split(key)
        candidates = sample_prior(sub, batch)
        ok = np.asarray(jax.vmap(satisfied)(candidates))
        if ok.any():
            return candidates[int(np.argmax(ok))]
        tries += batch
    return None


def _skew(parameters, n):
    rows, cols = np.triu_indices(n, k=1)
    upper = jnp.zeros((n, n)).at[rows, cols].set(parameters)
    return upper - upper.T


def optimised_bounds(restrictions, quantities, n: int, key, starts: int = 20, candidates: int = 4000):
    """Smallest and largest value of every quantity over rotations that satisfy the restrictions, for restrictions and quantities
    of any form: a search, so the result is an inner bound that may miss the true end if the search stops at a local best.

    Starting points are allowed rotations found among `candidates` random ones; from the best `starts` for each bound, SLSQP
    moves the rotation as Q exp(K) with K skew-symmetric, so every step stays a rotation. Returns (lower, upper, empty).
    """
    from scipy.optimize import minimize

    pool = UniformOrthogonal(n).sample(key, (candidates,))
    allowed = np.asarray(jax.vmap(lambda Q: jnp.all(restrictions(Q) >= 0))(pool))
    if not allowed.any():
        return None, None, True
    feasible = pool[np.asarray(allowed)]
    values = np.asarray(jax.vmap(quantities)(feasible))
    lower, upper = values.min(axis=0), values.max(axis=0)
    m = n * (n - 1) // 2
    rotate = jax.jit(lambda Q, k: Q @ jax.scipy.linalg.expm(_skew(k, n)))
    for i in range(values.shape[1]):
        for sign, best in ((1.0, lower), (-1.0, upper)):
            order = np.argsort(sign * values[:, i])[:starts]
            objective = jax.jit(jax.value_and_grad(lambda k, Q: sign * quantities(rotate(Q, k))[i]))
            constraint = jax.jit(lambda k, Q: restrictions(rotate(Q, k)))
            constraint_jacobian = jax.jit(jax.jacfwd(lambda k, Q: restrictions(rotate(Q, k))))
            for start in order:
                Q0 = feasible[start]
                result = minimize(lambda k: tuple(np.asarray(v, dtype=float) for v in objective(jnp.asarray(k), Q0)), np.zeros(m), jac=True,
                                  method="SLSQP", constraints=[{"type": "ineq", "fun": lambda k: np.asarray(constraint(jnp.asarray(k), Q0)),
                                                                "jac": lambda k: np.asarray(constraint_jacobian(jnp.asarray(k), Q0))}],
                                  options={"maxiter": 200, "ftol": 1e-12})
                Q = rotate(Q0, jnp.asarray(result.x))
                if bool(jnp.all(restrictions(Q) >= -1e-9)):
                    value = float(quantities(Q)[i])
                    best[i] = min(best[i], value) if sign > 0 else max(best[i], value)
    return lower, upper, False
