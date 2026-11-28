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
model, numerically. Otherwise (restrictions that involve several columns, or nonlinear quantities) a search over rotations
gives the bounds (`rotation_bounds`), checked in tests against the exact method where both apply and against formula cases.
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


def linear_in_rotation(function, n: int, key, extra: int = 8):
    """If `function` (a rotation (n, n) to a flat vector) equals A vec(Q) + b at every rotation, (A, b); else None.

    Fitted by least squares at n^2 + 1 + `extra` random rotations and checked, exactly, at as many fresh ones; as with
    `linear_in_one_column`, only rotations are used, so a row of the inverse of chol(Sigma) Q counts as linear. Every restriction
    of a structural VAR met so far has this form (responses, the coefficients of a shock's equation, covariances with a proxy),
    which makes the search in `rotation_bounds` cheap.
    """
    count = n * n + 1 + extra
    fit, check = (UniformOrthogonal(n).sample(k, (count,)) for k in jax.random.split(key))
    design = lambda rotations: np.column_stack([np.asarray(rotations).reshape(count, -1), np.ones(count)])
    at_fit, at_check = np.asarray(jax.vmap(function)(fit)), np.asarray(jax.vmap(function)(check))
    if at_fit.shape[1] == 0:
        return np.zeros((0, n * n)), np.zeros(0)
    coefficients = np.linalg.lstsq(design(fit), at_fit, rcond=None)[0]
    scale = max(1.0, float(np.abs(at_fit).max()))
    if not (np.allclose(design(fit) @ coefficients, at_fit, atol=1e-9 * scale) and np.allclose(design(check) @ coefficients, at_check, atol=1e-9 * scale)):
        return None
    return coefficients[:-1].T, coefficients[-1]


def _cayley(Q, theta):
    """Q times the rotation (I - K/2)^-1 (I + K/2), K skew-symmetric from theta: a rotation for any theta, near Q for small theta,
    and cheaper than the matrix exponential."""
    n = Q.shape[0]
    K = _skew(theta, n)
    eye = jnp.eye(n)
    return Q @ jnp.linalg.solve(eye - 0.5 * K, eye + 0.5 * K)


def _newton_on_rotations(merit, Q, extra, weights, steps):
    """Maximise merit(Q, extra, weight) over rotations Q (and a vector `extra`) for each weight in `weights` in turn, `steps`
    Newton steps each, in local coordinates Q cayley(theta), re-centred after every step so that derivatives are always taken
    at theta = 0. Positive curvature has its sign flipped (and tiny curvature is raised), so every step goes uphill with a
    sensible length; the backtracking line search accepts only finite values, so a merit that is minus infinity outside a
    region never leaves it. Returns the last (Q, extra)."""
    m = Q.shape[0] * (Q.shape[0] - 1) // 2
    local = lambda x, Q, extra, weight: merit(_cayley(Q, x[:m]), extra + x[m:], weight)
    lengths = 2.0 ** -jnp.arange(30)
    zero = jnp.zeros(m + extra.shape[0])

    def step(state, weight):
        Q, extra = state
        value, gradient = jax.value_and_grad(local)(zero, Q, extra, weight)
        hessian = jax.hessian(local)(zero, Q, extra, weight)
        e, V = jnp.linalg.eigh(hessian)
        e = -jnp.maximum(jnp.abs(e), 1e-8 * jnp.maximum(1.0, jnp.max(jnp.abs(e))))   # curvature made negative, keeping its size
        direction = -V @ ((V.T @ gradient) / e)
        values = jax.vmap(lambda t: local(t * direction, Q, extra, weight))(lengths)
        good = jnp.isfinite(values) & (values >= value + 1e-4 * lengths * (gradient @ direction))
        best = jnp.argmax(good)
        x = jnp.where(good[best], lengths[best] * direction, zero)
        return _cayley(Q, x[:m]), extra + x[m:]

    def stage(k, state):
        return jax.lax.fori_loop(0, steps, lambda _, s: step(s, weights[k]), state)

    return jax.lax.fori_loop(0, weights.shape[0], stage, (Q, extra))


def _maximise_on_rotations(objective, constraints, Q, extra, mu0, stages, steps):
    """Maximise objective(Q, extra) subject to constraints(Q, extra) > 0 over rotations Q (and a vector `extra`), from a
    strictly allowed start: a log-barrier method, mu falling tenfold per stage, with `_newton_on_rotations`. Every step stays
    allowed, so the last (Q, extra), which is returned, is allowed."""
    def barrier(Q, extra, mu):
        g = constraints(Q, extra)
        return objective(Q, extra) + mu * jnp.sum(jnp.where(g > 0, jnp.log(jnp.where(g > 0, g, 1.0)), -jnp.inf))

    return _newton_on_rotations(barrier, Q, extra, mu0 * 10.0 ** -jnp.arange(stages), steps)


def _search(restrictions, quantities, Q0, i, sign, size, stages, steps):
    """From an allowed rotation Q0, the largest value of sign x quantity i over allowed rotations reached from it."""
    Q, _ = _maximise_on_rotations(lambda Q, e: sign * quantities(Q)[i], lambda Q, e: restrictions(Q), Q0, jnp.zeros(0), 0.1 * size, stages, steps)
    return Q, quantities(Q)[i], jnp.all(restrictions(Q) >= 0)


SOFT_MARGIN = 1e-6      # how far inside its restrictions the softened search aims, in units of each restriction's typical size
SOFT_WEIGHTS = 10.0 ** np.arange(9)


def _soften(restrictions, quantities, Q0, i, sign, size, scale, steps):
    """From any rotation Q0, allowed or not, the largest value of sign x quantity i reached by treating the restrictions as a
    penalty that hardens stage by stage, then making the result exact (decision 53).

    The penalty is half the sum of squares of each restriction's shortfall below a small margin, measured in units of `scale`
    (each restriction's typical size); at weight w the merit is sign x quantity / size / w minus the penalty, so as w rises from
    1 to 10^8 the search moves from mostly the objective to mostly the restrictions, without huge numbers. Forbidden rotations
    are allowed along the way, so the search can reach an extreme from outside the allowed set. The barrier search cannot:
    staying inside, it is pushed away from places where many restrictions meet at once, which is where extremes under
    restrictions on several columns can lie. The end point is inside by about the margin; the barrier search then finishes
    from there with a small barrier, which keeps the result allowed."""
    def merit(Q, extra, w):
        shortfall = jnp.minimum(0.0, restrictions(Q) / scale - SOFT_MARGIN)
        return sign * quantities(Q)[i] / size / w - 0.5 * jnp.sum(shortfall**2)

    Q, _ = _newton_on_rotations(merit, Q0, jnp.zeros(0), jnp.asarray(SOFT_WEIGHTS), steps)
    Q, _ = _maximise_on_rotations(lambda Q, e: sign * quantities(Q)[i], lambda Q, e: restrictions(Q), Q, jnp.zeros(0), 1e-7 * size, 6, steps)
    return Q, quantities(Q)[i], jnp.all(restrictions(Q) >= 0)


def _phase_one(restrictions, Q0, t0, stages, steps):
    """From a rotation that breaks some restrictions, the rotation reached with the largest smallest restriction value, and that value."""
    Q, t = _maximise_on_rotations(lambda Q, e: e[0], lambda Q, e: restrictions(Q) - e[0], Q0, t0[None], 1e-2 * jnp.abs(t0) + 1e-8, stages, steps)
    return Q, t[0]


def _linear(matrix, offset):
    return lambda Q: matrix @ Q.reshape(-1) + offset


# Compiled once for restrictions and quantities given as matrices (the usual case), so that nothing recompiles draw by draw.
_linear_search = jax.jit(jax.vmap(lambda S, s0, C, c0, Q0, i, sign, size, stages, steps: _search(_linear(S, s0), _linear(C, c0), Q0, i, sign, size, stages, steps),
                                  in_axes=(None, None, None, None, 0, 0, 0, 0, None, None)), static_argnums=(8, 9))
_linear_phase_one = jax.jit(jax.vmap(lambda S, s0, Q0, t0, stages, steps: _phase_one(_linear(S, s0), Q0, t0, stages, steps), in_axes=(None, None, 0, 0, None, None)),
                            static_argnums=(4, 5))
_linear_soften = jax.jit(jax.vmap(lambda S, s0, C, c0, scale, Q0, i, sign, size, steps: _soften(_linear(S, s0), _linear(C, c0), Q0, i, sign, size, scale, steps),
                                  in_axes=(None, None, None, None, None, 0, 0, 0, 0, None)), static_argnums=(9,))


def rotation_bounds(restrictions, quantities, n: int, key, pool: int = 20_000, starts: int = 4, soft_starts: int = 16, stages: int = 12, steps: int = 6):
    """Smallest and largest value of every quantity over rotations that satisfy the restrictions (every row >= 0), for
    restrictions and quantities of any form, including ones that involve several columns of the rotation.

    Two searches, and for each bound the more extreme result that is allowed. `pool` random rotations are drawn first.
    - The barrier search: those that satisfy the restrictions give starting points (for each bound, the `starts` best); from
      each, the barrier method of `_maximise_on_rotations` moves the rotation, staying a rotation and allowed at every step,
      until the bound is reached to about 1e-10 of the quantity's scale.
    - The softened search (decision 53): from `soft_starts` random rotations per bound, allowed or not, `_soften` treats the
      restrictions as a penalty that hardens stage by stage, then makes the result exactly allowed. It reaches extremes that
      lie where many restrictions meet at once, which the barrier search is pushed away from.
    A bound can be missed only where every start of both searches leads to a lower local best (checked in tests against exact
    methods). If no pool rotation is allowed, a first phase maximises the smallest restriction value from the most nearly
    allowed ones; if that stays negative, the allowed set is taken to be empty (it may then be nonempty but too small for the
    search to find). Restrictions and quantities linear in the rotation (`linear_in_rotation`) are evaluated through their
    fitted matrices, by programs compiled once; others through the functions themselves, compiled afresh for each call.

    Returns (lower, upper, argmin, argmax, empty); argmin and argmax are the rotations attaining each bound (quantities, n, n).
    """
    keys = jax.random.split(key, 3)
    restrict_form, quantity_form = linear_in_rotation(restrictions, n, keys[0]), linear_in_rotation(quantities, n, keys[1])
    linear = restrict_form is not None and quantity_form is not None
    if linear:
        S, s0, C, c0 = (jnp.asarray(a) for a in (*restrict_form, *quantity_form))
        restrictions, quantities = _linear(S, s0), _linear(C, c0)
        search = lambda *args: _linear_search(S, s0, C, c0, *args, stages, steps)
        phase_one = lambda *args: _linear_phase_one(S, s0, *args, stages, 4 * steps)   # rare and small: more steps
        soften = lambda scale, *args: _linear_soften(S, s0, C, c0, scale, *args, steps)
    else:
        search = jax.jit(jax.vmap(lambda Q0, i, sign, size: _search(restrictions, quantities, Q0, i, sign, size, stages, steps)))
        phase_one = jax.jit(jax.vmap(lambda Q0, t0: _phase_one(restrictions, Q0, t0, stages, 4 * steps)))
        soften = jax.jit(jax.vmap(lambda scale, Q0, i, sign, size: _soften(restrictions, quantities, Q0, i, sign, size, scale, steps), in_axes=(None, 0, 0, 0, 0)))

    candidates = UniformOrthogonal(n).sample(keys[2], (pool,))
    pool_rotations = candidates
    g = np.asarray(jax.vmap(restrictions)(candidates))
    allowed = np.all(g > 0, axis=1) if g.shape[1] else np.ones(pool, dtype=bool)
    if not allowed.any():
        order = np.argsort(-g.min(axis=1))[:8]
        least = g[order].min(axis=1)
        found, margin = phase_one(candidates[order], jnp.asarray(least - 0.1 * np.abs(least) - 1e-6))
        found = np.asarray(found)[np.asarray(margin) > 0]
        if not len(found):
            k = int(np.asarray(quantities(candidates[0])).size)
            return np.full(k, np.nan), np.full(k, np.nan), np.full((k, n, n), np.nan), np.full((k, n, n), np.nan), True
        candidates, allowed = jnp.asarray(found), np.ones(len(found), dtype=bool)
    feasible = np.asarray(candidates)[np.flatnonzero(allowed)]
    values = np.asarray(jax.vmap(quantities)(jnp.asarray(feasible)))                                # (feasible, quantities)
    k = values.shape[1]
    # `starts` problems per bound always (repeating the best start when there are fewer allowed rotations): fixed shapes.
    problems = [(i, sign, int(np.argsort(-sign * values[:, i])[min(j, len(feasible) - 1)])) for i in range(k) for sign in (1.0, -1.0) for j in range(starts)]
    index, signs, chosen = (np.asarray(a) for a in zip(*problems))
    size = np.maximum(np.abs(values[chosen, index]), 1e-6)
    found, reached, ok = (np.asarray(a) for a in search(jnp.asarray(feasible[chosen]), jnp.asarray(index), jnp.asarray(signs), jnp.asarray(size)))
    lower, upper = values.min(axis=0), values.max(axis=0)
    argmin, argmax = feasible[values.argmin(axis=0)], feasible[values.argmax(axis=0)]
    results = [zip(index, signs, found, reached, ok)]
    if soft_starts:
        # Units: each restriction's and each quantity's spread over the pool (1 where a function does not vary).
        spread = lambda x: np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        scale = spread(g) if g.shape[1] else np.ones(0)
        quantity_scale = spread(np.asarray(jax.vmap(quantities)(pool_rotations[: min(pool, 2000)])))
        problems = [(i, sign) for i in range(k) for sign in (1.0, -1.0) for _ in range(soft_starts)]
        index, signs = (np.asarray(a) for a in zip(*problems))
        chosen = np.asarray(jax.random.choice(jax.random.fold_in(keys[2], 1), pool, (len(problems),), replace=len(problems) > pool))
        found, reached, ok = (np.asarray(a) for a in soften(jnp.asarray(scale), pool_rotations[chosen], jnp.asarray(index), jnp.asarray(signs), jnp.asarray(quantity_scale[index])))
        results.append(zip(index, signs, found, reached, ok))
    for i, sign, Q, value, good in itertools.chain(*results):
        if not good:
            continue
        if sign > 0 and value > upper[i]:
            upper[i], argmax[i] = value, Q
        elif sign < 0 and value < lower[i]:
            lower[i], argmin[i] = value, Q
    return lower, upper, argmin, argmax, False
