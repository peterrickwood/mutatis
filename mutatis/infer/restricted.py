"""Unknowns the likelihood ignores, for one posterior draw of everything else: a restricted prior draw, or ranges.

Used by the exact regression formula. The restrictions are the specification's; the reported quantities are the run's.
Functions are compiled once, with the draw passed as an argument, so nothing is recompiled draw by draw.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.dist import UniformOrthogonal
from mutatis.dist.domains import Orthogonal
from mutatis.infer.identification import fit_one_column, linear_in_one_column, linear_in_rotation, one_column_extremes, rotation_bounds
from mutatis.quantities.expressions import evaluate_quantity
from mutatis.spec.expr import parse_expression


class RestrictedOthers:
    def __init__(self, problem, others, report):
        self.problem, self.others, self.report = problem, tuple(others), tuple(report)
        self.restrictions = problem.restrictions
        self.report_exprs = [parse_expression(q) for q in self.report]
        self.restriction_values = jax.jit(problem.restriction_values)
        self.report_values = jax.jit(self._report_values)
        self.satisfied = jax.jit(jax.vmap(lambda base, extra: jnp.all(problem.restriction_values({**base, **extra}) >= 0), in_axes=(None, 0)))
        self.no_prior = [u for u in self.others if u.no_prior]
        if self.no_prior and (len(self.no_prior) > 1 or not isinstance(self.no_prior[0].domain, Orthogonal)):
            raise ValueError("ranges are supported for one rotation with no prior; other unknowns with no prior are not supported yet")
        self.method = None          # "one column" (exact) or "search" (rotation_bounds), decided at the first draw
        self.one_column = None
        self.report_sizes = None

    def _context(self, values):
        return self.problem.context(values)

    def _report_values(self, values):
        ctx = self._context(values)
        parts = [jnp.ravel(evaluate_quantity(q, ctx)) for q in self.report_exprs]
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def complete(self, base, key, max_tries):
        """(values with the other unknowns filled in, (lower, upper) bounds of the report or None), or None if nothing is allowed."""
        with_prior = [u for u in self.others if not u.no_prior]
        values = dict(base)
        if with_prior:
            def sample(k, size):
                keys = jax.random.split(k, len(with_prior))
                out = {}
                for u, kk in zip(with_prior, keys):
                    dist = u.prior.build({})
                    out[u.name] = dist.sample(kk, (size,)) if dist.event_shape else dist.sample(kk, (size, *u.shape))
                return out
            key, fit_key = jax.random.split(key)
            draw = self._draw(sample, values, key, max_tries, self._screen(with_prior, values, fit_key))
            if draw is None:
                return None
            values.update(draw)
        if not self.no_prior:
            return values, None
        return self._bounds(values, key)

    def _screen(self, with_prior, values, key):
        """When the only unknown drawn is a rotation and every restriction is linear in it (`linear_in_rotation`, fitted for this
        draw), a cheap test of many candidates at once: the fitted matrix times each rotation. Otherwise None."""
        if len(with_prior) != 1 or not isinstance(with_prior[0].domain, Orthogonal) or not self.restrictions:
            return None
        u = with_prior[0]
        form = linear_in_rotation(lambda Q: self.restriction_values({**values, u.name: Q}), u.shape[0], key)
        if form is None:
            return None
        S, s0 = form
        return lambda candidates: np.all(np.asarray(candidates[u.name]).reshape(len(candidates[u.name]), -1) @ S.T + s0 >= 0, axis=1)

    def _draw(self, sample, values, key, max_tries, screen=None, batch=500):
        """The first prior draw that satisfies the restrictions, in batches of `batch`, or None after `max_tries`. With a cheap
        `screen`, batches are larger, and a candidate it passes is still checked by the restrictions themselves."""
        batch = 20 * batch if screen is not None else batch
        tries = 0
        while tries < max_tries:
            key, sub = jax.random.split(key)
            candidates = sample(sub, min(batch, max_tries - tries))
            ok = screen(candidates) if screen is not None else np.asarray(self.satisfied(values, candidates))
            for i in np.flatnonzero(ok)[:3] if screen is not None else np.flatnonzero(ok)[:1]:
                candidate = {k: v[i] for k, v in candidates.items()}
                if screen is None or bool(self.satisfied(values, {k: v[None] for k, v in candidate.items()})[0]):
                    return candidate
            tries += len(next(iter(candidates.values())))
        return None

    def _bounds(self, values, key):
        """Ranges by the exact one-column method when every restriction and reported quantity is linear in one column of the
        rotation (checked at the first draw); otherwise by a search over rotations (`rotation_bounds`)."""
        rotation = self.no_prior[0]
        n = rotation.shape[0]
        restrict = lambda Q: self.restriction_values({**values, rotation.name: Q})
        report = lambda Q: self.report_values({**values, rotation.name: Q})
        if self.method is None:
            form = linear_in_one_column(restrict, report, n, key)
            self.method = "one column" if form is not None else "search"
            self.one_column = None if form is None else form.column
            self.report_sizes = [int(np.size(np.asarray(evaluate_quantity(q, self._context({**values, rotation.name: jnp.eye(n)}))))) for q in self.report_exprs]
        if self.method == "search":
            lower, upper, _, argmax, empty = rotation_bounds(restrict, report, n, key)
            if empty:
                return None
            return {**values, rotation.name: jnp.asarray(argmax[0])}, (lower, upper)
        column = self.one_column
        lower, upper, argmin, argmax, empty = one_column_extremes(fit_one_column(restrict, report, n, column, key))
        if empty:
            return None
        # Store some allowed rotation with the draw: one whose column attains a bound, completed to an orthonormal basis.
        q = argmax[0] if len(argmax) else argmin[0]
        basis = np.linalg.qr(np.column_stack([q, np.eye(n)]))[0][:, :n]
        basis[:, 0] = q
        allowed = np.roll(basis, column, axis=1)
        values = {**values, rotation.name: jnp.asarray(allowed)}
        return values, (lower, upper)

    def split(self, lower, upper):
        """Bounds per reported quantity: {quantity: (lower (draws, size), upper (draws, size))}."""
        out, start = {}, 0
        for text, size in zip(self.report, self.report_sizes):
            out[text] = (lower[:, start:start + size], upper[:, start:start + size])
            start += size
        return out


def ranges_posterior(problem, key, sampler, n_draws: int = 1000, report: tuple[str, ...] = (), max_tries: int = 10_000, **options):
    """Ranges with any sampler: for a problem whose only unknown without a prior is a rotation the likelihood ignores, and
    whose other unknowns have proper priors (a model the exact regression formula does not cover).

    The other unknowns are sampled by `sampler` with the rotation held fixed and the restrictions left out, since they involve
    the rotation; the likelihood is first checked, at prior draws, not to change with the rotation. Then for each of `n_draws`
    draws from that posterior, every `report` quantity's smallest and largest value over the rotations the restrictions allow
    (`RestrictedOthers`), dropping draws for which none is allowed (the share kept is `plausibility`), as the exact regression
    formula does for regressions.
    """
    from mutatis.infer.posterior import Posterior

    no_prior = [u for u in problem.layout.unknowns if u.no_prior]
    if len(no_prior) != 1 or not isinstance(no_prior[0].domain, Orthogonal):
        raise ValueError("ranges with a sampler need exactly one unknown with no prior, a rotation")
    rotation = no_prior[0]
    if not all(u.prior.proper for u in problem.layout.unknowns if not u.no_prior and u.prior is not None):
        raise ValueError("ranges with a sampler need proper priors on everything but the rotation")
    n = rotation.shape[0]
    key_check, key_sample, key_pick, key_ranges = jax.random.split(key, 4)
    reduced = problem.with_fixed({rotation.name: jnp.eye(n)}, apply_restrictions=False)
    k1, k2, k3 = jax.random.split(key_check, 3)
    alternatives = [problem.with_fixed({rotation.name: Q}, apply_restrictions=False) for Q in UniformOrthogonal(n).sample(k1, (2,))]
    for z in jax.vmap(reduced.draw_prior)(jax.random.split(k2, 4)):
        a, b = (float(p.log_likelihood(z)) for p in alternatives)
        if not np.isclose(a, b, rtol=1e-9, atol=1e-9):
            raise ValueError(f"the likelihood changes with {rotation.name} ({a} against {b}), so it cannot be left without a prior")

    sampled = sampler(reduced, key_sample, **options)
    others = RestrictedOthers(problem, (rotation,), report)
    draws, lower, upper = [], [], []
    for z, draw_key in zip(sampled.resample(key_pick, n_draws), jax.random.split(key_ranges, n_draws)):
        values, _ = reduced.layout.constrain(jnp.asarray(z))
        result = others.complete({k: v for k, v in values.items()}, draw_key, max_tries)
        if result is None:
            continue
        values, bounds = result
        draws.append(np.asarray(problem.layout.unconstrain(values)))
        lower.append(bounds[0])
        upper.append(bounds[1])
    if not draws:
        raise ValueError("no draw has a rotation that satisfies the restrictions")
    diagnostics = {"sampler": sampled.method, "sampler_diagnostics": sampled.diagnostics, "plausibility": len(draws) / n_draws,
                   "bounds": others.split(np.stack(lower), np.stack(upper))}
    return Posterior(np.stack(draws), np.full(len(draws), 1.0 / len(draws)), None, f"{sampled.method} + ranges", diagnostics)
