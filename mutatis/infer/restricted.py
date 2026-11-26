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
from mutatis.infer.identification import fit_one_column, linear_in_one_column, one_column_extremes
from mutatis.quantities.expressions import Context, evaluate_quantity
from mutatis.spec.expr import parse_expression, parse_requirement

_SIGN = {">=": 1.0, ">": 1.0, "<=": -1.0, "<": -1.0}


class RestrictedOthers:
    def __init__(self, problem, others, report):
        self.problem, self.others, self.report = problem, tuple(others), tuple(report)
        data = problem.data
        first = data.start.first if problem.start == "data" else 0
        self.series = {o.series: jnp.asarray(data.observed[first:, i]) for i, o in enumerate(problem.system.observations)}
        self.series.update({name: jnp.asarray(values[first:]) for name, values in data.auxiliary.items()})
        self.restrictions = []
        for name, quantity, require in problem.system.restrictions:
            op, bound = parse_requirement(require)
            if op not in _SIGN:
                raise ValueError(f"restriction {name}: only inequalities are supported so far, not {op!r}")
            self.restrictions.append((parse_expression(quantity), _SIGN[op], bound))
        self.report_exprs = [parse_expression(q) for q in self.report]
        self.restriction_values = jax.jit(self._restriction_values)
        self.report_values = jax.jit(self._report_values)
        self.satisfied = jax.jit(jax.vmap(lambda base, extra: jnp.all(self._restriction_values({**base, **extra}) >= 0), in_axes=(None, 0)))
        self.no_prior = [u for u in self.others if u.no_prior]
        if self.no_prior and (len(self.no_prior) > 1 or not isinstance(self.no_prior[0].domain, Orthogonal)):
            raise ValueError("ranges are supported for one rotation with no prior; this needs the general optimiser, not built yet")
        self.one_column = None
        self.report_sizes = None

    def _context(self, values):
        params = {k: v for k, v in values.items() if k in self.problem.system.parameters}
        solved = self.problem.solve(params)
        return Context(self.problem.system, solved, self.problem.observed, self.problem.outside, self.problem.start_state, self.series)

    def _restriction_values(self, values):
        """Every restriction as a flat vector that is >= 0 where the restriction holds."""
        ctx = self._context(values)
        parts = [sign * (jnp.ravel(evaluate_quantity(q, ctx)) - evaluate_quantity(b, ctx)) for q, sign, b in self.restrictions]
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

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
            satisfied = lambda candidate: self.satisfied(values, {k: v[None] for k, v in candidate.items()})[0]
            draw = self._draw(sample, values, key, max_tries)
            if draw is None:
                return None
            values.update(draw)
        if not self.no_prior:
            return values, None
        return self._bounds(values, key)

    def _draw(self, sample, values, key, max_tries, batch=500):
        tries = 0
        while tries < max_tries:
            key, sub = jax.random.split(key)
            candidates = sample(sub, batch)
            ok = np.asarray(self.satisfied(values, candidates))
            if ok.any():
                i = int(np.argmax(ok))
                return {k: v[i] for k, v in candidates.items()}
            tries += batch
        return None

    def _bounds(self, values, key):
        rotation = self.no_prior[0]
        n = rotation.shape[0]
        restrict = lambda Q: self.restriction_values({**values, rotation.name: Q})
        report = lambda Q: self.report_values({**values, rotation.name: Q})
        if self.one_column is None:
            form = linear_in_one_column(restrict, report, n, key)
            if form is None:
                raise ValueError("the restrictions or reported quantities depend on more than one column of the rotation, or not "
                                 "linearly; ranges then need the general optimiser, not built yet")
            self.one_column = form.column
            self.report_sizes = [int(np.size(np.asarray(evaluate_quantity(q, self._context({**values, rotation.name: jnp.eye(n)}))))) for q in self.report_exprs]
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
