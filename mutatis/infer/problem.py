"""Contract 2: the posterior problem, which every sampler consumes.

A posterior problem is a vector of unconstrained numbers z, a log prior and a log likelihood as two separate functions of z
(bringing the data in gradually needs them apart), a way to draw z from the prior, and a layout that says which part of z is
which unknown. `build_problem` assembles one from a compiled system, a solver (a function from parameter values to a solved
model, so that this layer never depends on how models are solved), and arranged data.

The unknowns are the parameters that have a prior; with the `sample` route also every period's shocks, in standard form (the
shocks are shocks_from_standard(u), see contract 1), and, when the starting state is not read from the data, a standard normal
vector u that gives it as mean + L u (see `_initial_from_standard`). Standard forms keep the sampler's geometry from changing
with the parameters, which matters: with the shocks themselves as unknowns, SMC returned confidently wrong posteriors (risk
experiment 1, PLAN.md). Fixed parameters are constants. Parameters with no prior need the bounds route, which comes
with the first demonstration.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.distributions import DistributionSpec
from mutatis.compile.system import System
from mutatis.dist.domains import Domain, Real
from mutatis.infer import routes
from mutatis.measure.data import Arranged
from mutatis.numerics.factor import cholesky_with_jitter
from mutatis.solve.contract import SolvedModel

ROUTES = ("read_off", "kalman", "sample")


class ProblemError(ValueError):
    pass


@dataclass(frozen=True)
class Unknown:
    name: str
    shape: tuple[int, ...]
    domain: Domain
    prior: DistributionSpec | None   # None for shocks and the starting state, whose distributions come from the solved model
    offset: int
    size: int                        # length of its unconstrained part of z
    no_prior: bool = False           # restricted, with no prior behind it: results are reported as ranges


@dataclass(frozen=True)
class UnknownLayout:
    unknowns: tuple[Unknown, ...]
    fixed: dict

    @property
    def size(self) -> int:
        return sum(u.size for u in self.unknowns)

    def constrain(self, z) -> tuple[dict, jax.Array]:
        """Values of every unknown (and the fixed parameters) from z, and the log Jacobian of the map from z."""
        values, log_det = dict(self.fixed), jnp.zeros(())
        for u in self.unknowns:
            part = z[u.offset:u.offset + u.size]
            values[u.name] = u.domain.to_constrained(part, u.shape)
            log_det = log_det + u.domain.log_det_jacobian(part, u.shape)
        return values, log_det

    def unconstrain(self, values: dict) -> jax.Array:
        return jnp.concatenate([jnp.ravel(u.domain.to_unconstrained(values[u.name])) for u in self.unknowns]) if self.unknowns else jnp.zeros(0)


class PosteriorProblem:
    def __init__(self, system: System, solve: Callable[[dict], SolvedModel], data: Arranged, route: str, start: str, layout: UnknownLayout,
                 observed, outside, start_state):
        self.system, self.solve, self.data, self.route, self.start, self.layout = system, solve, data, route, start, layout
        self.observed, self.outside, self.start_state = observed, outside, start_state
        self.parameter_unknowns = tuple(u for u in layout.unknowns if u.prior is not None)
        self.no_prior_unknowns = tuple(u for u in layout.unknowns if u.no_prior)
        self.proper = all(u.prior.proper for u in self.parameter_unknowns) and not self.no_prior_unknowns
        self.gradient_ready = all(u.domain.has_transform for u in layout.unknowns) and not self.no_prior_unknowns

    @property
    def size(self) -> int:
        return self.layout.size

    def parameters(self, z) -> dict:
        values, _ = self.layout.constrain(z)
        return {k: v for k, v in values.items() if k in self.system.parameters}

    def log_densities(self, z) -> tuple[jax.Array, jax.Array]:
        """(log prior, log likelihood) at z, solving the model once."""
        values, log_det = self.layout.constrain(z)
        params = {k: v for k, v in values.items() if k in self.system.parameters}
        log_prior = log_det
        for u in self.parameter_unknowns:
            log_prior = log_prior + jnp.sum(u.prior.build(params).log_prob(values[u.name]))
        solved = self.solve(params)
        if self.route == "sample":
            standard = values["shocks"]
            log_prior = log_prior + jnp.sum(jax.vmap(solved.standard_log_prob)(standard))
            shocks = jax.vmap(solved.shocks_from_standard)(standard)
            if self.start == "model":
                standard = values["initial"]
                start = _initial_from_standard(solved, standard)
                log_prior = log_prior - 0.5 * jnp.sum(standard**2) - 0.5 * standard.size * jnp.log(2 * jnp.pi)
            else:
                start = self.start_state
            log_likelihood = routes.sample(solved, self.observed, self.outside, shocks, start)
        elif self.route == "kalman":
            log_likelihood = routes.kalman(solved, self.observed, self.outside, None if self.start == "model" else self.start_state)
        else:
            log_likelihood = routes.read_off(solved, self.observed, self.outside, self.start_state)
        log_prior = jnp.where(jnp.isfinite(log_prior), log_prior, -jnp.inf)
        return log_prior, log_likelihood

    def log_prior(self, z):
        return self.log_densities(z)[0]

    def log_likelihood(self, z):
        return self.log_densities(z)[1]

    def log_posterior(self, z):
        log_prior, log_likelihood = self.log_densities(z)
        return log_prior + log_likelihood

    def draw_prior(self, key) -> jax.Array:
        """One draw of z from the prior. Every prior must be proper."""
        if not self.proper:
            raise ProblemError("cannot draw from an improper prior (flat or jeffreys) or from a parameter with no prior; use the exact formula instead")
        values = dict(self.layout.fixed)
        keys = jax.random.split(key, len(self.layout.unknowns) + 2)
        for u, subkey in zip(self.parameter_unknowns, keys):
            params = {k: v for k, v in values.items() if k in self.system.parameters}
            dist = u.prior.build(params)
            values[u.name] = dist.sample(subkey) if dist.event_shape else dist.sample(subkey, u.shape)
        if self.route == "sample":
            solved = self.solve({k: v for k, v in values.items() if k in self.system.parameters})
            values["shocks"] = solved.sample_standard(keys[-2], self.observed.shape[0])
            if self.start == "model":
                values["initial"] = jax.random.normal(keys[-1], (self.system.n_variables,))
        return self.layout.unconstrain(values)


def _initial_from_standard(solved: SolvedModel, standard):
    """The starting state as mean + L u, with u standard normal and L a Cholesky factor of the starting covariance.

    Sampling u rather than the state itself works even when the covariance is singular, which it is whenever some variables are
    exact functions of others (the price in the forward-looking toy is a fixed multiple of demand).
    """
    initial = solved.initial_gaussian()
    if initial is None:
        raise ProblemError("sampling the starting state needs a normal starting distribution: give `initial` values, or start from the data")
    mean, cov = initial
    return mean + cholesky_with_jitter(cov) @ standard


def choose_route(system: System, data: Arranged) -> str:
    """The cheapest method that applies to this model and data. The choice depends only on the specification and the data. Every
    solution is linear for now (a nonlinear system is solved to first order around its steady state), so the conditions on
    linearity are on the solution, not on the equations."""
    exact = [o.error is None for o in system.observations]
    first = data.start.first if data.start else 0
    complete = not np.isnan(data.observed[first:]).any()
    gaussian = all(s.distribution.gaussian for s in system.shocks.values()) and all(o.error is None or o.error.gaussian for o in system.observations)
    if all(exact) and len(exact) == len(system.shock_labels) and data.start is not None and complete:
        return "read_off"
    if gaussian:
        return "kalman"
    if not any(exact):
        return "sample"
    raise ProblemError("no method applies: series observed exactly need either normal noise and a linear model (kalman), or as many "
                       "shocks as series and a starting state readable from the data (read_off)")


def build_problem(system: System, solve: Callable[[dict], SolvedModel], data: Arranged, route: str = "auto", start: str = "auto") -> PosteriorProblem:
    route = choose_route(system, data) if route == "auto" else route
    if route not in ROUTES:
        raise ProblemError(f"unknown route {route!r}; known: {ROUTES}")
    if start == "auto":
        start = "data" if data.start is not None else "model"
    if start == "data" and data.start is None:
        raise ProblemError("the starting state cannot be read from the data: some part of it is not observed exactly")
    if route == "read_off" and start != "data":
        raise ProblemError("reading shocks off the data needs the starting state from the data")
    first = data.start.first if start == "data" else 0
    observed, outside = jnp.asarray(data.observed[first:]), jnp.asarray(data.outside[first:])
    start_state = jnp.asarray(data.start.state) if start == "data" else None

    fixed, unknowns, offset = {}, [], 0
    for key, info in system.parameters.items():
        if info.kind == "fixed":
            fixed[key] = info.fixed_value
            continue
        size = info.domain.unconstrained_size(info.shape)
        unknowns.append(Unknown(key, info.shape, info.domain, info.prior, offset, size, info.kind == "no_prior"))
        offset += size
    if route == "sample":
        shape = (observed.shape[0], len(system.shock_labels))
        unknowns.append(Unknown("shocks", shape, Real(), None, offset, math.prod(shape)))
        offset += math.prod(shape)
        if start == "model":
            unknowns.append(Unknown("initial", (system.n_variables,), Real(), None, offset, system.n_variables))
            offset += system.n_variables
    layout = UnknownLayout(tuple(unknowns), fixed)
    return PosteriorProblem(system, solve, data, route, start, layout, observed, outside, start_state)
