"""Solving a linear equation system: the matrices, then the rule for stepping forward.

The system's residual is linear, so its derivatives at zero give it exactly:

    A E[x(t+1)] + B x(t) + C x(t-1) + D e(t) + G z(t) + f = 0

(x variables, e shocks, z outside variables). With expectation terms the forward-looking solve applies; without, the backward
one. The constant f moves the resting point: kappa solves (A P + B + A) kappa = -f, because E[x(t+1)] = kappa + P x(t).

Two limits for now: the system must be linear (nonlinear systems need a resting point and a linear approximation around it,
which comes in a later stage), and a model with expectation terms may not have outside variables, because the solution would
then depend on what people expect the outside variables to do.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.system import System
from mutatis.dist import MultivariateNormal
from mutatis.numerics.kalman import stationary_covariance
from mutatis.numerics.linear_re import solve_backward, solve_forward_looking
from mutatis.numerics.status import Status
from mutatis.solve.contract import GaussianNoise, LinearForm, SolvedModel


class SolveError(ValueError):
    pass


class LinearSolved(SolvedModel):
    status: jax.Array
    exact: tuple[bool, ...] = eqx.field(static=True)
    kappa: jax.Array
    P: jax.Array
    Q: jax.Array
    R: jax.Array
    d: jax.Array
    Z: jax.Array
    shock_dists: tuple
    shock_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    shock_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    error_dists: tuple
    all_gaussian: bool = eqx.field(static=True)
    initial_mean: jax.Array | None
    initial_cov: jax.Array | None

    def step(self, state, shocks, outside):
        return self.kappa + self.P @ state + self.Q @ shocks + self.R @ outside

    def observe(self, state):
        return self.d + self.Z @ state

    def shock_log_prob(self, shocks):
        total = jnp.zeros(())
        for dist, positions, shape in zip(self.shock_dists, self.shock_positions, self.shock_shapes):
            values = shocks[np.array(positions, dtype=int)].reshape(shape)
            total = total + jnp.sum(dist.log_prob(values))
        return total

    def sample_shocks(self, key, n_periods):
        k = sum(len(p) for p in self.shock_positions)
        out = jnp.zeros((n_periods, k))
        for dist, positions, shape, subkey in zip(self.shock_dists, self.shock_positions, self.shock_shapes, jax.random.split(key, len(self.shock_dists))):
            draws = dist.sample(subkey, (n_periods,)) if dist.event_shape else dist.sample(subkey, (n_periods, *shape))
            out = out.at[:, np.array(positions, dtype=int)].set(draws.reshape(n_periods, -1))
        return out

    def error_log_prob(self, observed, predicted):
        total = jnp.zeros(())
        for i, dist in enumerate(self.error_dists):
            if dist is None:
                continue
            seen = ~jnp.isnan(observed[i])
            total = total + jnp.where(seen, dist.log_prob(jnp.where(seen, observed[i] - predicted[i], 0.0)), 0.0)
        return total

    def initial_gaussian(self):
        return None if self.initial_mean is None else (self.initial_mean, self.initial_cov)

    def initial_log_prob(self, state):
        if self.initial_mean is None:
            raise SolveError("this model has no starting distribution: give `initial` values, or start from the data")
        return MultivariateNormal(self.initial_mean, self.initial_cov).log_prob(state)

    def sample_initial(self, key):
        if self.initial_mean is None:
            raise SolveError("this model has no starting distribution: give `initial` values, or start from the data")
        return MultivariateNormal(self.initial_mean, self.initial_cov).sample(key)

    def linear_form(self):
        return LinearForm(self.kappa, self.P, self.Q, self.R, self.d, self.Z)

    def gaussian_noise(self):
        if not self.all_gaussian:
            return None
        cov = shock_covariance(self.shock_dists, self.shock_positions, self.shock_shapes)
        error_variance = jnp.array([0.0 if dist is None else dist.scale ** 2 for dist in self.error_dists]) if self.error_dists else jnp.zeros(0)
        return GaussianNoise(cov, error_variance)


def shock_covariance(shock_dists, positions, shapes):
    """The covariance of one period's shocks, all normal with mean zero, over the flat shock vector."""
    k = sum(len(p) for p in positions)
    cov = jnp.zeros((k, k))
    for dist, where, shape in zip(shock_dists, positions, shapes):
        index = np.array(where, dtype=int)
        block = dist.cov if isinstance(dist, MultivariateNormal) else jnp.diag(jnp.broadcast_to(jnp.asarray(dist.scale) ** 2, shape).reshape(-1))
        cov = cov.at[np.ix_(index, index)].set(block)
    return cov


def solve_linear(system: System, params: dict) -> LinearSolved:
    if not system.linear:
        raise SolveError("this system is not linear; nonlinear systems are not supported yet")
    n, k, m = system.n_variables, len(system.shock_labels), len(system.outside_labels)
    zeros = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), jnp.zeros(k), jnp.zeros(m))
    A, B, C, D, G = jax.jacfwd(system.residual, argnums=(0, 1, 2, 3, 4))(*zeros, params)
    f = system.residual(*zeros, params)
    if system.has_expectations:
        if m:
            raise SolveError("a model with expectation terms cannot have outside variables yet: give each one a block of its own")
        P, Q, status = solve_forward_looking(A, B, C, D)
        ok = status == Status.OK
        kappa = -jnp.linalg.solve(jnp.where(ok, A @ P + B + A, jnp.eye(n)), f)
        R = jnp.zeros((n, m))
    else:
        P, Q, status = solve_backward(B, C, jnp.concatenate([D, G, f[:, None]], axis=1))
        Q, R, kappa = Q[:, :k], Q[:, k:k + m], Q[:, -1]
    d = system.observe(jnp.zeros(n), params)
    Z = jax.jacfwd(system.observe)(jnp.zeros(n), params) if system.observations else jnp.zeros((0, n))

    shock_dists, positions, shapes = [], [], []
    for info in system.shocks.values():
        shock_dists.append(info.distribution.build(params))
        positions.append(tuple(int(i) for i in system.layout.shocks[info.key].positions.reshape(-1)))
        shapes.append(info.shape)
    error_dists = tuple(None if o.error is None else o.error.build(params) for o in system.observations)
    all_gaussian = all(s.distribution.gaussian for s in system.shocks.values()) and all(o.error is None or o.error.gaussian for o in system.observations)

    initial_mean, initial_cov = _initial(system, params, P, Q, kappa, m, all_gaussian, shock_dists, positions, shapes)
    solved = LinearSolved(status, tuple(o.error is None for o in system.observations), kappa, P, Q, R, d, Z, tuple(shock_dists), tuple(positions),
                          tuple(shapes), error_dists, all_gaussian, initial_mean, initial_cov)
    return solved


def _initial(system: System, params, P, Q, kappa, m, all_gaussian, shock_dists, positions, shapes):
    """The starting distribution: the `initial` priors if every variable has one, otherwise the long-run distribution if it exists."""
    n = system.n_variables
    if system.initial and set(system.initial) == set(system.model_variables) and not system.auxiliary:
        means, variances = jnp.zeros(n), jnp.zeros(n)
        for variable, spec in system.initial.items():
            if spec.family != "normal":
                raise SolveError(f"the starting value of {variable!r} must have a normal distribution for now")
            dist = spec.build(params)
            index = system.layout.variables[variable].positions.reshape(-1)
            means = means.at[index].set(jnp.broadcast_to(dist.loc, index.shape))
            variances = variances.at[index].set(jnp.broadcast_to(jnp.asarray(dist.scale) ** 2, index.shape))
        return means, jnp.diag(variances)
    if system.initial:
        raise SolveError("give a starting value (`initial`) for every variable, or for none")
    if m or not all_gaussian:
        return None, None
    cov, status = stationary_covariance(P, Q @ shock_covariance(shock_dists, positions, shapes) @ Q.T)
    mean = jnp.linalg.solve(jnp.eye(n) - P, kappa)
    usable = status == Status.OK
    return jnp.where(usable, mean, jnp.nan), jnp.where(usable, cov, jnp.nan)
