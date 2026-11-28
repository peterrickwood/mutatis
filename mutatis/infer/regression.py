"""The exact posterior for models made of regressions with flat priors on the coefficients: draws without sampling error.

It applies when the model is, in effect, one or more blocks of regressions. A VAR is the standard one-block case; Read's
first-stage regression of a proxy on the VAR's variables, over a shorter sample, is a second block. The conditions:

- every variable can be read from the data (observed exactly, or a lag of something observed), so that each block's errors
  are its equations' residuals at the data with its shocks set to zero;
- in each block, those errors are an exact linear function of some coefficients with flat priors, and every equation of the
  block uses the same regressors;
- each block's errors have a covariance equal to one parameter of its own, with a Jeffreys or inverse Wishart prior, and are
  independent of every other block's errors;
- each block's equations give its outputs one for one (written `output = ...`), and no blocks depend on each other's current
  values in a circle, so the data's density is the product of the blocks' error densities;
- anything else has no effect on the likelihood, such as the rotation of an SVAR. Those are drawn from their prior subject to
  the restrictions, or, with no prior, given ranges (`mutatis.infer.identification`).

Each block uses the periods in which every value its equations need is present, so blocks can have different samples. A
variable with missing values must be an output of one block that uses it only in the current period, and no other block may
use it: then leaving out that block's periods drops exactly the missing values' densities and nothing else.

Each condition is checked numerically on the model itself, not assumed from what kind of model it is. Then, for each block
independently (Zellner; Kadiyala and Karlsson), Sigma | data ~ InverseWishart(S + Psi0, T - k + nu0) and the coefficients
B | Sigma, data are matrix normal around the least-squares estimate with covariance Sigma (x) (X'X)^-1, where T is the block's
number of periods, S its residual cross-product, k its number of regressors, and (nu0, Psi0) are zero for the Jeffreys prior.

With a flat prior the marginal likelihood is not defined, so none is reported.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.dist import InverseWishart, UniformOrthogonal
from mutatis.dist.domains import Orthogonal, PositiveDefinite, Real
from mutatis.infer.posterior import Posterior
from mutatis.measure.data import data_states


class NotApplicable(ValueError):
    pass


@dataclass(frozen=True)
class BlockRegression:
    """One block's regression: which entries of which unknowns are the coefficient of which regressor in which of its equations."""

    block: str
    coefficients: tuple[tuple[str, int, int, int], ...]   # (unknown name, flat position within it, equation row, regressor)
    covariance: str
    periods: np.ndarray       # the periods used, counted from the first period the model explains
    regressors: np.ndarray    # (periods, k)
    responses: np.ndarray     # (periods, rows): the errors when every coefficient is zero
    prior_df: float
    prior_scale: np.ndarray


@dataclass(frozen=True)
class RegressionForm:
    blocks: tuple[BlockRegression, ...]
    others: tuple = ()        # unknowns the likelihood ignores: drawn from their prior under the restrictions, or given ranges


def regression_form(problem) -> RegressionForm:
    system, data = problem.system, problem.data
    if system.has_expectations:
        raise NotApplicable("a model with expectations of future values is not a set of regressions")
    states = data_states(system, data.observed, data.outside)
    if states is None:
        raise NotApplicable("every variable must be read from the data (observed exactly, or a lag of something observed)")
    now, previous = states
    outside = data.outside[system.presample:]
    if np.isnan(outside).any():
        raise NotApplicable("an outside variable has missing values in the periods the model explains")

    coefficient_units, covariance_units, other_units = [], [], []
    for unknown in problem.layout.unknowns:
        family = unknown.prior.family if unknown.prior is not None else None
        if family == "flat" and isinstance(unknown.domain, Real):
            coefficient_units.append(unknown)
        elif family in ("jeffreys", "inverse_wishart") and isinstance(unknown.domain, PositiveDefinite):
            covariance_units.append(unknown)
        elif unknown.no_prior or (unknown.prior is not None and unknown.prior.proper):
            other_units.append(unknown)  # must not affect the likelihood: checked below
        else:
            raise NotApplicable(f"{unknown.name} has a prior ({family}) that is neither flat on a coefficient nor a covariance prior")
    if not covariance_units or not coefficient_units:
        raise NotApplicable("needs coefficients with flat priors and a covariance parameter for each block")

    sizes = [int(np.prod(u.shape)) for u in coefficient_units]
    p = sum(sizes)

    def values_for(beta, covariances, others):
        values, start = dict(problem.layout.fixed), 0
        for unit, size in zip(coefficient_units, sizes):
            values[unit.name] = beta[start:start + size].reshape(unit.shape)
            start += size
        return {**values, **covariances, **others}

    def params_of(values):
        return {k: v for k, v in values.items() if k in system.parameters}

    n, n_shocks = system.n_variables, len(system.shock_labels)

    def residual(values, now_t, previous_t, outside_t, shocks):
        return system.residual(jnp.zeros(n), now_t, previous_t, shocks, outside_t, params_of(values))

    rng = np.random.default_rng(0)
    beta1 = jnp.asarray(0.1 * rng.normal(size=p))
    covariances1 = {}
    for unit in covariance_units:
        A = rng.normal(size=unit.shape)
        covariances1[unit.name] = jnp.asarray(A @ A.T + unit.shape[0] * np.eye(unit.shape[0]))
    others1, others2 = ({u.name: _any_value(u, jax.random.key(seed)) for u in other_units} for seed in (1, 2))
    values1 = values_for(beta1, covariances1, others1)

    # Which values each equation row uses, at a generic point (random states, so that no dependence is hidden by a zero).
    point = [jnp.asarray(rng.normal(size=size)) for size in (n, n, outside.shape[1])]
    jac_now, jac_previous, jac_outside, jac_shocks = (np.asarray(j) for j in jax.jacfwd(lambda a, b, c, e: residual(values1, a, b, c, e), argnums=(0, 1, 2, 3))(*point, jnp.zeros(n_shocks)))
    rows = system.equation_rows()
    owned = {block: [system.variable_labels.index(label) for label in labels] for block, labels in system.outputs_by_block.items()}
    owned_anywhere = {i for indices in owned.values() for i in indices}
    if "auxiliary" in rows:
        owned["auxiliary"] = [i for i in range(n) if i not in owned_anywhere]

    _check_outputs(rows, owned, jac_now)
    _check_missing(system, rows, owned, now, previous, jac_now, jac_previous)

    # The errors of every row in every period, as a function of the coefficients; the states with missing values set to zero
    # (only rows that use them see them, and those periods are left out below).
    now0, previous0 = np.nan_to_num(now), np.nan_to_num(previous)
    all_errors = jax.vmap(lambda a, b, c, beta, cov, oth: residual(values_for(beta, cov, oth), a, b, c, jnp.zeros(n_shocks)), in_axes=(0, 0, 0, None, None, None))

    def errors(beta, covariances=covariances1, others=others1):
        return all_errors(now0, previous0, outside, beta, covariances, others)

    base = errors(jnp.zeros(p))
    X_all = -np.asarray(jax.jacfwd(errors)(jnp.zeros(p)))                       # (periods, rows, p)
    if not np.allclose(errors(beta1), base - jnp.einsum("trp,p->tr", X_all, beta1), rtol=1e-8, atol=1e-10):
        raise NotApplicable("the errors are not a linear function of the coefficients alone")
    if other_units and not np.allclose(errors(beta1, others=others2), errors(beta1), rtol=1e-10, atol=1e-12):
        raise NotApplicable(f"the likelihood depends on {', '.join(u.name for u in other_units)}, so the formula does not apply")
    base = np.asarray(base)

    # Each row's shocks and the covariance of its errors, which must not depend on the unknowns the likelihood ignores.
    shock_cov = _shock_covariance(problem, params_of(values1))
    loadings = {}
    for others in (others1, others2):
        J = np.asarray(jax.jacfwd(lambda e: residual(values_for(beta1, covariances1, others), *point, e))(jnp.zeros(n_shocks)))
        for block, r in rows.items():
            if block in loadings and not np.allclose(loadings[block], J[r] @ shock_cov @ J[r].T, rtol=1e-8):
                raise NotApplicable(f"the covariance of block {block}'s errors depends on {', '.join(u.name for u in other_units)}")
            loadings[block] = J[r] @ shock_cov @ J[r].T
    stochastic = [block for block, r in rows.items() if np.abs(jac_shocks[r]).max(initial=0) > 0]
    for i, a in enumerate(stochastic):
        for b in stochastic[i + 1:]:
            if np.abs(jac_shocks[rows[a]] @ shock_cov @ jac_shocks[rows[b]].T).max() > 0:
                raise NotApplicable(f"the errors of blocks {a} and {b} are correlated; each block's errors must be independent of the others'")

    owner = {}
    start = 0
    for unit, size in zip(coefficient_units, sizes):
        for position in range(size):
            owner[start + position] = (unit.name, position)
        start += size
    used_by = np.any(X_all != 0, axis=0)                                         # (rows, p)
    if not np.array_equal(used_by.sum(axis=0), np.ones(p)):
        raise NotApplicable("every coefficient must affect exactly one equation's error (an unused coefficient has a flat posterior)")

    blocks = []
    for block, r in rows.items():
        if block not in stochastic:
            needed = _needed(r, jac_now, jac_previous)
            usable = ~np.isnan(now[:, needed[0]]).any(axis=1) & ~np.isnan(previous[:, needed[1]]).any(axis=1)
            if np.abs(base[usable][:, r]).max(initial=0) > 1e-9 * max(1.0, np.abs(now0).max()) or used_by[r].any():
                raise NotApplicable(f"block {block} has no shocks, so its equations must hold exactly at the data with no coefficients to estimate")
            continue
        blocks.append(_block_regression(block, r, rows, now, previous, jac_now, jac_previous, X_all, base, used_by, owner,
                                        loadings[block], covariance_units, covariances1, values1, system))
    if len({b.covariance for b in blocks}) != len(blocks) or len(blocks) != len(covariance_units):
        raise NotApplicable("each block needs a covariance parameter of its own")
    return RegressionForm(tuple(blocks), tuple(other_units))


def _shock_covariance(problem, params) -> np.ndarray:
    noise = problem.solve(params).gaussian_noise()
    if noise is None:
        raise NotApplicable("the shocks must be normal")
    return np.asarray(noise.shock_cov)


def _check_outputs(rows, owned, jac_now):
    """Each block's equations give its outputs one for one, and no blocks depend on each other's current values in a circle:
    then the change of variables from errors to data has a Jacobian of one."""
    for block, r in rows.items():
        own = jac_now[np.ix_(r, owned.get(block, []))]
        if own.shape[0] != own.shape[1] or not (np.isin(own, (0.0, 1.0)).all() and (own.sum(axis=0) == 1).all() and (own.sum(axis=1) == 1).all()):
            raise NotApplicable(f"block {block}'s equations must each give one of its outputs, written as output = ...")
    depends = {a: {b for b in rows if b != a and np.abs(jac_now[np.ix_(rows[a], owned.get(b, []))]).max(initial=0) > 0} for a in rows}
    state = {}

    def visit(a):
        state[a] = "open"
        for b in depends[a]:
            if state.get(b) == "open" or (b not in state and visit(b)):
                return True
        state[a] = "done"
        return False

    if any(visit(a) for a in rows if a not in state):
        raise NotApplicable("some blocks depend on each other's current values in a circle")


def _needed(r, jac_now, jac_previous):
    return (np.flatnonzero(np.abs(jac_now[r]).max(axis=0) > 0), np.flatnonzero(np.abs(jac_previous[r]).max(axis=0) > 0))


def _check_missing(system, rows, owned, now, previous, jac_now, jac_previous):
    missing = np.flatnonzero(np.isnan(now).any(axis=0) | np.isnan(previous).any(axis=0))
    for i in missing:
        label = system.variable_labels[i]
        home = [b for b, indices in owned.items() if i in indices]
        users_now = [b for b, r in rows.items() if np.abs(jac_now[r, i]).max() > 0]
        users_previous = [b for b, r in rows.items() if np.abs(jac_previous[r, i]).max() > 0]
        if users_previous or any(b not in home for b in users_now):
            raise NotApplicable(f"{label} has missing values, so it may be used only by its own block and only in the current period "
                                f"(used now by {users_now}, lagged by {users_previous})")


def _block_regression(block, r, rows, now, previous, jac_now, jac_previous, X_all, base, used_by, owner, error_cov, covariance_units, covariances1, values1, system):
    needed_now, needed_previous = _needed(r, jac_now, jac_previous)
    usable = ~np.isnan(now[:, needed_now]).any(axis=1) & ~np.isnan(previous[:, needed_previous]).any(axis=1)
    periods = np.flatnonzero(usable)
    X = X_all[periods][:, r]                                                     # (periods, rows of the block, p)
    columns = [np.flatnonzero(used_by[row]) for row in r]
    if len({len(c) for c in columns}) != 1:
        raise NotApplicable(f"the equations of block {block} use different numbers of regressors")
    regressors = X[:, 0, columns[0]]
    if not all(np.allclose(X[:, i, c], regressors) for i, c in enumerate(columns)):
        raise NotApplicable(f"the equations of block {block} use different regressors")
    matches = [u for u in covariance_units if u.shape == error_cov.shape and np.allclose(error_cov, covariances1[u.name], rtol=1e-8)]
    if len(matches) != 1:
        raise NotApplicable(f"the covariance of block {block}'s errors is not one covariance parameter itself")
    covariance = matches[0]
    coefficients = tuple((*owner[int(c)], i, j) for i, row_columns in enumerate(columns) for j, c in enumerate(row_columns))
    prior = covariance.prior.build({k: v for k, v in values1.items() if k in system.parameters})
    rows_n = len(r)
    prior_df, prior_scale = (0.0, np.zeros((rows_n, rows_n))) if covariance.prior.family == "jeffreys" else (float(prior.df), np.asarray(prior.scale))
    return BlockRegression(block, coefficients, covariance.name, periods, regressors, base[periods][:, r], prior_df, prior_scale)


def _any_value(unit, key):
    """Some allowed value of an unknown: a draw from its prior, or for a rotation with no prior, a random rotation."""
    if unit.prior is not None:
        dist = unit.prior.build({})
        return dist.sample(key) if dist.event_shape else dist.sample(key, unit.shape)
    if isinstance(unit.domain, Orthogonal):
        return UniformOrthogonal(unit.shape[0]).sample(key)
    raise NotApplicable(f"{unit.name} has no prior and is not a rotation; ranges are only supported for rotations so far")


def _draw_block(block: BlockRegression, key, n_draws):
    """Exact draws of one block's covariance and coefficients: (covariances (draws, n, n), coefficients (draws, k, n)), and its
    least-squares summary."""
    X, Y = block.regressors, block.responses
    periods, k = X.shape
    n_series = Y.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    B_hat = XtX_inv @ X.T @ Y
    residuals = Y - X @ B_hat
    df = periods - k + block.prior_df
    if df <= n_series - 1:
        raise NotApplicable(f"block {block.block}: too few periods ({periods}) for {k} regressors and {n_series} equations")
    scale = residuals.T @ residuals + block.prior_scale
    key_cov, key_coef = jax.random.split(key)
    covariances = np.asarray(InverseWishart(df, jnp.asarray(scale)).sample(key_cov, (n_draws,)))
    L_x = np.linalg.cholesky(XtX_inv)
    W = np.asarray(jax.random.normal(key_coef, (n_draws, k, n_series)))
    B = B_hat + np.einsum("ij,njk,nlk->nil", L_x, W, np.linalg.cholesky(covariances))   # B_hat + L_x W L_cov'
    summary = {"least_squares": B_hat, "residual_cross_product": residuals.T @ residuals, "regressors": k, "periods": periods}
    return covariances, B, summary


def regression_posterior(problem, key, n_draws: int = 4000, report: tuple[str, ...] = (), max_tries: int = 10_000) -> Posterior:
    """Exact draws. With unknowns the likelihood ignores, each draw also gets: a value of them drawn from their prior that
    satisfies the restrictions; or, for a rotation with no prior, the range of every `report` quantity over all allowed rotations.
    Draws for which nothing satisfies the restrictions are dropped; the share kept is reported as `plausibility`. Restrictions
    that involve only what is estimated drop the draws that break them, the same truncation the other samplers apply through
    the log prior.
    """
    form = regression_form(problem)
    key_blocks, key_others = jax.random.split(key)
    drawn = [_draw_block(block, k, n_draws) for block, k in zip(form.blocks, jax.random.split(key_blocks, len(form.blocks)))]
    fitted = {block.covariance for block in form.blocks} | {c[0] for block in form.blocks for c in block.coefficients}

    def base_values(n):
        values = dict(problem.layout.fixed)
        for unit in problem.layout.unknowns:
            if unit.name in fitted:
                values[unit.name] = np.zeros(unit.shape)
        for block, (covariances, B, _) in zip(form.blocks, drawn):
            for name, position, row, regressor in block.coefficients:
                flat = values[name].reshape(-1).copy()
                flat[position] = B[n, regressor, row]
                values[name] = flat.reshape(values[name].shape)
            values[block.covariance] = covariances[n]
        return {k: jnp.asarray(v) for k, v in values.items()}

    diagnostics = {"blocks": {block.block: summary for block, (_, _, summary) in zip(form.blocks, drawn)}}
    if not form.others:
        kept = list(range(n_draws))
        if problem.restrictions:   # restrictions on what is estimated: the posterior truncated to where they hold
            allowed = jax.jit(lambda values: jnp.all(problem.restriction_values(values) >= 0))
            kept = [n for n in kept if bool(allowed(base_values(n)))]
            if not kept:
                raise NotApplicable("no draw satisfies the restrictions")
            diagnostics["plausibility"] = len(kept) / n_draws
        draws = [np.asarray(problem.layout.unconstrain(base_values(n))) for n in kept]
        return Posterior(np.stack(draws), np.full(len(draws), 1.0 / len(draws)), None, "exact_regression", diagnostics)

    from mutatis.infer.restricted import RestrictedOthers  # here to keep the plain case free of the quantity machinery

    others = RestrictedOthers(problem, form.others, report)
    draws, lower, upper = [], [], []
    for n, draw_key in zip(range(n_draws), jax.random.split(key_others, n_draws)):
        result = others.complete(base_values(n), draw_key, max_tries)
        if result is None:
            continue
        values, bounds = result
        draws.append(np.asarray(problem.layout.unconstrain(values)))
        if bounds is not None:
            lower.append(bounds[0])
            upper.append(bounds[1])
    if not draws:
        raise NotApplicable("no draw satisfies the restrictions")
    diagnostics["plausibility"] = len(draws) / n_draws
    if lower:
        diagnostics["bounds"] = others.split(np.stack(lower), np.stack(upper))
    return Posterior(np.stack(draws), np.full(len(draws), 1.0 / len(draws)), None, "exact_regression", diagnostics)
