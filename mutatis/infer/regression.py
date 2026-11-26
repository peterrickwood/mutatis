"""The exact posterior for regressions with flat priors on the coefficients: draws without sampling error from a sampler.

It applies when the model is, in effect, a set of regressions that all use the same regressors (a VAR is the standard case):

- the shocks are read off the data (every series observed exactly);
- the one-step-ahead errors v(t) = y(t) - E[y(t) | past] are an exact linear function of some coefficients, which have flat
  priors, and every series' error uses the same regressors;
- the errors' covariance equals one parameter, Sigma, with a Jeffreys or inverse Wishart prior;
- anything else has no effect on the likelihood (checked), such as the rotation of an SVAR. Those are drawn from their prior
  subject to the restrictions, or, with no prior, given ranges (`mutatis.infer.identification`).

Each condition is checked numerically on the model itself, not assumed from what kind of model it is. Then (Zellner; Kadiyala
and Karlsson) Sigma | data ~ InverseWishart(S + Psi0, T - k + nu0) and the coefficients B | Sigma, data are matrix normal around
the least-squares estimate with covariance Sigma (x) (X'X)^-1, where S is the residual cross-product, k the number of regressors,
and (nu0, Psi0) are zero for the Jeffreys prior.

With a flat prior the marginal likelihood is not defined, so none is reported.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.quantities.recover import recover_shocks
from mutatis.dist import InverseWishart, UniformOrthogonal
from mutatis.dist.domains import Orthogonal, PositiveDefinite, Real
from mutatis.infer.posterior import Posterior


class NotApplicable(ValueError):
    pass


@dataclass(frozen=True)
class RegressionForm:
    """What the checks found: which entries of which unknowns are the coefficient of which regressor in which equation."""

    coefficients: tuple[tuple[str, int, int, int], ...]   # (unknown name, flat position within it, series, regressor)
    covariance: str
    regressors: np.ndarray        # (periods, k)
    responses: np.ndarray         # (periods, series): the errors when every coefficient is zero
    prior_df: float
    prior_scale: np.ndarray
    others: tuple = ()            # unknowns the likelihood ignores: drawn from their prior under the restrictions, or given ranges


def _errors(problem, values: dict):
    """The one-step-ahead errors of every observed series in every period, for given values of the unknowns."""
    params = {k: v for k, v in values.items() if k in problem.system.parameters}
    solved = problem.solve(params)
    form = solved.linear_form()
    _, states = recover_shocks(solved, problem.observed, problem.outside, problem.start_state)
    previous = jnp.concatenate([problem.start_state[None], states[:-1]])
    predicted = form.kappa + previous @ form.P.T + problem.outside @ form.R.T
    return problem.observed - form.d - predicted @ form.Z.T


def regression_form(problem) -> RegressionForm:
    if problem.route != "read_off":
        raise NotApplicable("the exact regression formula needs the shocks read off the data")
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
    if len(covariance_units) != 1 or not coefficient_units:
        raise NotApplicable("needs coefficients with flat priors and exactly one covariance parameter")
    covariance = covariance_units[0]
    n_series = problem.observed.shape[1]
    if covariance.shape != (n_series, n_series):
        raise NotApplicable("the covariance parameter must be one entry per pair of observed series")

    sizes = [int(np.prod(u.shape)) for u in coefficient_units]

    def values_for(beta, cov, others=None):
        values, start = dict(problem.layout.fixed), 0
        for unit, size in zip(coefficient_units, sizes):
            values[unit.name] = beta[start:start + size].reshape(unit.shape)
            start += size
        values[covariance.name] = cov
        values.update(others if others is not None else {u.name: _any_value(u, jax.random.key(0)) for u in other_units})
        return values

    p = sum(sizes)
    errors = lambda beta, cov: _errors(problem, values_for(beta, cov)).reshape(-1)
    identity = jnp.eye(n_series)
    base = errors(jnp.zeros(p), identity)
    X = -jax.jacfwd(errors)(jnp.zeros(p), identity)
    rng = np.random.default_rng(0)
    beta1 = jnp.asarray(0.1 * rng.normal(size=p))
    A = rng.normal(size=(n_series, n_series))
    cov1 = jnp.asarray(A @ A.T + n_series * np.eye(n_series))
    if not np.allclose(errors(beta1, cov1), base - X @ beta1, rtol=1e-8, atol=1e-10):
        raise NotApplicable("the errors are not a linear function of the coefficients alone")

    # Every series' error must use the same regressors.
    X = np.asarray(X).reshape(problem.observed.shape[0], n_series, p)
    columns = [np.flatnonzero(np.any(X[:, i, :] != 0, axis=0)) for i in range(n_series)]
    used = np.concatenate(columns)
    if len(used) != p or len(set(used.tolist())) != p:
        raise NotApplicable("every coefficient must affect exactly one series' error (an unused coefficient has a flat posterior)")
    if len({len(c) for c in columns}) != 1:
        raise NotApplicable("the series' errors use different numbers of regressors")
    regressors = X[:, 0, columns[0]]
    if not all(np.allclose(X[:, i, columns[i]], regressors) for i in range(n_series)):
        raise NotApplicable("the series' errors use different regressors")

    # The errors' covariance must be the covariance parameter itself, whatever the other unknowns are.
    for seed in (1, 2):
        others = {u.name: _any_value(u, jax.random.key(seed)) for u in other_units}
        solved = problem.solve({k: v for k, v in values_for(beta1, cov1, others).items() if k in problem.system.parameters})
        form, noise = solved.linear_form(), solved.gaussian_noise()
        if noise is None or not np.allclose(form.Z @ form.Q @ noise.shock_cov @ (form.Z @ form.Q).T, cov1, rtol=1e-8):
            raise NotApplicable(f"the errors' covariance is not {covariance.name} itself")
        if other_units and not np.allclose(_errors(problem, values_for(beta1, cov1, others)).reshape(-1), errors(beta1, cov1), rtol=1e-10):
            raise NotApplicable(f"the likelihood depends on {', '.join(u.name for u in other_units)}, so the formula does not apply")

    owner = {}
    start = 0
    for unit, size in zip(coefficient_units, sizes):
        for position in range(size):
            owner[start + position] = (unit.name, position)
        start += size
    coefficients = tuple((*owner[int(c)], i, j) for i in range(n_series) for j, c in enumerate(columns[i]))
    params = {k: v for k, v in values_for(beta1, cov1).items() if k in problem.system.parameters}
    prior = covariance.prior.build(params)
    prior_df, prior_scale = (0.0, np.zeros((n_series, n_series))) if covariance.prior.family == "jeffreys" else (float(prior.df), np.asarray(prior.scale))
    return RegressionForm(coefficients, covariance.name, regressors, np.asarray(base).reshape(-1, n_series), prior_df, prior_scale, tuple(other_units))


def _any_value(unit, key):
    """Some allowed value of an unknown: a draw from its prior, or for a rotation with no prior, a random rotation."""
    if unit.prior is not None:
        dist = unit.prior.build({})
        return dist.sample(key) if dist.event_shape else dist.sample(key, unit.shape)
    if isinstance(unit.domain, Orthogonal):
        return UniformOrthogonal(unit.shape[0]).sample(key)
    raise NotApplicable(f"{unit.name} has no prior and is not a rotation; ranges are only supported for rotations so far")


def regression_posterior(problem, key, n_draws: int = 4000, report: tuple[str, ...] = (), max_tries: int = 10_000) -> Posterior:
    """Exact draws. With unknowns the likelihood ignores, each draw also gets: a value of them drawn from their prior that
    satisfies the restrictions; or, for a rotation with no prior, the range of every `report` quantity over all allowed rotations.
    Draws for which nothing satisfies the restrictions are dropped; the share kept is reported as `plausibility`.
    """
    form = regression_form(problem)
    X, Y = form.regressors, form.responses
    periods, k = X.shape
    n_series = Y.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    B_hat = XtX_inv @ X.T @ Y
    residuals = Y - X @ B_hat
    df = periods - k + form.prior_df
    if df <= n_series - 1:
        raise NotApplicable(f"too few periods ({periods}) for {k} regressors and {n_series} series")
    scale = residuals.T @ residuals + form.prior_scale
    key_cov, key_coef, key_others = jax.random.split(key, 3)
    covariances = InverseWishart(df, jnp.asarray(scale)).sample(key_cov, (n_draws,))
    L_x = np.linalg.cholesky(XtX_inv)
    W = np.asarray(jax.random.normal(key_coef, (n_draws, k, n_series)))
    L_cov = np.linalg.cholesky(np.asarray(covariances))
    B = B_hat + np.einsum("ij,njk,nlk->nil", L_x, W, L_cov)   # B_hat + L_x W L_cov'

    def base_values(n):
        values = dict(problem.layout.fixed)
        for unit in problem.layout.unknowns:
            if unit.name != form.covariance and unit not in form.others:
                values[unit.name] = np.zeros(unit.shape)
        for name, position, series, regressor in form.coefficients:
            flat = values[name].reshape(-1).copy()
            flat[position] = B[n, regressor, series]
            values[name] = flat.reshape(values[name].shape)
        values[form.covariance] = np.asarray(covariances[n])
        return {k: jnp.asarray(v) for k, v in values.items()}

    diagnostics = {"least_squares": B_hat, "residual_cross_product": residuals.T @ residuals, "regressors": k}
    if not form.others:
        draws = [np.asarray(problem.layout.unconstrain(base_values(n))) for n in range(n_draws)]
        return Posterior(np.stack(draws), np.full(n_draws, 1.0 / n_draws), None, "exact_regression", diagnostics)

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
