"""Reference posteriors by numerical integration on a grid, for targets with two or three unknowns.

The grid is a box in the unconstrained unknowns around the posterior's peak. Along each axis the points are evenly spaced in
u, with z = peak + scale * sinh(u) and `scale` the posterior standard deviation from the curvature at the peak: points crowd
where the mass is and thin out in the tails, so the box can reach far into a long tail while the bulk stays finely resolved.
The log posterior is evaluated at every point; the evidence is the trapezoid rule in u (each point weighted by
scale * cosh(u) * the spacing in u), and each unknown's marginal is the sum over the other axes. For a smooth density that has
decayed to nothing at the edges, the trapezoid rule converges faster than any power of the spacing; the cumulative
distributions, read off between points, are accurate to the square of the spacing. Each face of the box is pushed outwards,
separately, until the density on it is below e^-25 of the peak: posteriors of a log standard deviation can have a long tail on
one side only (towards zero, where the data stop distinguishing values).

Values are reported for the constrained quantities (for example a standard deviation rather than its logarithm): `constrain`
maps each axis's grid values to its quantity, and must be increasing, as the log and logistic transforms are.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from mutatis import _jax_config  # noqa: F401  (64-bit floats)


class QuadratureError(RuntimeError):
    pass


@dataclass(frozen=True)
class GridPosterior:
    log_evidence: float
    mean: np.ndarray          # of each constrained quantity
    sd: np.ndarray
    nodes: tuple              # per quantity: constrained values at the grid nodes along its axis (increasing)
    cumulative: tuple         # per quantity: the marginal cumulative probability at those nodes
    width: float              # the box's widest reach from the peak, in posterior sds from the curvature at the peak
    edge_log_ratio: float     # largest log density on the box's faces, relative to the peak

    def cdf(self, j: int, values) -> np.ndarray:
        return np.interp(np.asarray(values, dtype=float), self.nodes[j], self.cumulative[j], left=0.0, right=1.0)


def find_peak(log_posterior, starts: np.ndarray) -> np.ndarray:
    """The highest peak reached by BFGS from each start."""
    value_and_grad = jax.jit(jax.value_and_grad(lambda z: -log_posterior(z)))

    def fun(z):
        v, g = value_and_grad(jnp.asarray(z))
        v, g = float(v), np.asarray(g)
        return (v, g) if np.isfinite(v) and np.all(np.isfinite(g)) else (1e300, np.zeros_like(g))

    best, best_value = None, np.inf
    for start in starts:
        result = minimize(fun, np.asarray(start, dtype=float), jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
        if np.isfinite(result.fun) and result.fun < best_value:
            best, best_value = result.x, result.fun
    if best is None:
        raise QuadratureError("no start reached a finite peak")
    return best


def grid_posterior(log_posterior, constrain, starts: np.ndarray, n_points: int, width: float = 10.0, batch: int = 8192,
                   max_widenings: int = 12) -> GridPosterior:
    """`log_posterior(z)`: log prior + log likelihood of the unconstrained unknowns (a density in z, Jacobian included).
    `constrain(k, values)`: the constrained quantity for axis k's unconstrained values."""
    peak = find_peak(log_posterior, starts)
    size = len(peak)
    hessian = np.asarray(jax.hessian(log_posterior)(jnp.asarray(peak)))
    try:
        cov = np.linalg.inv(-hessian)
        scale = np.sqrt(np.diag(cov))
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        raise QuadratureError(f"the curvature at the peak is not negative definite: {hessian}") from None
    evaluate = jax.jit(jax.vmap(log_posterior))
    # Each axis is z = peak + scale * sinh(u) with u evenly spaced: points crowd where the mass is and thin out in the tails,
    # so a box many standard deviations wide still has fine spacing in the bulk. The sum is the trapezoid rule in u.
    lower, upper = np.full(size, -np.arcsinh(width)), np.full(size, np.arcsinh(width))
    for _ in range(max_widenings + 1):
        us = [np.linspace(lower[k], upper[k], n_points) for k in range(size)]
        axes = [peak[k] + scale[k] * np.sinh(us[k]) for k in range(size)]
        log_w = [np.log(scale[k] * np.cosh(us[k]) * (us[k][1] - us[k][0])) for k in range(size)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, size)
        values = np.concatenate([np.asarray(evaluate(jnp.asarray(grid[i:i + batch]))) for i in range(0, len(grid), batch)])
        values = np.where(np.isnan(values), -np.inf, values).reshape((n_points,) * size)
        top = values.max()
        if not np.isfinite(top):
            raise QuadratureError("the log posterior is not finite anywhere on the grid")
        edges = {(k, i): float(np.max(np.take(values, i, axis=k))) - top for k in range(size) for i in (0, n_points - 1)}
        edge = max(edges.values())
        if edge < -25.0:
            break
        for (k, i), e in edges.items():
            if e >= -25.0:
                if i == 0:
                    lower[k] -= 1.0
                else:
                    upper[k] += 1.0
    else:
        raise QuadratureError(f"the posterior has not decayed at the box's edges (log ratio {edge:.1f}) after {max_widenings} widenings")
    width = float(np.sinh(max(np.max(-lower), np.max(upper))))
    weighted = values.copy()
    for k in range(size):
        shape = [1] * size
        shape[k] = n_points
        weighted = weighted + log_w[k].reshape(shape)
    log_evidence = float(logsumexp(weighted))
    masses = np.exp(weighted - log_evidence)
    means, sds, nodes, cumulative = [], [], [], []
    for k in range(size):
        marginal = masses.sum(axis=tuple(j for j in range(size) if j != k))
        c = np.asarray(constrain(k, axes[k]), dtype=float)
        if np.any(np.diff(c) <= 0):
            raise QuadratureError(f"the transform of axis {k} is not increasing")
        mean = float(np.sum(marginal * c))
        means.append(mean)
        sds.append(float(np.sqrt(max(np.sum(marginal * c**2) - mean**2, 0.0))))
        nodes.append(c)
        cumulative.append(np.cumsum(marginal) - marginal / 2)
    return GridPosterior(log_evidence, np.array(means), np.array(sds), tuple(nodes), tuple(cumulative), width, edge)
