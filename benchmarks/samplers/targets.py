"""Synthetic posterior problems with known answers, for comparing samplers.

Each target implements contract 2 (`size`, `proper`, `gradient_ready`, `log_prior`, `log_likelihood`, `draw_prior`) and carries
a `Reference`: the true mean, standard deviation and, where known, cumulative distribution of each checked quantity, the true
log evidence, and for targets with separate peaks the true weight of each peak. `source` says how each reference was
obtained; tests/benchmarks checks every one independently (exact samplers, Monte Carlo, numerical integration).

Most targets use one device: for a normalised density p and a normal prior, the likelihood p / prior * exp(c) gives the
posterior p exactly and the evidence exp(c) exactly, whatever p is. The prior on each unknown is centred at zero with
standard deviation |mean| + 5 sd of that unknown under p, so it covers p comfortably.

`TARGETS` maps each name to a builder; `build(name)` makes the target.
"""

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, multigammaln
from jax.scipy.stats import norm
from scipy import stats

from benchmarks.samplers.quadrature import grid_posterior
from mutatis import _jax_config  # noqa: F401  (64-bit floats, as everywhere in mutatis)
from tests.infer.targets import NormalMean


@dataclass(frozen=True)
class Reference:
    names: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray
    cdfs: tuple                      # per quantity: a function from values to cumulative probabilities, or None if unknown
    log_evidence: float
    source: str
    peak_weights: tuple[float, ...] | None = None


class Target:
    """Contract 2 plus the known answer. Subclasses set name, group, description, size and reference."""

    proper = True
    gradient_ready = True
    name: str
    group: str
    description: str
    size: int

    def quantities(self, z: np.ndarray) -> np.ndarray:
        """The checked quantities for draws z of shape (n, size): by default the unknowns themselves."""
        return np.asarray(z)

    def peak_of(self, z: np.ndarray) -> np.ndarray | None:
        """For targets with separate peaks: the index of the peak each draw belongs to."""
        return None


# ---------------------------------------------------------------------------------------------------------------------------
# Densities with a normal prior and the likelihood p / prior * exp(c)


class DensityTarget(Target):
    c: float

    def _set_prior(self, mean: np.ndarray, sd: np.ndarray):
        self.prior_sd = jnp.asarray(np.abs(mean) + 5.0 * sd)

    def log_density(self, z):
        raise NotImplementedError

    def exact_draws(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError

    def log_prior(self, z):
        return jnp.sum(norm.logpdf(z, 0.0, self.prior_sd))

    def log_likelihood(self, z):
        return self.log_density(z) - self.log_prior(z) + self.c

    def draw_prior(self, key):
        return self.prior_sd * jax.random.normal(key, (self.size,))


def _normal_cdfs(mean, sd):
    return tuple((lambda x, m=m, s=s: stats.norm.cdf(x, m, s)) for m, s in zip(mean, sd))


class Gaussian(DensityTarget):
    def __init__(self, name, group, description, mean, cov, c, source):
        self.name, self.group, self.description, self.c = name, group, description, c
        self.size = len(mean)
        self.m, self.cov = np.asarray(mean, float), np.asarray(cov, float)
        self.chol = np.linalg.cholesky(self.cov)
        self._L, self._m = jnp.asarray(self.chol), jnp.asarray(self.m)
        self._log_det = float(np.sum(np.log(np.diag(self.chol))))
        sd = np.sqrt(np.diag(self.cov))
        self._set_prior(self.m, sd)
        self.reference = Reference(tuple(f"z[{i}]" for i in range(self.size)), self.m, sd, _normal_cdfs(self.m, sd), c, source)

    def log_density(self, z):
        u = jax.scipy.linalg.solve_triangular(self._L, z - self._m, lower=True)
        return -0.5 * jnp.sum(u**2) - self._log_det - 0.5 * self.size * jnp.log(2 * jnp.pi)

    def exact_draws(self, rng, n):
        return self.m + rng.standard_normal((n, self.size)) @ self.chol.T


def correlated_normal(d: int, condition: float) -> Gaussian:
    rng = np.random.default_rng(1000 + d)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    variances = np.logspace(0.0, -np.log10(condition), d)
    cov = q @ np.diag(variances) @ q.T
    cov = (cov + cov.T) / 2
    return Gaussian(f"correlated_normal_d{d}_c{int(condition)}", "Correlated normal",
                    f"Normal posterior in {d} dimensions, randomly rotated, covariance condition number {condition:.0e}",
                    rng.standard_normal(d), cov, -2.0 - 0.1 * d, "exact (normal)")


def varied_scales(d: int = 20) -> Gaussian:
    sd = np.logspace(-3.0, 2.0, d)
    sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    return Gaussian(f"scales_d{d}", "Widely varying scales",
                    f"Independent normal posterior in {d} dimensions with standard deviations from 1e-3 to 1e2",
                    2.0 * sd * sign, np.diag(sd**2), -1.5, "exact (normal)")


class TwoPeaks(DensityTarget):
    """Two normal peaks: weight 0.3 with sd 1 and weight 0.7 with sd 0.3, centres 10 apart along the diagonal."""

    weights, sds, separation = (0.3, 0.7), (1.0, 0.3), 10.0

    def __init__(self, d: int):
        self.name, self.group, self.size, self.c = f"two_peaks_d{d}", "Two separated peaks", d, -3.0
        self.description = (f"Two normal peaks in {d} dimensions, weights 0.3 and 0.7, standard deviations 1 and 0.3, "
                            f"centres {self.separation:g} apart")
        u = np.ones(d) / np.sqrt(d)
        self.centres = (-self.separation / 2 * u, self.separation / 2 * u)
        self._centres = jnp.asarray(np.stack(self.centres))
        mean = sum(w * c for w, c in zip(self.weights, self.centres))
        second = sum(w * (s**2 + c**2) for w, s, c in zip(self.weights, self.sds, self.centres))
        sd = np.sqrt(second - mean**2)
        self._set_prior(mean, sd)
        cdfs = tuple((lambda x, i=i: sum(w * stats.norm.cdf(x, c[i], s) for w, s, c in zip(self.weights, self.sds, self.centres)))
                     for i in range(d))
        self.reference = Reference(tuple(f"z[{i}]" for i in range(d)), mean, sd, cdfs, self.c, "exact (mixture of normals)",
                                   peak_weights=self.weights)

    def _component_log_densities(self, z):
        return jnp.stack([jnp.log(w) + jnp.sum(norm.logpdf(z, self._centres[k], s)) for k, (w, s) in enumerate(zip(self.weights, self.sds))])

    def log_density(self, z):
        return jax.scipy.special.logsumexp(self._component_log_densities(z))

    def peak_of(self, z):
        parts = np.asarray(jax.vmap(self._component_log_densities)(jnp.asarray(z)))
        return np.argmax(parts, axis=1)

    def exact_draws(self, rng, n):
        k = rng.choice(2, size=n, p=self.weights)
        return np.stack(self.centres)[k] + np.asarray(self.sds)[k][:, None] * rng.standard_normal((n, self.size))


def _gauss_hermite(n: int = 200):
    """Nodes and weights for expectations over a standard normal."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


class Banana(DensityTarget):
    """Haario's twisted normal: x ~ N(0, diag(100, 1, ..., 1)), y2 = x2 + b (x1^2 - 100). The twist preserves volume."""

    def __init__(self, d: int, b: float, label: str):
        self.name, self.group, self.size, self.b, self.c = f"banana_d{d}_{label}", "Banana", d, b, -1.0
        self.description = f"Twisted normal ('banana') in {d} dimensions, twist b = {b:g}"
        mean = np.zeros(d)
        sd = np.ones(d)
        sd[0], sd[1] = 10.0, np.sqrt(1.0 + 2e4 * b**2)
        self._set_prior(mean, sd)
        # y2's distribution: P(y2 <= t) = E over u ~ N(0, 1) of Phi(t - b (100 u^2 - 100)), with x1 = 10 u. The integrand is
        # smooth, so the trapezoid rule on a fine grid in u is accurate once its steepest part is resolved. Tabulated over t
        # (finely where the mass is) the first time it is needed, then interpolated.
        table = {}

        def cdf_y2(x):
            if not table:
                u = np.linspace(-8.5, 8.5, 6801)
                wu = stats.norm.pdf(u) * (u[1] - u[0])
                t = np.unique(np.concatenate([np.linspace(-100.0 * b - 9.0, 100.0 * b + 60.0, 4001),
                                              np.linspace(100.0 * b + 60.0, b * ((10.0 * 8.5) ** 2 - 100.0) + 9.0, 1001)]))
                f = np.concatenate([np.sum(wu * stats.norm.cdf(t[i:i + 250, None] - b * (100.0 * u**2 - 100.0)), axis=1)
                                    for i in range(0, len(t), 250)])
                table.update(t=t, f=f)
            return np.interp(np.asarray(x, float), table["t"], table["f"], left=0.0, right=1.0)

        cdfs = [lambda x: stats.norm.cdf(x, 0.0, 10.0), cdf_y2] + [lambda x: stats.norm.cdf(x)] * (d - 2)
        self.reference = Reference(tuple(f"z[{i}]" for i in range(d)), mean, sd, tuple(cdfs), self.c,
                                   "exact (moments in closed form; the twisted coordinate's distribution by numerical integration)")

    def log_density(self, z):
        x2 = z[1] - self.b * (z[0] ** 2 - 100.0)
        return norm.logpdf(z[0], 0.0, 10.0) + norm.logpdf(x2) + jnp.sum(norm.logpdf(z[2:]))

    def exact_draws(self, rng, n):
        x = rng.standard_normal((n, self.size))
        x[:, 0] *= 10.0
        x[:, 1] += self.b * (x[:, 0] ** 2 - 100.0)
        return x


class Funnel(DensityTarget):
    """Neal's funnel: v ~ N(0, 3^2) and each x_i ~ N(0, e^v)."""

    def __init__(self, d: int = 10):
        self.name, self.group, self.size, self.c = f"funnel_d{d}", "Funnel", d, -2.0
        self.description = f"Neal's funnel in {d} dimensions: v ~ N(0, 9), x_i ~ N(0, exp(v))"
        mean = np.zeros(d)
        sd = np.full(d, np.exp(9.0 / 4.0))
        sd[0] = 3.0
        self._set_prior(mean, sd)
        nodes, weights = _gauss_hermite()

        def cdf_x(x):
            x = np.asarray(x, float)
            return np.sum(weights * stats.norm.cdf(x[..., None] * np.exp(-3.0 * nodes / 2.0)), axis=-1)

        cdfs = [lambda x: stats.norm.cdf(x, 0.0, 3.0)] + [cdf_x] * (d - 1)
        self.reference = Reference(tuple(["v"] + [f"x[{i}]" for i in range(1, d)]), mean, sd, tuple(cdfs), self.c,
                                   "exact (moments in closed form; each x's distribution by 200-point Gauss-Hermite)")

    def log_density(self, z):
        return norm.logpdf(z[0], 0.0, 3.0) + jnp.sum(norm.logpdf(z[1:], 0.0, jnp.exp(z[0] / 2.0)))

    def exact_draws(self, rng, n):
        v = 3.0 * rng.standard_normal(n)
        return np.column_stack([v, np.exp(v / 2.0)[:, None] * rng.standard_normal((n, self.size - 1))])


class StudentT(DensityTarget):
    """Multivariate Student-t with 5 degrees of freedom and a correlated scale matrix."""

    nu = 5.0

    def __init__(self, d: int):
        self.name, self.group, self.size, self.c = f"student_t_d{d}", "Heavy tails (Student-t)", d, -2.5
        self.description = f"Multivariate Student-t in {d} dimensions, 5 degrees of freedom, correlated scale matrix"
        rng = np.random.default_rng(2000 + d)
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        cov = q @ np.diag(np.logspace(0.0, -1.0, d)) @ q.T
        s = np.sqrt(np.diag(cov))
        self.scale = cov / np.outer(s, s)   # a correlation matrix: unit marginal scales
        self.scale = (self.scale + self.scale.T) / 2
        self.loc = rng.standard_normal(d)
        self.chol = np.linalg.cholesky(self.scale)
        self._L, self._loc = jnp.asarray(self.chol), jnp.asarray(self.loc)
        self._log_det = float(np.sum(np.log(np.diag(self.chol))))
        marginal_scale = np.sqrt(np.diag(self.scale))
        sd = marginal_scale * np.sqrt(self.nu / (self.nu - 2.0))
        self._set_prior(self.loc, sd)
        cdfs = tuple((lambda x, m=m, s=s: stats.t.cdf((np.asarray(x) - m) / s, self.nu)) for m, s in zip(self.loc, marginal_scale))
        self.reference = Reference(tuple(f"z[{i}]" for i in range(d)), self.loc, sd, cdfs, self.c, "exact (Student-t)")

    def log_density(self, z):
        u = jax.scipy.linalg.solve_triangular(self._L, z - self._loc, lower=True)
        nu, d = self.nu, self.size
        return (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * jnp.log(nu * jnp.pi) - self._log_det
                - (nu + d) / 2 * jnp.log1p(jnp.sum(u**2) / nu))

    def exact_draws(self, rng, n):
        g = rng.standard_normal((n, self.size)) @ self.chol.T
        return self.loc + g / np.sqrt(rng.chisquare(self.nu, n) / self.nu)[:, None]


# ---------------------------------------------------------------------------------------------------------------------------
# Independent normal means: tests.infer.targets.NormalMean, with its reference


class NormalMeans(Target):
    group = "Independent normal means"

    def __init__(self, d: int):
        rng = np.random.default_rng(d)
        self.inner = NormalMean(rng.normal(size=d) + rng.normal(size=(50, d)))
        self.name, self.size = f"normal_means_d{d}", d
        self.description = f"{d} unknown means, prior N(0, 10^2), 50 observations each with sd 1"
        sd = np.full(d, np.sqrt(self.inner.post_var))
        self.reference = Reference(tuple(f"z[{i}]" for i in range(d)), self.inner.post_mean, sd, _normal_cdfs(self.inner.post_mean, sd),
                                   self.inner.log_evidence, "exact (conjugate normal)")

    def log_prior(self, z):
        return self.inner.log_prior(z)

    def log_likelihood(self, z):
        return self.inner.log_likelihood(z)

    def draw_prior(self, key):
        return self.inner.draw_prior(key)


# ---------------------------------------------------------------------------------------------------------------------------
# Conjugate multivariate regression (the form of a Bayesian vector autoregression)


class ConjugateRegression(Target):
    """Y = X B + E, each row of E ~ N(0, Sigma), with the normal-inverse-Wishart prior Sigma ~ IW(S0, nu0) and
    B | Sigma ~ MN(B0, Omega0, Sigma). X holds a constant and `lags` lags of Y, simulated from a stable process.

    Unknowns: B (row by row), then the Cholesky factor of Sigma by rows of its lower triangle with the diagonal as logs.
    Posterior: Sigma ~ IW(S_T, nu_T), B | Sigma ~ MN(B_T, Omega_T, Sigma); each coefficient is Student-t with nu_T - n + 1
    degrees of freedom and each variance inverse-gamma. The evidence is in closed form.
    """

    group = "Conjugate multivariate regression"

    def __init__(self, n: int, lags: int, periods: int = 150):
        self.n, self.lags, self.periods = n, lags, periods
        self.k = 1 + n * lags
        self.name = f"conjugate_regression_n{n}_p{lags}"
        self.description = (f"{n} series regressed on a constant and {lags} lag(s) of all of them ({periods} periods), "
                            f"normal-inverse-Wishart prior")
        rng = np.random.default_rng(3000 + 10 * n + lags)
        Y, X = self._simulate(rng)
        self.Y, self.X = Y, X
        self.nu0 = n + 3.0
        self.S0 = np.eye(n) * (self.nu0 - n - 1)
        self.B0 = np.zeros((self.k, n))
        self.Omega0 = np.eye(self.k) * 10.0**2
        self.n_chol = n * (n + 1) // 2
        self.size = self.k * n + self.n_chol
        self.rows, self.cols = np.tril_indices(n)
        self._diagonal_positions = np.flatnonzero(self.rows == self.cols)
        self._jY, self._jX = jnp.asarray(Y), jnp.asarray(X)
        self._posterior()
        self.reference = self._reference()

    def _simulate(self, rng):
        n, p = self.n, self.lags
        while True:
            A = [0.5 * np.eye(n) + 0.1 * rng.standard_normal((n, n))] + [0.1 / (j + 1) * rng.standard_normal((n, n)) for j in range(1, p)]
            companion = np.zeros((n * p, n * p))
            companion[:n] = np.hstack(A)
            companion[n:, :-n] = np.eye(n * (p - 1))
            if np.max(np.abs(np.linalg.eigvals(companion))) < 0.9:
                break
        c = rng.normal(0.0, 0.5, n)
        G = np.eye(n) + 0.3 * rng.standard_normal((n, n))
        sigma = 0.5 * G @ G.T
        chol = np.linalg.cholesky(sigma)
        burn = 100
        y = np.zeros((self.periods + p + burn, n))
        for t in range(p, len(y)):
            y[t] = c + sum(A[j] @ y[t - 1 - j] for j in range(p)) + chol @ rng.standard_normal(n)
        y = y[burn:]
        Y = y[p:]
        X = np.column_stack([np.ones(len(Y))] + [y[p - 1 - j:len(y) - 1 - j] for j in range(p)])
        return Y, X

    def _posterior(self):
        O0i = np.linalg.inv(self.Omega0)
        self.OmegaT = np.linalg.inv(O0i + self.X.T @ self.X)
        self.BT = self.OmegaT @ (O0i @ self.B0 + self.X.T @ self.Y)
        self.ST = self.S0 + self.Y.T @ self.Y + self.B0.T @ O0i @ self.B0 - self.BT.T @ np.linalg.inv(self.OmegaT) @ self.BT
        self.ST = (self.ST + self.ST.T) / 2
        self.nuT = self.nu0 + self.periods
        n, T = self.n, self.periods
        log_det = lambda a: np.linalg.slogdet(a)[1]
        self.log_evidence = float(-T * n / 2 * np.log(np.pi) + n / 2 * (log_det(self.OmegaT) - log_det(self.Omega0))
                                  + self.nu0 / 2 * log_det(self.S0) - self.nuT / 2 * log_det(self.ST)
                                  + multigammaln(self.nuT / 2, n) - multigammaln(self.nu0 / 2, n))

    def _reference(self):
        n, k, nu = self.n, self.k, self.nuT
        names, mean, sd, cdfs = [], [], [], []
        dof = nu - n + 1
        for i in range(k):
            for j in range(n):
                scale = np.sqrt(self.OmegaT[i, i] * self.ST[j, j] / dof)
                names.append(f"B[{i},{j}]")
                mean.append(self.BT[i, j])
                sd.append(scale * np.sqrt(dof / (dof - 2)))
                cdfs.append(lambda x, m=self.BT[i, j], s=scale: stats.t.cdf((np.asarray(x) - m) / s, dof))
        for i, j in zip(self.rows, self.cols):
            S = self.ST
            names.append(f"Sigma[{i},{j}]")
            mean.append(S[i, j] / (nu - n - 1))
            var = ((nu - n + 1) * S[i, j] ** 2 + (nu - n - 1) * S[i, i] * S[j, j]) / ((nu - n) * (nu - n - 1) ** 2 * (nu - n - 3))
            sd.append(np.sqrt(var))
            cdfs.append((lambda x, a=(nu - n + 1) / 2, b=S[i, i] / 2: stats.invgamma.cdf(x, a, scale=b)) if i == j else None)
        return Reference(tuple(names), np.array(mean), np.array(sd), tuple(cdfs), self.log_evidence,
                         "exact (normal-inverse-Wishart: Student-t coefficients, inverse-gamma variances, closed-form evidence)")

    # --- the map from the unknowns to (B, Sigma) ---

    def unpack(self, z):
        B = jnp.reshape(z[:self.k * self.n], (self.k, self.n))
        w = z[self.k * self.n:]
        L = jnp.zeros((self.n, self.n)).at[self.rows, self.cols].set(w)
        diagonal = jnp.diag(L)
        L = L - jnp.diag(diagonal) + jnp.diag(jnp.exp(diagonal))
        return B, L

    def pack(self, B, sigma):
        L = np.linalg.cholesky(sigma)
        w = L[self.rows, self.cols].copy()
        w[self.rows == self.cols] = np.log(np.diag(L))
        return np.concatenate([np.ravel(B), w])

    def log_jacobian(self, z):
        """log |d vech(Sigma) / dz|: 2^n prod L_ii^(n - i + 1) for Sigma = L L', times prod L_ii for L_ii = exp(w_ii)."""
        _, L = self.unpack(z)
        i = jnp.arange(1, self.n + 1)
        return self.n * jnp.log(2.0) + jnp.sum((self.n - i + 2) * jnp.log(jnp.diag(L)))

    def _log_iw(self, sigma_chol, S, nu):
        n = self.n
        inv = jax.scipy.linalg.cho_solve((sigma_chol, True), jnp.eye(n))
        log_det_sigma = 2 * jnp.sum(jnp.log(jnp.diag(sigma_chol)))
        return (nu / 2 * jnp.linalg.slogdet(S)[1] - nu * n / 2 * jnp.log(2.0) - multigammaln(nu / 2, n)
                - (nu + n + 1) / 2 * log_det_sigma - 0.5 * jnp.trace(S @ inv))

    def _log_mn(self, B, M, Omega, sigma_chol):
        k, n = self.k, self.n
        inv_sigma = jax.scipy.linalg.cho_solve((sigma_chol, True), jnp.eye(n))
        D = B - M
        log_det_sigma = 2 * jnp.sum(jnp.log(jnp.diag(sigma_chol)))
        return (-k * n / 2 * jnp.log(2 * jnp.pi) - n / 2 * jnp.linalg.slogdet(Omega)[1] - k / 2 * log_det_sigma
                - 0.5 * jnp.trace(inv_sigma @ D.T @ jnp.linalg.solve(Omega, D)))

    def log_prior_values(self, B, L):
        return self._log_iw(L, jnp.asarray(self.S0), self.nu0) + self._log_mn(B, jnp.asarray(self.B0), jnp.asarray(self.Omega0), L)

    def log_posterior_values(self, B, L):
        return self._log_iw(L, jnp.asarray(self.ST), self.nuT) + self._log_mn(B, jnp.asarray(self.BT), jnp.asarray(self.OmegaT), L)

    def log_likelihood_values(self, B, L):
        E = self._jY - self._jX @ B
        U = jax.scipy.linalg.solve_triangular(L, E.T, lower=True)
        return -self.periods * self.n / 2 * jnp.log(2 * jnp.pi) - self.periods * jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * jnp.sum(U**2)

    def log_prior(self, z):
        B, L = self.unpack(z)
        return self.log_prior_values(B, L) + self.log_jacobian(z)

    def log_likelihood(self, z):
        B, L = self.unpack(z)
        return self.log_likelihood_values(B, L)

    def _draw(self, key, S, nu, M, Omega):
        """Sigma ~ IW(S, nu) by inverting a Bartlett-decomposed Wishart(S^-1, nu) draw, then B | Sigma ~ MN(M, Omega, Sigma)."""
        n, k = self.n, self.k
        k1, k2, k3 = jax.random.split(key, 3)
        chi = 2.0 * jax.random.gamma(k1, (nu - jnp.arange(n)) / 2.0)
        A = jnp.tril(jax.random.normal(k2, (n, n)), -1) + jnp.diag(jnp.sqrt(chi))
        C = jnp.linalg.cholesky(jnp.linalg.inv(S))
        W = C @ A @ A.T @ C.T
        sigma = jnp.linalg.inv(W)
        sigma = (sigma + sigma.T) / 2
        L = jnp.linalg.cholesky(sigma)
        B = M + jnp.linalg.cholesky(Omega) @ jax.random.normal(k3, (k, n)) @ L.T
        w = L[self.rows, self.cols].at[self._diagonal_positions].set(jnp.log(jnp.diag(L)))
        return jnp.concatenate([jnp.ravel(B), w])

    def draw_prior(self, key):
        return self._draw(key, jnp.asarray(self.S0), self.nu0, jnp.asarray(self.B0), jnp.asarray(self.Omega0))

    def draw_posterior(self, key):
        return self._draw(key, jnp.asarray(self.ST), self.nuT, jnp.asarray(self.BT), jnp.asarray(self.OmegaT))

    @cached_property
    def _quantities(self):
        def one(z):
            B, L = self.unpack(z)
            sigma = L @ L.T
            return jnp.concatenate([jnp.ravel(B), sigma[self.rows, self.cols]])
        return jax.jit(jax.vmap(one))

    def quantities(self, z):
        return np.asarray(self._quantities(jnp.asarray(z)))


# ---------------------------------------------------------------------------------------------------------------------------
# mutatis toy models: references by numerical integration


REFERENCE_CACHE = Path(__file__).resolve().parent / "results" / "_references"
REFERENCE_VERSION = 2   # 2: points spaced by sinh along each axis (benchmarks/samplers/quadrature.py)


class ModelTarget(Target):
    """A mutatis toy model as a posterior problem, checked on its parameters against a grid integration of the posterior in
    which the shocks are averaged out exactly (the Kalman route)."""

    grid_points: int

    def __init__(self, name, group, description, problem, kalman_problem, names):
        self.name, self.group, self.description = name, group, description
        self.problem, self.kalman_problem, self.names = problem, kalman_problem, tuple(names)
        self.size = problem.size
        self.proper, self.gradient_ready = problem.proper, problem.gradient_ready

    def log_prior(self, z):
        return self.problem.log_prior(z)

    def log_likelihood(self, z):
        return self.problem.log_likelihood(z)

    def draw_prior(self, key):
        return self.problem.draw_prior(key)

    @cached_property
    def _quantities(self):
        return jax.jit(jax.vmap(lambda z: jnp.stack([jnp.reshape(self.problem.parameters(z)[k], ()) for k in self.names])))

    def quantities(self, z):
        return np.asarray(self._quantities(jnp.asarray(z)))

    def grid(self):
        """The grid integration of the Kalman-route posterior, cached on disk (the data are simulated from fixed seeds)."""
        path = REFERENCE_CACHE / f"{self.name}-v{REFERENCE_VERSION}-n{self.grid_points}.npz"
        if path.exists():
            saved = np.load(path, allow_pickle=True)
            return {k: saved[k] for k in saved.files}
        kp = self.kalman_problem
        order = [next(u.name for u in kp.layout.unknowns if u.offset == k) for k in range(kp.size)]
        if sorted(order) != sorted(self.names):
            raise ValueError(f"the Kalman problem's unknowns {order} are not the checked parameters {self.names}")
        starts = np.stack([np.asarray(kp.draw_prior(k)) for k in jax.random.split(jax.random.key(0), 8)])

        def constrain(k, values):   # each parameter is a scalar with its own transform, so the other coordinates do not matter
            z = np.tile(starts[0], (len(values), 1))
            z[:, k] = values
            return np.asarray(jax.vmap(lambda zz: jnp.reshape(kp.parameters(zz)[order[k]], ()))(jnp.asarray(z)))

        g = grid_posterior(kp.log_posterior, constrain, starts, self.grid_points)
        result = {"order": np.array(order), "log_evidence": g.log_evidence, "mean": g.mean, "sd": g.sd, "width": g.width,
                  "edge_log_ratio": g.edge_log_ratio}
        for k in range(len(order)):
            result[f"nodes{k}"], result[f"cumulative{k}"] = g.nodes[k], g.cumulative[k]
        REFERENCE_CACHE.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.npz")
        np.savez(partial, **result)
        partial.replace(path)   # whole or not at all, for runs reading the cache in parallel
        return result

    @cached_property
    def reference(self):
        g = self.grid()
        order = [str(x) for x in g["order"]]
        index = [order.index(name) for name in self.names]
        cdfs = tuple((lambda x, k=k: np.interp(np.asarray(x, float), g[f"nodes{k}"], g[f"cumulative{k}"], left=0.0, right=1.0)) for k in index)
        return Reference(self.names, g["mean"][index], g["sd"][index], cdfs, float(g["log_evidence"]),
                         f"grid integration of the posterior with the shocks averaged out exactly (Kalman filter), "
                         f"{self.grid_points} points per axis, sinh-spaced, reaching ±{float(g['width']):.0f} sds")


def _toy(name: str, periods: int):
    from tests.infer.test_routes import setup  # the toy models and their simulated data, as the route tests build them
    return setup(name, n_periods=periods, seed=0)


def hidden_level_shocks(periods: int) -> ModelTarget:
    from mutatis.infer.problem import build_problem
    _, system, _, solve, data = _toy("hidden_level", periods)
    target = ModelTarget(f"hidden_level_shocks_T{periods}", "State space, shocks sampled",
                         f"A drifting level seen with noise, {periods} periods, every shock and the starting value sampled "
                         f"({periods + 3} unknowns); checked on the two standard deviations",
                         build_problem(system, solve, data, route="sample", start="model"),
                         build_problem(system, solve, data, route="kalman", start="model"), ["level.sigma_eta", "y.sigma_eps"])
    target.grid_points = 401
    return target


def forward_looking(periods: int = 150) -> ModelTarget:
    from mutatis.infer.problem import build_problem
    _, system, _, solve, data = _toy("forward_looking", periods)
    problem = build_problem(system, solve, data)
    target = ModelTarget("forward_looking", "Forward-looking model",
                         f"Price set by expectations of next quarter's price, demand AR(1), {periods} periods, "
                         f"shocks averaged out by the Kalman filter (3 unknowns)", problem, problem,
                         ["demand.rho", "demand.sigma_x", "price.a"])
    target.grid_points = 81
    return target


# ---------------------------------------------------------------------------------------------------------------------------

TARGETS: dict[str, Callable[[], Target]] = {}
for _d in (2, 10, 50, 100, 200, 500):
    TARGETS[f"normal_means_d{_d}"] = lambda d=_d: NormalMeans(d)
for _d in (10, 50, 100):
    for _c in (100, 10000):
        TARGETS[f"correlated_normal_d{_d}_c{_c}"] = lambda d=_d, c=_c: correlated_normal(d, float(c))
TARGETS["scales_d20"] = lambda: varied_scales(20)
for _d in (1, 5, 20, 50):
    TARGETS[f"two_peaks_d{_d}"] = lambda d=_d: TwoPeaks(d)
TARGETS["banana_d2_strong"] = lambda: Banana(2, 0.1, "strong")
TARGETS["banana_d10_mild"] = lambda: Banana(10, 0.03, "mild")
TARGETS["banana_d50_mild"] = lambda: Banana(50, 0.03, "mild")
for _d in (10, 20):
    TARGETS[f"funnel_d{_d}"] = lambda d=_d: Funnel(d)
for _d in (10, 50, 100):
    TARGETS[f"student_t_d{_d}"] = lambda d=_d: StudentT(d)
for _n, _p in ((2, 1), (2, 4), (4, 1), (4, 4), (6, 2)):
    TARGETS[f"conjugate_regression_n{_n}_p{_p}"] = lambda n=_n, p=_p: ConjugateRegression(n, p)
for _t in (50, 200, 400):
    TARGETS[f"hidden_level_shocks_T{_t}"] = lambda t=_t: hidden_level_shocks(t)
TARGETS["forward_looking"] = forward_looking


def build(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"no target {name!r}; known: {', '.join(TARGETS)}")
    return TARGETS[name]()
