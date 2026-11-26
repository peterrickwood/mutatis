"""Probability distributions for priors, shocks and measurement errors.

Each distribution gives the log of its density, draws samples, and names its domain. Parameters may be JAX arrays, including
values that depend on other unknowns (a shock's size is often a parameter of the model).

Distributions over single numbers act elementwise: `log_prob` returns one value per element, and whoever assembles a model
adds them up. Distributions over vectors or matrices (`event_shape` not empty) return one value per vector or matrix.

`Flat` is not a probability distribution: it has no total probability and cannot be sampled. It stands for the limit of a
wider and wider prior, and is only usable where the answer is known exactly without sampling (README, "Flat" needs care).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy import special, stats
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray
from jaxtyping import Real as RealNumber

from mutatis import _jax_config  # noqa: F401
from mutatis.dist.domains import Domain, Interval, Orthogonal, Positive, PositiveDefinite, Real

Number = RealNumber[ArrayLike, "..."]  # a number or array of numbers: Python, NumPy or JAX, integer or float


class Distribution(eqx.Module):
    """A probability distribution."""

    proper: eqx.AbstractClassVar[bool]

    @property
    def domain(self) -> Domain:
        raise NotImplementedError

    @property
    def event_shape(self) -> tuple[int, ...]:
        return ()

    def log_prob(self, x: Number) -> Float[Array, "..."]:
        raise NotImplementedError

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()) -> Float[Array, "..."]:
        raise NotImplementedError


def _outside(log_density, inside):
    return jnp.where(inside, log_density, -jnp.inf)


class Normal(Distribution):
    loc: Number = 0.0
    scale: Number = 1.0
    proper = True

    @property
    def domain(self):
        return Real()

    def log_prob(self, x):
        return stats.norm.logpdf(x, self.loc, self.scale)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + jnp.broadcast_shapes(jnp.shape(self.loc), jnp.shape(self.scale))
        return self.loc + self.scale * jax.random.normal(key, shape)


class HalfNormal(Distribution):
    """The absolute value of a normal with mean zero."""

    scale: Number = 1.0
    proper = True

    @property
    def domain(self):
        return Positive()

    def log_prob(self, x):
        return _outside(jnp.log(2.0) + stats.norm.logpdf(x, 0.0, self.scale), x >= 0)

    def sample(self, key, sample_shape=()):
        return jnp.abs(self.scale * jax.random.normal(key, sample_shape + jnp.shape(self.scale)))


class Gamma(Distribution):
    """Gamma with shape `a` and rate `rate` (mean a / rate)."""

    a: Number
    rate: Number
    proper = True

    @classmethod
    def from_mean_sd(cls, mean, sd) -> "Gamma":
        return cls(a=(mean / sd) ** 2, rate=mean / sd**2)

    @property
    def domain(self):
        return Positive()

    def log_prob(self, x):
        return _outside(stats.gamma.logpdf(x, self.a, scale=1.0 / self.rate), x > 0)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + jnp.broadcast_shapes(jnp.shape(self.a), jnp.shape(self.rate))
        return jax.random.gamma(key, self.a, shape) / self.rate


class Beta(Distribution):
    a: Number
    b: Number
    proper = True

    @classmethod
    def from_mean_sd(cls, mean, sd) -> "Beta":
        k = mean * (1.0 - mean) / sd**2 - 1.0
        return cls(a=mean * k, b=(1.0 - mean) * k)

    @property
    def domain(self):
        return Interval(0.0, 1.0)

    def log_prob(self, x):
        return _outside(stats.beta.logpdf(x, self.a, self.b), (x > 0) & (x < 1))

    def sample(self, key, sample_shape=()):
        shape = sample_shape + jnp.broadcast_shapes(jnp.shape(self.a), jnp.shape(self.b))
        return jax.random.beta(key, self.a, self.b, shape)


class InverseGamma(Distribution):
    """Inverse gamma with shape `a` and scale `scale`: if y ~ Gamma(a, rate=scale) then 1 / y ~ InverseGamma(a, scale)."""

    a: Number
    scale: Number
    proper = True

    @classmethod
    def from_mean_sd(cls, mean, sd) -> "InverseGamma":
        a = 2.0 + (mean / sd) ** 2
        return cls(a=a, scale=mean * (a - 1.0))

    @property
    def domain(self):
        return Positive()

    def log_prob(self, x):
        safe = jnp.where(x > 0, x, 1.0)
        log_density = self.a * jnp.log(self.scale) - special.gammaln(self.a) - (self.a + 1.0) * jnp.log(safe) - self.scale / safe
        return _outside(log_density, x > 0)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + jnp.broadcast_shapes(jnp.shape(self.a), jnp.shape(self.scale))
        return self.scale / jax.random.gamma(key, self.a, shape)


class StudentT(Distribution):
    df: Number
    loc: Number = 0.0
    scale: Number = 1.0
    proper = True

    @property
    def domain(self):
        return Real()

    def log_prob(self, x):
        return stats.t.logpdf(x, self.df, self.loc, self.scale)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + jnp.broadcast_shapes(jnp.shape(self.df), jnp.shape(self.loc), jnp.shape(self.scale))
        return self.loc + self.scale * jax.random.t(key, self.df, shape)


class Uniform(Distribution):
    low: Number = 0.0
    high: Number = 1.0
    proper = True

    @property
    def domain(self):
        return Interval(self.low, self.high)

    def log_prob(self, x):
        return _outside(-jnp.log(self.high - self.low) + jnp.zeros_like(x), (x > self.low) & (x < self.high))

    def sample(self, key, sample_shape=()):
        shape = sample_shape + jnp.broadcast_shapes(jnp.shape(self.low), jnp.shape(self.high))
        return jax.random.uniform(key, shape, minval=self.low, maxval=self.high)


class MultivariateNormal(Distribution):
    mean: Float[ArrayLike, " n"]
    cov: Float[ArrayLike, "n n"]
    proper = True

    @property
    def domain(self):
        return Real()

    @property
    def event_shape(self):
        return (self.mean.shape[-1],)

    def log_prob(self, x):
        return stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def sample(self, key, sample_shape=()):
        L = jnp.linalg.cholesky(self.cov)
        z = jax.random.normal(key, sample_shape + self.event_shape)
        return self.mean + z @ L.T


class InverseWishart(Distribution):
    """Inverse Wishart over n-by-n covariance matrices, with `df` degrees of freedom and scale matrix `scale` (mean scale / (df - n - 1))."""

    df: Number
    scale: Float[ArrayLike, "n n"]
    proper = True

    @property
    def domain(self):
        return PositiveDefinite()

    @property
    def event_shape(self):
        return self.scale.shape[-2:]

    def log_prob(self, x):
        n = self.scale.shape[-1]
        _, logdet_scale = jnp.linalg.slogdet(self.scale)
        sign_x, logdet_x = jnp.linalg.slogdet(x)
        trace = jnp.trace(jnp.linalg.solve(x, self.scale), axis1=-2, axis2=-1)
        log_density = (0.5 * self.df * logdet_scale - 0.5 * self.df * n * jnp.log(2.0) - special.multigammaln(0.5 * self.df, n)
                       - 0.5 * (self.df + n + 1) * logdet_x - 0.5 * trace)
        return _outside(log_density, sign_x > 0)

    def sample(self, key, sample_shape=()):
        # If x ~ InverseWishart(df, S) then inv(x) ~ Wishart(df, inv(S)). Wishart draws use the Bartlett decomposition.
        n = self.scale.shape[-1]
        L = jnp.linalg.cholesky(jnp.linalg.inv(self.scale))
        key_diag, key_below = jax.random.split(key)
        chi2 = jax.random.chisquare(key_diag, self.df - jnp.arange(n), sample_shape + (n,))
        below = jnp.tril(jax.random.normal(key_below, sample_shape + (n, n)), k=-1)
        A = below + jnp.sqrt(chi2)[..., :, None] * jnp.eye(n)
        LA = L @ A
        return jnp.linalg.inv(LA @ jnp.swapaxes(LA, -1, -2))


class UniformOrthogonal(Distribution):
    """The even ("Haar") distribution over n-by-n orthogonal matrices. Its density is constant, so `log_prob` is 0 on the domain."""

    n: int = eqx.field(static=True)
    proper = True

    @property
    def domain(self):
        return Orthogonal()

    @property
    def event_shape(self):
        return (self.n, self.n)

    def log_prob(self, x):
        return _outside(jnp.zeros(x.shape[:-2]), self.domain.contains(x))

    def sample(self, key, sample_shape=()):
        # QR of a matrix of independent standard normals, with the signs fixed so that R has a positive diagonal (Mezzadri 2007).
        A = jax.random.normal(key, sample_shape + (self.n, self.n))
        Q, R = jnp.linalg.qr(A)
        return Q * jnp.sign(jnp.diagonal(R, axis1=-2, axis2=-1))[..., None, :]


class Flat(Distribution):
    """Equal weight everywhere on a domain. Not a probability distribution: see the module docstring."""

    on: Domain = Real()
    proper = False

    @property
    def domain(self):
        return self.on

    def log_prob(self, x):
        return _outside(jnp.zeros_like(x), self.on.contains(x))

    def sample(self, key, sample_shape=()):
        raise ValueError("a flat prior has no total probability and cannot be sampled")


class JeffreysCovariance(Distribution):
    """The conventional uninformative prior for an n-by-n covariance matrix, with density proportional to det(x)^(-(n+1)/2).

    Not a probability distribution (no total probability, cannot be sampled), like `Flat`.
    """

    n: int = eqx.field(static=True)
    proper = False

    @property
    def domain(self):
        return PositiveDefinite()

    @property
    def event_shape(self):
        return (self.n, self.n)

    def log_prob(self, x):
        sign, logdet = jnp.linalg.slogdet(x)
        return _outside(-0.5 * (self.n + 1) * logdet, sign > 0)

    def sample(self, key, sample_shape=()):
        raise ValueError("the Jeffreys prior has no total probability and cannot be sampled")
