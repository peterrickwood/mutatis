"""Distributions: densities against SciPy, parameterisations by mean and standard deviation, and samples against their moments.

Sample checks compare a sample mean with the true mean. The allowed gap is five Monte Carlo standard errors, estimated from
the sample itself, so a correct implementation fails about once in 1.7 million runs per check.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from tests.helpers import assert_covariance_matches, assert_mean_matches
from mutatis.dist import (Beta, Flat, Gamma, HalfNormal, InverseGamma, InverseWishart, MultivariateNormal, Normal, PositiveDefinite, StudentT, Uniform,
                          UniformOrthogonal)

N_DRAWS = 20_000
POINTS = jnp.array([0.05, 0.3, 0.9, 1.7, 4.0])

# (mutatis distribution, the same distribution in SciPy)
SCALAR_CASES = [
    (Normal(0.3, 1.7), ss.norm(0.3, 1.7)),
    (HalfNormal(0.8), ss.halfnorm(scale=0.8)),
    (Gamma(2.5, 1.5), ss.gamma(2.5, scale=1 / 1.5)),
    (Beta(2.0, 5.0), ss.beta(2.0, 5.0)),
    (InverseGamma(3.0, 2.0), ss.invgamma(3.0, scale=2.0)),
    (StudentT(4.0, -0.5, 2.0), ss.t(4.0, -0.5, 2.0)),
    (Uniform(-1.0, 2.5), ss.uniform(-1.0, 3.5)),
]
IDS = [type(d).__name__ for d, _ in SCALAR_CASES]


@pytest.mark.parametrize("dist, reference", SCALAR_CASES, ids=IDS)
def test_log_prob_matches_scipy(dist, reference):
    inside = POINTS[(POINTS > reference.support()[0]) & (POINTS < reference.support()[1])]
    np.testing.assert_allclose(dist.log_prob(inside), reference.logpdf(np.asarray(inside)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dist, x", [(HalfNormal(1.0), -0.1), (Gamma(2.0, 1.0), -1.0), (Beta(2.0, 2.0), 1.5), (InverseGamma(3.0, 1.0), -2.0),
                                     (Uniform(0.0, 1.0), 2.0)], ids=lambda v: type(v).__name__)
def test_log_prob_outside_domain_is_minus_infinity(dist, x):
    assert float(dist.log_prob(x)) == -np.inf


@pytest.mark.parametrize("dist, reference", SCALAR_CASES, ids=IDS)
def test_sample_mean(dist, reference):
    draws = dist.sample(jax.random.key(0), (N_DRAWS,))
    assert draws.shape == (N_DRAWS,)
    assert_mean_matches(draws, reference.mean())


@pytest.mark.parametrize("cls, mean, sd", [(Gamma, 1.5, 0.4), (Beta, 0.7, 0.1), (InverseGamma, 0.5, 0.2)])
def test_from_mean_sd(cls, mean, sd):
    dist = cls.from_mean_sd(mean, sd)
    reference = {Gamma: lambda d: ss.gamma(d.a, scale=1 / d.rate), Beta: lambda d: ss.beta(d.a, d.b), InverseGamma: lambda d: ss.invgamma(d.a, scale=d.scale)}
    frozen = reference[cls](dist)
    np.testing.assert_allclose([frozen.mean(), frozen.std()], [mean, sd], rtol=1e-12)


def test_multivariate_normal():
    mean, cov = jnp.array([1.0, -2.0, 0.5]), jnp.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
    dist = MultivariateNormal(mean, cov)
    x = jnp.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]])
    np.testing.assert_allclose(dist.log_prob(x), ss.multivariate_normal(mean, cov).logpdf(np.asarray(x)), rtol=1e-12)
    draws = dist.sample(jax.random.key(1), (N_DRAWS,))
    assert_mean_matches(draws, np.asarray(mean))
    assert_covariance_matches(draws, mean, cov)


def test_inverse_wishart():
    scale, df = jnp.array([[2.0, 0.5], [0.5, 1.0]]), 7.0
    dist = InverseWishart(df, scale)
    x = jnp.array([[0.6, 0.1], [0.1, 0.3]])
    np.testing.assert_allclose(dist.log_prob(x), ss.invwishart(df, np.asarray(scale)).logpdf(np.asarray(x)), rtol=1e-12)
    draws = dist.sample(jax.random.key(2), (N_DRAWS,))
    assert bool(jnp.all(jax.vmap(PositiveDefinite().contains)(draws[:100])))
    assert_mean_matches(draws.reshape(N_DRAWS, -1), np.asarray(scale).ravel() / (df - 2 - 1))


def test_uniform_orthogonal():
    n = 4
    draws = UniformOrthogonal(n).sample(jax.random.key(3), (N_DRAWS,))
    np.testing.assert_allclose(jnp.einsum("kji,kjl->kil", draws, draws), jnp.broadcast_to(jnp.eye(n), draws.shape), atol=1e-12)
    # Under the even distribution every entry squared has mean 1/n, and both signs of the determinant are equally likely.
    assert_mean_matches(draws[:, 0, 0] ** 2, 1 / n)
    assert_mean_matches(np.sign(np.linalg.det(np.asarray(draws))), 0.0)
    assert float(UniformOrthogonal(n).log_prob(draws[0])) == 0.0
    assert float(UniformOrthogonal(n).log_prob(draws[0] * 1.01)) == -np.inf


def test_flat_is_improper():
    dist = Flat()
    assert not dist.proper
    np.testing.assert_array_equal(dist.log_prob(jnp.array([-1e9, 0.0, 3.0])), 0.0)
    with pytest.raises(ValueError):
        dist.sample(jax.random.key(0))


@pytest.mark.parametrize("dist", [Gamma(2.5, 1.5), Beta(2.0, 5.0), Uniform(-1.0, 2.5), InverseGamma(3.0, 2.0)], ids=lambda d: type(d).__name__)
def test_density_moved_to_unconstrained_numbers_integrates_to_one(dist):
    # Moving a density from x to z = to_unconstrained(x) multiplies it by the Jacobian. If either is wrong the total is not 1.
    z = jnp.linspace(-30.0, 30.0, 200_001)
    x = jax.vmap(lambda v: dist.domain.to_constrained(v[None], ()))(z)
    log_jacobian = jax.vmap(lambda v: dist.domain.log_det_jacobian(v[None], ()))(z)
    total = jnp.trapezoid(jnp.exp(dist.log_prob(x) + log_jacobian), z)
    np.testing.assert_allclose(total, 1.0, rtol=1e-6)
