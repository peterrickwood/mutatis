"""Calibration of the samplers: over many simulated data sets, the truth must fall in the posterior as often as the
posterior says it should.

For SMC and NUTS, simulation-based calibration (Talts and others, 2018): draw the unknowns from the prior, simulate data, sample
the posterior, and record where the drawn value falls in it (the posterior's cumulative probability there). If the sampler is
right, those positions are uniform on (0, 1), whatever the model. The target here has a skewed posterior (a mean and a scale).

For the exact regression formula the prior is flat and cannot be drawn from, so the check is frequentist: with fixed true values
and many simulated data sets, the positions are again uniform, because with a flat prior on the coefficients and the Jeffreys
prior on the covariance the posterior of each coefficient matches its sampling distribution (to within the difference between
Student t distributions with 196 and 197 degrees of freedom here).

Uniformity is judged by the Kolmogorov-Smirnov test for each unknown separately (positions of different unknowns from the same
data set are not independent), failing below p = 0.001 divided by the number of unknowns. Seeds are fixed, so each test is
deterministic.

Tiers: SMC and NUTS are medium (about 2 minutes together); the exact regression formula is slow (about 11 minutes, because it
builds and compiles a new problem for each of its 150 data sets).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.infer.nuts import nuts
from mutatis.infer.problem import build_problem
from mutatis.infer.regression import regression_posterior
from mutatis.infer.smc import smc
from mutatis.spec import edits as e
from tests.infer.test_routes import PARAMS, setup
from tests.spec.fixtures import toy

class MeanAndScale:
    """y(i) ~ N(mu, sigma^2) for i = 1..10, with mu ~ N(0, 1) and sigma ~ HalfNormal(1). Unknowns z = (mu, log sigma)."""

    size = 2

    def __init__(self, y):
        self.y = jnp.asarray(y)

    def log_prior(self, z):
        return jax.scipy.stats.norm.logpdf(z[0]) + jnp.log(2.0) + jax.scipy.stats.norm.logpdf(jnp.exp(z[1])) + z[1]  # + log |d sigma / dz|

    def log_likelihood(self, z):
        return jnp.sum(jax.scipy.stats.norm.logpdf(self.y, z[0], jnp.exp(z[1])))

    def draw_prior(self, key):
        k1, k2 = jax.random.split(key)
        return jnp.array([jax.random.normal(k1), jnp.log(jnp.abs(jax.random.normal(k2)))])


def _simulate(rng):
    truth = np.array([rng.normal(), np.log(abs(rng.normal()))])
    return truth, truth[0] + np.exp(truth[1]) * rng.normal(size=10)


def _uniform(positions, name):
    positions = np.asarray(positions)
    p = [ss.kstest(column, "uniform").pvalue for column in positions.T]
    counts = [np.histogram(column, bins=5, range=(0, 1))[0].tolist() for column in positions.T]
    print(f"\n{name}: {len(positions)} data sets; counts in fifths per unknown {counts}; Kolmogorov-Smirnov p {np.round(p, 3).tolist()}")
    assert min(p) > 1e-3 / len(p)


@pytest.mark.medium
def test_smc_is_calibrated():
    rng, positions = np.random.default_rng(0), []
    for replication in range(100):
        truth, y = _simulate(rng)
        posterior = smc(MeanAndScale(y), jax.random.key(replication), n_particles=500)
        positions.append([np.sum(posterior.weights * (posterior.particles[:, j] < truth[j])) for j in range(2)])
    _uniform(positions, "SMC")


@pytest.mark.medium
def test_nuts_is_calibrated():
    rng, positions = np.random.default_rng(1), []
    for replication in range(100):
        truth, y = _simulate(rng)
        posterior = nuts(MeanAndScale(y), jax.random.key(replication), n_chains=1, n_warmup=300, n_draws=400)
        positions.append([np.mean(posterior.particles[:, j] < truth[j]) for j in range(2)])
    _uniform(positions, "NUTS")


@pytest.mark.slow
def test_exact_regression_is_calibrated():
    flat = [e.SetPrior("dynamics.c", prior="flat"), e.SetPrior("dynamics.B", prior="flat"), e.SetPrior("dynamics.Sigma", prior="jeffreys")]
    model = toy("two_variable")
    for edit in flat:
        model = edit.apply(model)
    truth = np.concatenate([np.ravel(PARAMS["two_variable"]["dynamics.c"]), np.ravel(PARAMS["two_variable"]["dynamics.B"])])
    positions = []
    for replication in range(150):
        _, system, _, solve, data = setup("two_variable", n_periods=200, model=model, seed=replication)
        problem = build_problem(system, solve, data)
        draws = jax.vmap(problem.parameters)(regression_posterior(problem, jax.random.key(replication), n_draws=1000).particles)
        coefficients = np.concatenate([np.asarray(draws["dynamics.c"]), np.asarray(draws["dynamics.B"]).reshape(-1, 4)], axis=1)
        positions.append(np.mean(coefficients < truth, axis=0))
    _uniform(positions, "Exact regression")
