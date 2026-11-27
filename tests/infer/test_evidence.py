"""The evidence step, on draws whose distribution is exact, so that only the estimators are tested (no sampler involved)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.infer.evidence import EvidenceError, bridge, evidence, stepping_stones
from mutatis.infer.posterior import Posterior
from tests.infer.targets import NormalMean, TwoPeaks


def _normal_target(d, seed=0):
    rng = np.random.default_rng(seed)
    return NormalMean(rng.normal(size=d) + rng.normal(size=(50, d)))


def _exact_tempered(target, betas, n, key):
    """Independent draws from every rung of a ladder over NormalMean (each rung is normal), and ell = log likelihood there."""
    prior_precision, total = 1.0 / target.prior_sd**2, np.asarray(target.y).sum(axis=0)
    ells = []
    for beta, k in zip(betas, jax.random.split(key, len(betas))):
        precision = prior_precision + beta * target.n
        draws = beta * total / precision + jax.random.normal(k, (n, target.size)) / np.sqrt(precision)
        ells.append(np.asarray(jax.vmap(target.log_likelihood)(draws)))
    return np.stack(ells, axis=-1)  # (n, rungs)


@pytest.mark.parametrize("seed", range(4))
def test_stepping_stones_on_exact_rungs(seed):
    target = _normal_target(5, seed)
    betas = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 20)])
    ell = _exact_tempered(target, betas, 4000, jax.random.key(seed))[None]
    result = stepping_stones(betas, ell)
    assert abs(result.log_value - target.log_evidence) < 4 * result.standard_error, (result, target.log_evidence)
    assert result.standard_error < 0.1


def test_stepping_stones_where_the_likelihood_is_zero_in_places():
    """Prior N(0, 1); likelihood N(1, 0.5^2) for x > 0 and zero for x <= 0. At beta = 0 the draws cover x <= 0, where
    ell = -infinity; above it every rung is a normal truncated to x > 0."""
    betas = np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 15)])
    rng = np.random.default_rng(0)
    n, ells = 20_000, []
    for beta in betas:
        if beta == 0:
            x = rng.normal(size=n)
        else:
            precision = 1.0 + 4.0 * beta
            mean, sd = 4.0 * beta / precision, 1.0 / np.sqrt(precision)
            x = ss.truncnorm.rvs(-mean / sd, np.inf, mean, sd, size=n, random_state=rng)
        ells.append(np.where(x > 0, ss.norm.logpdf(x, 1.0, 0.5), -np.inf))
    exact = ss.norm.logpdf(0.0, 1.0, np.sqrt(1.25)) + ss.norm.logsf(0.0, 0.8, np.sqrt(0.2))
    result = stepping_stones(betas, np.stack(ells, axis=-1)[None])
    assert abs(result.log_value - exact) < 4 * result.standard_error, (result, exact)


def test_stepping_stones_needs_a_full_ladder():
    with pytest.raises(EvidenceError, match="beta = 0"):
        stepping_stones(np.array([0.1, 1.0]), np.zeros((1, 10, 2)))


def test_bridge_on_exact_normal_draws():
    target = _normal_target(5)
    draws = target.post_mean + np.sqrt(target.post_var) * np.random.default_rng(0).normal(size=(4, 2000, 5))
    log_target = jax.jit(jax.vmap(lambda z: target.log_prior(z) + target.log_likelihood(z)))
    result = bridge(log_target, draws, jax.random.key(0))
    assert abs(result.log_value - target.log_evidence) < 4 * result.standard_error, (result, target.log_evidence)
    assert result.standard_error < 0.05


def test_bridge_on_exact_draws_from_two_peaks():
    target = TwoPeaks()
    rng = np.random.default_rng(0)
    upper = rng.uniform(size=(4, 4000)) < 0.7
    # each peak of the posterior: prior N(0, 25) x N(c, 0.3^2) is normal with variance 1 / (1/25 + 1/0.09)
    var = 1.0 / (1.0 / 25.0 + 1.0 / 0.09)
    draws = (np.where(upper, 3.0, -3.0) * var / 0.09 + np.sqrt(var) * rng.normal(size=(4, 4000)))[..., None]
    log_target = jax.jit(jax.vmap(lambda z: target.log_prior(z) + target.log_likelihood(z)))
    exact = float(ss.norm.logpdf(3.0, 0.0, np.sqrt(25.0 + 0.09)))
    result = bridge(log_target, draws, jax.random.key(0))
    assert abs(result.log_value - exact) < 4 * result.standard_error, (result, exact)


def test_evidence_refuses_improper_priors_and_chooses_its_method():
    target = _normal_target(2)

    class Improper(NormalMean):
        proper = False

    with pytest.raises(EvidenceError, match="improper"):
        evidence(Improper(np.zeros((5, 2))), None, jax.random.key(0))
    draws = target.post_mean + np.sqrt(target.post_var) * np.random.default_rng(0).normal(size=(2, 1000, 2))
    posterior = Posterior(draws.reshape(-1, 2), np.full(2000, 1 / 2000), None, "test", {"chains": draws, "ladder": None})
    assert evidence(target, posterior, jax.random.key(0)).method == "bridge"
    with pytest.raises(EvidenceError, match="ladder"):
        evidence(target, posterior, jax.random.key(0), "stepping_stones")
