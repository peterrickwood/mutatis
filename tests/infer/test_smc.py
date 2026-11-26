"""The tempered SMC sampler on targets with known answers. Tolerances come from the spread across independent runs."""

import jax
import numpy as np
import pytest

from mutatis.infer.smc import smc
from tests.infer.targets import NormalMean, TwoPeaks


def test_smoke():
    target = NormalMean(np.random.default_rng(0).normal(size=(20, 2)))
    result = smc(target, jax.random.key(0), n_particles=200)
    assert result.temperatures[-1] == 1.0 and np.isfinite(result.log_marginal_likelihood)
    assert np.isclose(result.weights.sum(), 1.0)


def _within_run_spread(estimates, truth, standard_errors=5.0):
    estimates = np.asarray(estimates)
    standard_error = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    np.testing.assert_array_less(np.abs(estimates.mean(axis=0) - truth), standard_errors * standard_error + 1e-12)


@pytest.mark.medium
def test_normal_mean_posterior_and_marginal_likelihood():
    target = NormalMean(np.array([1.0, -2.0, 0.5]) + np.random.default_rng(0).normal(size=(50, 3)))
    runs = [smc(target, jax.random.key(seed), n_particles=1000) for seed in range(6)]
    _within_run_spread([r.mean() for r in runs], target.post_mean)
    _within_run_spread([r.sd() for r in runs], np.full(3, np.sqrt(target.post_var)))
    _within_run_spread([r.log_marginal_likelihood for r in runs], target.log_evidence)


@pytest.mark.medium
def test_two_peaks_get_their_weights():
    runs = [smc(TwoPeaks(), jax.random.key(seed), n_particles=1000) for seed in range(6)]
    upper = [float(np.sum(r.weights[r.particles[:, 0] > 0])) for r in runs]
    assert all(0.0 < u < 1.0 for u in upper)  # every run finds both peaks
    _within_run_spread(upper, 0.7)
