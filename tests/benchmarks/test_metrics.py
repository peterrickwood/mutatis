"""The benchmark's accuracy measures, on draws whose quality is known: exact draws must pass, distorted ones must fail."""

import numpy as np
import scipy.stats as ss

from benchmarks.samplers import targets as T
from benchmarks.samplers.metrics import accuracy, evidence_accuracy, weighted_ks


def test_weighted_ks_equals_scipy_with_equal_weights():
    x = np.random.default_rng(0).normal(size=5000)
    assert abs(weighted_ks(x, np.ones_like(x), ss.norm.cdf) - ss.kstest(x, ss.norm.cdf).statistic) < 1e-12


def test_weighted_ks_uses_the_weights():
    x = np.array([-1.0, 1.0])
    # all weight on -1: the empirical distribution jumps from 0 to 1 at -1
    assert abs(weighted_ks(x, np.array([1.0, 0.0]), ss.norm.cdf) - (1 - ss.norm.cdf(-1.0))) < 1e-12


def test_exact_draws_are_accurate():
    t = T.build("correlated_normal_d10_c10000")
    z = t.exact_draws(np.random.default_rng(0), 20000)
    a = accuracy(t, z, np.ones(len(z)))
    assert a["posterior_ok"], a["failed"]


def test_shifted_draws_fail_on_the_mean():
    t = T.build("correlated_normal_d10_c100")
    z = t.exact_draws(np.random.default_rng(0), 20000)
    z[:, 3] += 0.2 * t.reference.sd[3]
    a = accuracy(t, z, np.ones(len(z)))
    assert not a["posterior_ok"] and a["worst_quantity"] == "z[3]" and abs(a["max_abs_mean_error"] - 0.2) < 0.03


def test_narrow_draws_fail_on_the_spread():
    t = T.build("correlated_normal_d10_c100")
    z = t.exact_draws(np.random.default_rng(0), 20000)
    z = t.m + 0.8 * (z - t.m)
    a = accuracy(t, z, np.ones(len(z)))
    assert not a["posterior_ok"] and abs(a["max_sd_ratio"] - 0.8) < 0.02


def test_wrong_peak_weights_fail():
    t = T.build("two_peaks_d5")
    rng = np.random.default_rng(0)
    exact = t.exact_draws(rng, 20000)
    assert accuracy(t, exact, np.ones(len(exact)))["posterior_ok"]
    peak = t.peak_of(exact)
    assert abs(np.mean(peak == 0) - 0.3) < 0.02
    # half the draws in each peak: the weights are wrong even though each peak's shape is right
    balanced = np.concatenate([exact[peak == 0][:5000], exact[peak == 1][:5000]])
    a = accuracy(t, balanced, np.ones(len(balanced)))
    assert not a["posterior_ok"] and abs(a["peak_weight_error"] - 0.2) < 0.01


def test_weights_are_used():
    t = T.build("two_peaks_d1")
    exact = t.exact_draws(np.random.default_rng(1), 20000)
    peak = t.peak_of(exact)
    balanced = np.concatenate([exact[peak == 0][:5000], exact[peak == 1][:5000]])
    weights = np.where(t.peak_of(balanced) == 0, 0.3, 0.7)
    assert accuracy(t, balanced, weights)["posterior_ok"]


def test_non_finite_draws_fail():
    t = T.build("normal_means_d2")
    z = np.array([[0.0, np.nan], [1.0, 1.0]])
    assert not accuracy(t, z, np.ones(2))["posterior_ok"]


def test_evidence_accuracy():
    assert evidence_accuracy(-10.3, 0.1, -10.0)["ok"] and not evidence_accuracy(-10.6, 0.1, -10.0)["ok"]
    assert not evidence_accuracy(None, None, -10.0)["ok"] and not evidence_accuracy(float("nan"), None, -10.0)["ok"]
    unknown = evidence_accuracy(-10.3, 0.1, float("nan"))
    assert unknown["ok"] is None and unknown["error"] is None and unknown["log_value"] == -10.3
    assert evidence_accuracy(None, None, float("nan"))["ok"] is None
