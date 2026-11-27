"""The general sampler (step rule, tempering ladder, replicas, starts, stopping rule), part by part and on targets with known
answers.

The ladder's own machinery (exchanges, respacing, round trips, the record used for the evidence) is tested with exact moves at
every rung and a fixed length, so that it is checked apart from any step rule and stopping rule: for a normal target every
rung's distribution is normal, and with exact moves the round-trip rate has a formula (Syed and others, JRSS B 2022, for the
alternating even/odd scheme). Whole-sampler tests run in this process (`processes=1`) unless they test the worker processes.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.infer.evidence import evidence
from mutatis.infer.mcmc import (MAX_WINDOW, MCMCError, NotConvergedWarning, Reference, _Climber, _climb_one, _convergence, _FixedLength,
                                _group_peaks, _plan, _Record, _respace, _Runner, _Sampler, _Search, _search_statement, _UntilConverged,
                                _welford_final, _windows, hardware, mcmc, nuts, split_rhat)
from mutatis.numerics.parallel import Local, compile_program
from mutatis.infer.problem import build_problem
from mutatis.infer.smc import smc
from tests.infer.targets import NormalMean, TwoPeaks
from tests.infer.test_routes import setup


def _normal_target(d=5, seed=0):
    rng = np.random.default_rng(seed)
    return NormalMean(rng.normal(size=d) + rng.normal(size=(50, d)))


def _logp_batch(target):
    return jax.jit(jax.vmap(lambda z: target.log_prior(z) + target.log_likelihood(z)))


def _climber(target) -> _Climber:
    log_target = lambda z: target.log_prior(z) + target.log_likelihood(z)
    return _Climber(target, log_target, jax.jit(jax.vmap(log_target)), _Runner(Local()))


def _banana_log_density(z, b=0.1):
    """Haario's twisted normal: one peak, at (0, 10), curved."""
    y2 = z[1] + b * (z[0] ** 2 - 100.0)
    return -0.5 * (z[0] ** 2 / 100.0 + y2**2)


# ------------------------------------------------------------------------------------------------------- pieces alone

def test_split_rhat():
    rng = np.random.default_rng(0)
    assert np.all(split_rhat(rng.normal(size=(4, 500, 3))) < 1.02)
    stuck = rng.normal(size=(4, 500, 1)) + np.array([0.0, 0.0, 0.0, 5.0])[:, None, None]
    assert split_rhat(stuck)[0] > 1.5


def test_warmup_windows():
    windows = _windows(40)
    assert sum(n for n, _ in windows) == 40
    assert windows[0][1] == "first" and windows[-1][1] == "last"
    slow = [n for n, kind in windows if kind == "slow"]
    assert slow[:-1] == [2**i for i in range(len(slow) - 1)] and slow[-1] >= slow[-2]  # doubling; the last absorbs the rest
    assert _windows(2) == [(2, "slow")]


def test_respace_evens_out_rejection():
    betas, barrier, cumulative = _respace(np.array([0.0, 0.5, 1.0]), np.array([0.9, 0.1]), 3)
    np.testing.assert_allclose(cumulative, [0.0, 0.9, 1.0])
    assert barrier == pytest.approx(1.0)
    np.testing.assert_allclose(betas, [0.0, 0.5 * 0.5 / 0.9, 1.0])  # half the barrier lies at beta = 0.278


def test_convergence_tests():
    rng = np.random.default_rng(0)
    # independent draws: 4 x 2,000 gives a bulk effective size near 8,000, above the 1,600 that precision 0.025 needs
    good = _convergence(rng.normal(size=(4, 2000, 3)), 0, 1, 0.025)
    assert good["converged"] and not good["findings"] and good["min_ess"] > 1600
    short = _convergence(rng.normal(size=(4, 200, 3)), 0, 1, 0.025)           # about 800 effective draws: not yet
    assert not short["converged"] and "bulk effective sample size" in " ".join(short["findings"])
    assert _convergence(rng.normal(size=(4, 200, 3)), 0, 1, 0.05)["converged"]  # 400 are enough for precision 0.05
    stuck = rng.normal(size=(4, 2000, 3)) + np.array([0.0, 0.0, 0.0, 4.0])[:, None, None]
    assert "R-hat" in " ".join(_convergence(stuck, 0, 1, 0.025)["findings"])
    ladder = _convergence(rng.normal(size=(4, 2000, 3)), 3, 10, 0.025)         # a ladder needs 2 round trips per replica
    assert not ladder["converged"] and "round trips" in " ".join(ladder["findings"])
    assert _convergence(rng.normal(size=(4, 2000, 3)), 8, 10, 0.025)["converged"]


def test_kept_draws_are_thinned_past_the_limit_and_totals_count_every_scan():
    record = _Record(ladder=True, divergences=True, limit=100)
    rng = np.random.default_rng(0)
    for _ in range(12):                                     # 12 chunks of 25 scans: 300 scans, 4 replicas, 3 rungs
        record.add([{"top": rng.normal(size=(25, 4, 2)), "ell": rng.normal(size=(25, 4, 3)),
                     "accept_prob": np.full((25, 4, 2), 0.5), "divergent": np.ones((25, 4, 2), bool)}])
    assert record.scans == 300 and record.stride == 4      # 300 kept would pass 100; every 4th scan keeps 75
    assert record.draws().shape == (4, 75, 2) and record.ratios().shape == (4, 75, 3)
    assert all(np.all(i % 4 == 0) for i in record.index)
    assert record.divergences == 300 * 4                   # counted at the posterior rung over every scan, kept or not
    np.testing.assert_allclose(record.swap_acceptance(), [0.5, 0.5])


def test_slow_warmup_windows_stop_doubling():
    rule = _UntilConverged(0.025, 3600.0, 0.0)
    history = [{"window": "first", "planned": 75, "scans": 75}]
    while True:
        n, kind = rule.next_window(history)
        assert n <= MAX_WINDOW
        if n == MAX_WINDOW:
            break
        history.append({"window": kind, "planned": n, "scans": n, "rhat": 2.0})   # chains never agree


def test_dense_mass_matrix_from_sums():
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal([1.0, -2.0, 0.5], [[1.0, 0.6, 0.0], [0.6, 2.0, -0.3], [0.0, -0.3, 0.5]], size=400)
    origin = draws[0]
    x = draws - origin
    n = len(draws)
    dense = np.asarray(_welford_final(jnp.asarray(float(n)), jnp.asarray(x.sum(0)), jnp.asarray(x.T @ x), True))
    expected = (n / (n + 5.0)) * np.cov(draws, rowvar=False) + 1e-3 * (5.0 / (n + 5.0)) * np.eye(3)
    np.testing.assert_allclose(dense, expected, rtol=1e-10)
    diagonal = np.asarray(_welford_final(jnp.asarray(float(n)), jnp.asarray(x.sum(0)), jnp.asarray((x**2).sum(0)), False))
    np.testing.assert_allclose(diagonal, np.diag(expected), rtol=1e-10)


def test_valley_test_separates_peaks_and_joins_one():
    two = TwoPeaks()   # peaks at -3 and +3, sd 0.3: the line between them crosses a deep valley
    labels, n = _group_peaks(np.array([[-3.0], [3.0], [2.95], [-3.05]]), _logp_batch(two))
    assert n == 2 and labels[1] == labels[2] and labels[0] == labels[3] and labels[0] != labels[1]
    one = _normal_target(d=5)
    rng = np.random.default_rng(0)
    near = one.post_mean + np.sqrt(one.post_var) * rng.normal(size=(6, 5))     # points within one normal posterior
    assert _group_peaks(near, _logp_batch(one))[1] == 1


def test_valley_test_joins_end_points_on_a_curved_peak():
    """A banana (Haario's twisted normal): optimisations end at its single mode, give or take, and must count as one peak."""
    batch = jax.jit(jax.vmap(_banana_log_density))
    mode = np.array([0.0, 10.0])
    ends = mode + 1e-3 * np.random.default_rng(0).normal(size=(8, 2))
    assert _group_peaks(ends, batch)[1] == 1


def test_a_climb_certifies_peaks_and_rejects_saddles_and_unfinished_points():
    """f(x, y) = -(x^2 - 1)^2 - y^2: peaks at (-1, 0) and (1, 0), a saddle at (0, 0). A climb that starts exactly at the saddle
    cannot move (the gradient is zero) and must not be certified, as the density rises along x there. Climbs from elsewhere
    end at a peak, certified."""
    f = lambda z: -(z[0] ** 2 - 1.0) ** 2 - z[1] ** 2
    climb = compile_program(_climb_one(f), np.zeros((8, 2)))
    starts = np.array([[0.0, 0.0], [0.3, 1.0], [-0.2, -2.0], [2.0, 0.5], [-3.0, 1.0], [0.0, 0.0], [0.7, 0.0], [-0.7, 0.0]])
    x, value, certified, sd = (np.asarray(a) for a in climb(starts))
    assert not certified[0] and not certified[5], x[[0, 5]]
    assert certified[[1, 2, 3, 4, 6, 7]].all()
    # the certificate: the predicted further rise is below 1e-8, so the value is within 1e-8 of the peak's (0), and with
    # curvature 8 along x the position within sqrt(2 x 1e-8 / 8) = 5e-5 of it
    assert np.all(value[certified] > -1e-8) and np.all(value[certified] <= 0.0)
    np.testing.assert_allclose(np.abs(x[certified][:, 0]), 1.0, atol=5e-5)
    # standard deviations from minus the Hessian at the end point: diag(12 x^2 - 4, 2)
    np.testing.assert_allclose(sd[1], [1 / np.sqrt(12 * x[1, 0] ** 2 - 4), 1 / np.sqrt(2.0)], rtol=1e-12)


def test_ridge_points_of_a_curved_peak_are_not_peaks():
    """On a banana, L-BFGS can stop partway along the curved ridge. Those end points used to count as separate peaks (the
    straight line between two of them dips across the curve). Climbs from points far along the ridge must all end at the one
    peak, though the valley test alone puts the starting points in separate peaks."""
    target = type("Banana", (), {"size": 2, "proper": True, "log_prior": lambda self, z: _banana_log_density(z),
                                 "log_likelihood": lambda self, z: 0.0, "draw_prior": lambda self, k: jnp.array([10.0, 3.0]) * jax.random.normal(k, (2,))})()
    ridge = np.array([[x1, 10.0 - 0.1 * (x1**2 - 100.0) + 10.0 * 0.0] for x1 in (-25.0, -15.0, 15.0, 25.0)])
    ridge[:, 1] = -0.1 * (ridge[:, 0] ** 2 - 100.0)                     # on the ridge: the twisted coordinate is zero
    assert _group_peaks(ridge, jax.jit(jax.vmap(_banana_log_density)))[1] >= 2
    assert _climber(target).count_peaks(ridge, float("inf")) == 1


def test_search_stops_once_enough_starts_find_nothing_new_and_states_its_detection_limit():
    one = _normal_target(d=5)
    search = _climber(one).search(jax.random.key(0), float("inf"), target=64)
    assert search.n_peaks == 1 and search.n_certified == search.n_starts and 64 <= search.n_starts < 64 + 2 * 64
    assert search.detection_limit == pytest.approx(np.log(20) / search.n_starts)
    # the limit's meaning: a peak reached from that share of starting draws is missed by every start with probability
    # (1 - p)^N, at most exp(-p N) = 5% (the two agree as N grows)
    p, N = search.detection_limit, search.n_starts
    assert np.exp(-p * N) == pytest.approx(0.05) and (1 - p) ** N <= 0.05
    assert _search_statement(search, "search", True) == (f"searched {search.n_starts:,} starts (1 peak found): a peak reached from "
                                                         f"fewer than 1 in {search.n_starts / np.log(20):,.0f} prior draws could have been missed")
    assert _search_statement(None, "prior", True).startswith("no peak search (starts='prior')")
    two = TwoPeaks()
    assert _climber(two).search(jax.random.key(0), float("inf"), target=64).n_peaks == 2


def test_search_stops_at_its_time_limit():
    search = _climber(_normal_target(d=5)).search(jax.random.key(0), 0.0)   # deadline already passed: one round, one batch
    assert search.n_starts == 8 and "time limit" in search.stopped


@pytest.mark.medium
@pytest.mark.parametrize("name, peaks", [("banana_d2_strong", 1), ("banana_d10_mild", 1), ("funnel_d10", 1), ("conjugate_regression_n2_p4", 1),
                                         ("two_peaks_d1", 2)])
def test_search_counts_the_peaks_of_benchmark_targets(name, peaks):
    """Targets on which the earlier search (optimisation end points, uncertified) reported false peaks: 2 or 3 on the strong
    banana, 6 on the 21-unknown regression. Each has one peak, except two_peaks_d1."""
    from benchmarks.samplers.targets import build

    target = build(name)
    search = _climber(target).search(jax.random.key(1), float("inf"), target=512)
    print(f"{name}: {search.summary()}")
    assert search.n_peaks == peaks


def test_replicas_and_processes_from_the_hardware(monkeypatch):
    cpu = lambda n: {"platform": "cpu", "devices": 1, "usable_cpus": n, "accelerator": False}
    assert _plan(cpu(1), "auto", "auto") == (4, 1)        # one CPU: 4 replicas in this process
    assert _plan(cpu(8), "auto", "auto") == (8, 8)        # one replica per CPU, each in its own process
    assert _plan(cpu(3), "auto", "auto") == (4, 2)        # at least 4 replicas, shared equally over the processes
    assert _plan(cpu(192), "auto", "auto") == (64, 64)    # at most 64 processes
    assert _plan(cpu(8), 4, "auto") == (4, 4)
    assert _plan(cpu(8), "auto", 1) == (4, 1)
    assert _plan({"platform": "gpu", "devices": 1, "usable_cpus": 8, "accelerator": True}, "auto", "auto") == (64, 1)

    class FakeDevice:
        platform = "gpu"

    monkeypatch.setattr(jax, "devices", lambda *a: [FakeDevice()])
    assert hardware()["accelerator"] is True


def test_refusals():
    target = _normal_target()
    with pytest.raises(MCMCError, match="step rule"):
        mcmc(target, jax.random.key(0), step="gibbs")
    with pytest.raises(MCMCError, match="rungs"):
        mcmc(target, jax.random.key(0), rungs=0)
    with pytest.raises(MCMCError, match="replicas"):
        mcmc(target, jax.random.key(0), replicas=1)
    with pytest.raises(MCMCError, match="positive"):
        mcmc(target, jax.random.key(0), precision=0.0)


def test_time_limit_returns_the_draws_and_says_loudly_they_are_suspect():
    target = _normal_target()
    with pytest.warns(NotConvergedWarning, match="NOT CONVERGED — RESULTS ARE SUSPECT"):
        posterior = mcmc(target, jax.random.key(0), starts="prior", processes=1, max_seconds=1e-3)
    d = posterior.diagnostics
    assert not d["converged"] and d["stopped_by"] == "time limit" and d["status"].startswith("NOT CONVERGED")
    assert d["phase"] == "warm-up" and len(posterior.particles) > 0


# --------------------------------------------------------------------------------- the ladder with exact moves at each rung

class _ExactRungs(_Sampler):
    """The ladder with each rung's move replaced by an exact draw from that rung's distribution (normal for NormalMean)."""

    def __init__(self, target):
        log_target = lambda z: target.log_prior(z) + target.log_likelihood(z)
        super().__init__(target.size, "nuts", Reference("prior", target.log_prior, target.draw_prior), target.log_likelihood, log_target, 0.8, 10, 10)
        self.prior_precision, self.n, self.total = 1.0 / target.prior_sd**2, target.n, jnp.sum(target.y, axis=0)

    def _move(self, key, z, beta, step_size, inv_mass, single):
        precision = self.prior_precision + beta * self.n
        draw = beta * self.total / precision + jax.random.normal(key, z.shape) / jnp.sqrt(precision)
        return draw, jnp.array(1.0), jnp.array(False), jnp.array(0), jnp.array(0)


def _exact_ladder(target, n_rungs, auto, n_draws, seed=0, replicas=4):
    sampler = _ExactRungs(target)
    start = jax.vmap(target.draw_prior)(jax.random.split(jax.random.key(seed + 100), replicas))
    return sampler.run(jax.random.key(seed), start, jnp.ones(target.size), n_rungs, auto, 64, _FixedLength(500, n_draws))


def test_exact_ladder_keeps_every_rung_distribution_and_finds_the_evidence():
    target = _normal_target(d=5)
    posterior = _exact_ladder(target, 12, True, 2000)
    draws = posterior.particles
    n = len(draws)
    # exchanges must leave the posterior rung exactly the posterior: means and variances within sampling error (draws are
    # correlated through exchanges, so allow 6 standard errors at the raw count)
    z = (draws.mean(axis=0) - target.post_mean) / np.sqrt(target.post_var / n)
    assert np.all(np.abs(z) < 6), z
    np.testing.assert_allclose(draws.var(axis=0) / target.post_var, 1.0, atol=6 * np.sqrt(2 / n))
    result = evidence(target, posterior, jax.random.key(1), "stepping_stones")
    assert abs(result.log_value - target.log_evidence) < max(4 * result.standard_error, 0.05), (result, target.log_evidence)


def test_exact_ladder_round_trips_match_the_formula():
    target = _normal_target(d=5)
    posterior = _exact_ladder(target, 10, False, 4000)
    accept = np.array(posterior.diagnostics["swap_acceptance"])
    expected = 4 * 4000 / (2 + 2 * np.sum((1 - accept) / accept))  # 4 replicas
    trips = posterior.diagnostics["round_trips"]
    assert abs(trips - expected) < 5 * np.sqrt(expected), (trips, expected)


def test_exact_ladder_respacing_evens_out_the_exchanges():
    posterior = _exact_ladder(_normal_target(d=20), 8, True, 1000)
    accept = np.array(posterior.diagnostics["swap_acceptance"])
    assert accept.max() - accept.min() < 0.15, accept
    assert posterior.diagnostics["rungs"] >= 2 * posterior.diagnostics["barrier"]


# ------------------------------------------------------------------------------------------------------- whole sampler

def _assert_matches_normal(posterior, target):
    """Means within 5 Monte Carlo errors (from the bulk effective sample size) and standard deviations within 5 of theirs."""
    ess = posterior.diagnostics["min_ess"]
    z = (posterior.mean() - target.post_mean) / np.sqrt(target.post_var)
    assert np.all(np.abs(z) < 5 / np.sqrt(ess)), (z, ess)
    np.testing.assert_allclose(posterior.sd() / np.sqrt(target.post_var), 1.0, atol=5 / np.sqrt(2 * ess))


def test_single_rung_nuts_on_a_normal_target():
    target = NormalMean(np.array([1.0, -2.0]) + np.random.default_rng(0).normal(size=(30, 2)))
    posterior = nuts(target, jax.random.key(0), n_chains=2, processes=1)
    d = posterior.diagnostics
    assert d["rungs"] == 1 and d["ladder"] is None and d["converged"], d["status"]
    assert d["max_rhat"] < 1.01 and d["min_ess"] >= 1600
    _assert_matches_normal(posterior, target)


@pytest.mark.medium
def test_decide_on_one_peak_runs_plain_chains_with_a_full_mass_matrix():
    target = _normal_target(d=10)
    posterior = mcmc(target, jax.random.key(0), rungs="decide", processes=1)
    d = posterior.diagnostics
    assert d["converged"] and d["rungs"] == 1 and d["peaks_found"] == 1 and d["ladder_reason"] is None, d["status"]
    assert d["mass_matrix"] == "full" and d["starts_used"] >= np.log(20) / 1e-4 and d["peak_search"].startswith("searched")
    _assert_matches_normal(posterior, target)


@pytest.mark.medium
@pytest.mark.parametrize("step", ["nuts", "hmc", "random_walk"])
def test_ladder_on_a_normal_target(step):
    target = _normal_target(d=10)
    posterior = mcmc(target, jax.random.key(0), step=step, rungs="auto", starts="pathfinder", processes=1)
    assert posterior.diagnostics["converged"], posterior.diagnostics["status"]
    _assert_matches_normal(posterior, target)
    result = evidence(target, posterior, jax.random.key(1))
    assert abs(result.log_value - target.log_evidence) < max(4 * result.standard_error, 0.1), (result, target.log_evidence)


@pytest.mark.medium
@pytest.mark.parametrize("rungs", ["auto", "decide"])   # the default (always a ladder), and a ladder only where the search finds peaks
def test_finds_two_separated_peaks_and_weighs_them(rungs):
    target = TwoPeaks()
    posterior = mcmc(target, jax.random.key(0), rungs=rungs, processes=1)
    d = posterior.diagnostics
    assert d["converged"] and d["peaks_found"] == 2 and d["rungs"] > 1, d["status"]
    assert rungs == "auto" or "peaks" in d["ladder_reason"], d["ladder_reason"]
    upper = np.mean(posterior.particles[:, 0] > 0)
    # the weight's Monte Carlo error: sd of the indicator (0.46) over the square root of the effective sample size
    assert abs(upper - 0.7) < 5 * 0.46 / np.sqrt(d["min_ess"]), (upper, d["min_ess"])
    exact = float(ss.norm.logpdf(3.0, 0.0, np.sqrt(25.0 + 0.09)))
    result = evidence(target, posterior, jax.random.key(1))
    assert abs(result.log_value - exact) < max(4 * result.standard_error, 0.05), (result, exact)


@pytest.mark.medium
def test_chains_in_separate_peaks_switch_the_ladder_on_during_warm_up():
    """Without the peak search (starts from prior draws), chains settle on both sides; warm-up must see that and change to a
    ladder, which then gets the weights right."""
    target = TwoPeaks()
    posterior = mcmc(target, jax.random.key(0), rungs="decide", starts="prior", processes=1)
    d = posterior.diagnostics
    assert d["escalated"] and "separate peaks" in d["ladder_reason"] and d["rungs"] > 1 and d["converged"], (d["ladder_reason"], d["status"])
    upper = np.mean(posterior.particles[:, 0] > 0)
    assert abs(upper - 0.7) < 5 * 0.46 / np.sqrt(d["min_ess"]), (upper, d["min_ess"])


@pytest.mark.medium
def test_chains_without_a_ladder_seen_in_separate_peaks_never_count_as_converged():
    """Plain chains (no ladder possible) that warm-up saw in separate peaks must be reported as not converged, even if they
    later all fall into one peak and agree."""
    target = TwoPeaks()
    with pytest.warns(NotConvergedWarning, match="separate peaks"):
        posterior = nuts(target, jax.random.key(0), processes=1, max_seconds=300.0)
    d = posterior.diagnostics
    assert not d["converged"] and any(w.get("peaks", 0) >= 2 for w in d["warmup"]), d["status"]


class _Truncated:
    """One unknown: prior N(0, 1); likelihood N(1, 0.5^2) for x > 0, and no likelihood at all (minus infinity) for x <= 0, as
    where a model has no solution. The posterior is a truncated normal, and the evidence has a closed form."""

    size, proper, gradient_ready = 1, True, True

    def log_prior(self, z):
        return jnp.sum(jax.scipy.stats.norm.logpdf(z))

    def log_likelihood(self, z):
        return jnp.where(z[0] > 0, jax.scipy.stats.norm.logpdf(z[0], 1.0, 0.5), -jnp.inf)

    def draw_prior(self, key):
        return jax.random.normal(key, (1,))

    @staticmethod
    def exact():
        # prior x likelihood = N(0; 1, 1.25) x N(x; 0.8, 0.2) on x > 0
        mean, var = 0.8, 0.2
        log_evidence = ss.norm.logpdf(0.0, 1.0, np.sqrt(1.25)) + ss.norm.logsf(0.0, mean, np.sqrt(var))
        truncated = ss.truncnorm((0 - mean) / np.sqrt(var), np.inf, mean, np.sqrt(var))
        return log_evidence, truncated.mean(), truncated.std()


@pytest.mark.medium
def test_ladder_where_the_likelihood_is_zero_in_places():
    target = _Truncated()
    log_evidence, mean, sd = target.exact()
    posterior = mcmc(target, jax.random.key(0), rungs="auto", processes=1)
    assert np.all(posterior.particles > 0)
    assert abs(posterior.mean()[0] - mean) < 5 * sd / np.sqrt(posterior.diagnostics["min_ess"])
    result = evidence(target, posterior, jax.random.key(1))
    assert abs(result.log_value - log_evidence) < max(4 * result.standard_error, 0.05), (result, log_evidence)


class _FlatMean(NormalMean):
    """NormalMean with a flat prior: improper, so the ladder must end at a fitted reference. Posterior N(mean of y, 1/n)."""

    proper = False

    def log_prior(self, z):
        return jnp.zeros(())

    def draw_prior(self, key):
        raise AssertionError("an improper prior cannot be drawn from")


@pytest.mark.medium
def test_ladder_with_an_improper_prior_uses_a_fitted_reference():
    rng = np.random.default_rng(0)
    target = _FlatMean(rng.normal(size=4) + rng.normal(size=(50, 4)))
    with pytest.raises(MCMCError, match="improper"):
        mcmc(target, jax.random.key(0), starts="prior")
    posterior = mcmc(target, jax.random.key(0), rungs="auto", processes=1)
    assert posterior.diagnostics["reference"] == "fitted"
    ess = posterior.diagnostics["min_ess"]
    z = (posterior.mean() - np.asarray(target.y).mean(axis=0)) * np.sqrt(target.n)
    assert np.all(np.abs(z) < 5 / np.sqrt(ess)), z


@pytest.mark.medium
def test_worker_processes_give_the_same_posterior():
    """Replicas split over 2 worker processes (each running the chunk program exported from here) must sample the same
    posterior, with and without a ladder."""
    target = _normal_target(d=5)
    for rungs in (1, "auto"):
        posterior = mcmc(target, jax.random.key(0), rungs=rungs, starts="pathfinder", replicas=4, processes=2)
        d = posterior.diagnostics
        assert d["processes"] == 2 and d["replicas"] == 4 and d["converged"], d["status"]
        _assert_matches_normal(posterior, target)


@pytest.mark.medium
def test_single_rung_nuts_with_sampled_shocks_agrees_with_the_kalman_route():
    _, system, params, solve, data = setup("hidden_level", n_periods=200)
    kalman = build_problem(system, solve, data, route="kalman", start="model")
    sampled = build_problem(system, solve, data, route="sample", start="model")
    reference = smc(kalman, jax.random.key(1), n_particles=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NotConvergedWarning)
        posterior = nuts(sampled, jax.random.key(2), processes=1)
    assert posterior.diagnostics["max_rhat"] < 1.01
    ref = jax.vmap(kalman.parameters)(reference.resample(jax.random.key(3), 4000))
    got = jax.vmap(sampled.parameters)(posterior.particles)
    for name in ref:
        a, b = np.asarray(ref[name]), np.asarray(got[name])
        # Both are Monte Carlo estimates; allow 5 standard errors of the difference, with effective sizes taken conservatively.
        standard_error = np.sqrt(a.var() / 500 + b.var() / 500)
        assert abs(a.mean() - b.mean()) < 5 * standard_error, f"{name}: {a.mean():.3f} against {b.mean():.3f}"
