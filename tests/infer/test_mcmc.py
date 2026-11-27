"""The general sampler (step rule, tempering ladder, replicas, checks), part by part and on targets with known answers.

The ladder's own machinery (exchanges, respacing, round trips, the record used for the evidence) is tested with exact moves at
every rung, so that it is checked apart from any step rule: for a normal target every rung's distribution is normal, and with
exact moves the round-trip rate has a formula (Syed and others, JRSS B 2022, for the alternating even/odd scheme).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mutatis.infer.evidence import evidence, stepping_stones
from mutatis.infer.mcmc import MCMCError, Reference, _checks, _respace, _Sampler, _windows, mcmc, nuts, split_rhat
from mutatis.infer.problem import build_problem
from mutatis.infer.smc import smc
from tests.infer.targets import NormalMean, TwoPeaks
from tests.infer.test_routes import setup


def _normal_target(d=5, seed=0):
    rng = np.random.default_rng(seed)
    return NormalMean(rng.normal(size=d) + rng.normal(size=(50, d)))


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


def test_checks_flag_disagreeing_replicas_and_pass_good_ones():
    rng = np.random.default_rng(0)
    good = _checks(rng.normal(size=(4, 1000, 3)), 0, 1, 0, None)
    assert good["converged"] and not good["findings"]
    stuck = rng.normal(size=(4, 1000, 3)) + np.array([0.0, 0.0, 0.0, 4.0])[:, None, None]
    bad = _checks(stuck, 3, 10, 1, 6.0)
    assert not bad["converged"]
    text = " ".join(bad["findings"])
    assert "R-hat" in text and "divergent" in text and "round trips" in text and "rungs" in text


def test_refusals():
    target = _normal_target()
    with pytest.raises(MCMCError, match="step rule"):
        mcmc(target, jax.random.key(0), step="gibbs")
    with pytest.raises(MCMCError, match="rungs"):
        mcmc(target, jax.random.key(0), rungs=0)


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
    return sampler.run(jax.random.key(seed), start, jnp.ones(target.size), n_rungs, auto, 64, 500, n_draws)


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

def test_single_rung_nuts_on_a_normal_target():
    target = NormalMean(np.array([1.0, -2.0]) + np.random.default_rng(0).normal(size=(30, 2)))
    posterior = nuts(target, jax.random.key(0), n_chains=2, n_warmup=300, n_draws=500)
    assert posterior.diagnostics["rungs"] == 1 and posterior.diagnostics["ladder"] is None
    assert posterior.diagnostics["max_rhat"] < 1.02
    assert np.all(np.abs(posterior.mean() - target.post_mean) < 5 * np.sqrt(target.post_var / 200))


@pytest.mark.medium
@pytest.mark.parametrize("step", ["nuts", "hmc", "random_walk"])
def test_ladder_on_a_normal_target(step):
    target = _normal_target(d=10)
    posterior = mcmc(target, jax.random.key(0), step=step, n_warmup=500, n_draws=1000)
    ess = posterior.diagnostics["min_ess"]
    z = (posterior.mean() - target.post_mean) / np.sqrt(target.post_var)
    assert np.all(np.abs(z) < 5 / np.sqrt(ess)), (z, ess)
    np.testing.assert_allclose(posterior.sd() / np.sqrt(target.post_var), 1.0, atol=5 / np.sqrt(ess) + 0.02)
    result = evidence(target, posterior, jax.random.key(1))
    assert abs(result.log_value - target.log_evidence) < max(4 * result.standard_error, 0.1), (result, target.log_evidence)


@pytest.mark.medium
def test_ladder_weighs_two_separated_peaks():
    target = TwoPeaks()
    posterior = mcmc(target, jax.random.key(0))
    upper = np.mean(posterior.particles[:, 0] > 0)
    assert abs(upper - 0.7) < 0.05, upper
    assert posterior.diagnostics["converged"], posterior.diagnostics["findings"]
    exact = float(ss.norm.logpdf(3.0, 0.0, np.sqrt(25.0 + 0.09)))
    result = evidence(target, posterior, jax.random.key(1))
    assert abs(result.log_value - exact) < max(4 * result.standard_error, 0.05), (result, exact)


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
    posterior = mcmc(target, jax.random.key(0))
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
    posterior = mcmc(target, jax.random.key(0), n_warmup=500, n_draws=1000)
    assert posterior.diagnostics["reference"] == "fitted"
    ess = posterior.diagnostics["min_ess"]
    z = (posterior.mean() - np.asarray(target.y).mean(axis=0)) * np.sqrt(target.n)
    assert np.all(np.abs(z) < 5 / np.sqrt(ess)), z


@pytest.mark.medium
def test_single_rung_nuts_with_sampled_shocks_agrees_with_the_kalman_route():
    _, system, params, solve, data = setup("hidden_level", n_periods=200)
    kalman = build_problem(system, solve, data, route="kalman", start="model")
    sampled = build_problem(system, solve, data, route="sample", start="model")
    reference = smc(kalman, jax.random.key(1), n_particles=2000)
    posterior = nuts(sampled, jax.random.key(2))
    assert posterior.diagnostics["max_rhat"] < 1.05
    ref = jax.vmap(kalman.parameters)(reference.resample(jax.random.key(3), 4000))
    got = jax.vmap(sampled.parameters)(posterior.particles)
    for name in ref:
        a, b = np.asarray(ref[name]), np.asarray(got[name])
        # Both are Monte Carlo estimates; allow 5 standard errors of the difference, with effective sizes taken conservatively.
        standard_error = np.sqrt(a.var() / 500 + b.var() / 500)
        assert abs(a.mean() - b.mean()) < 5 * standard_error, f"{name}: {a.mean():.3f} against {b.mean():.3f}"
