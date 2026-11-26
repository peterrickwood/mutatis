"""NUTS chains: on a known target, and on the sampled-shock route where SMC failed (risk experiment 1)."""

import jax
import numpy as np
import pytest

from mutatis.infer.nuts import nuts, split_rhat
from mutatis.infer.problem import build_problem
from mutatis.infer.smc import smc
from tests.infer.targets import NormalMean
from tests.infer.test_routes import setup


def test_split_rhat():
    rng = np.random.default_rng(0)
    assert np.all(split_rhat(rng.normal(size=(4, 500, 3))) < 1.02)
    stuck = rng.normal(size=(4, 500, 1)) + np.array([0.0, 0.0, 0.0, 5.0])[:, None, None]
    assert split_rhat(stuck)[0] > 1.5


def test_normal_target():
    target = NormalMean(np.array([1.0, -2.0]) + np.random.default_rng(0).normal(size=(30, 2)))
    posterior = nuts(target, jax.random.key(0), n_chains=2, n_warmup=300, n_draws=500)
    assert posterior.diagnostics["max_rhat"] < 1.02
    assert np.all(np.abs(posterior.mean() - target.post_mean) < 5 * np.sqrt(target.post_var / 200))  # about 200 effective draws, conservatively


@pytest.mark.medium
def test_sampled_shocks_agree_with_the_kalman_route():
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
