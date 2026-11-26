"""Diagnosis for risk experiment 1: is the sampled-shock posterior wrong, or is SMC failing to sample it?

On the hidden-level toy (200 periods), compare the parameter posterior from:
  - the Kalman route (2 unknowns), the reference;
  - a long NUTS chain on the sample route (203 unknowns), an independent sampler on the same density;
  - SMC on the sample route with increasingly strong moves.

Run: PYTHONPATH=. .venv/bin/python experiments/sample_route_diagnosis.py
"""

import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from mutatis.compile.system import compile_model
from mutatis.infer.problem import build_problem
from mutatis.infer.smc import smc
from mutatis.measure.data import arrange
from mutatis.quantities.simulate import simulate_data
from mutatis.spec.yaml_io import load_model
from mutatis.versions.pipeline import solver


def summary(problem, z):
    values = jax.vmap(problem.parameters)(z)
    return "  ".join(f"{k}={float(jnp.mean(v)):.3f}({float(jnp.std(v)):.3f})" for k, v in values.items())


def main():
    model = load_model(open("tests/models/hidden_level.yaml").read())
    system = compile_model(model)
    solve = solver(system)
    dataset = simulate_data(system, solve({"level.sigma_eta": jnp.asarray(0.3), "y.sigma_eps": jnp.asarray(0.5)}), jax.random.key(0), 200)
    data = arrange(system, dataset, model.data)
    kalman = build_problem(system, solve, data, route="kalman", start="model")
    sample = build_problem(system, solve, data, route="sample", start="model")

    start = time.perf_counter()
    reference = smc(kalman, jax.random.key(1), n_particles=2000)
    print(f"kalman SMC          {time.perf_counter() - start:7.1f} s   {summary(kalman, reference.resample(jax.random.key(2), 4000))}")

    start = time.perf_counter()
    logdensity = jax.jit(sample.log_posterior)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(jax.random.key(3), sample.draw_prior(jax.random.key(4)), num_steps=1000)
    kernel = jax.jit(blackjax.nuts(logdensity, **parameters).step)
    draws, key = [], jax.random.key(5)
    for _ in range(3000):
        key, sub = jax.random.split(key)
        state, _ = kernel(sub, state)
        draws.append(state.position)
    print(f"sample NUTS chain   {time.perf_counter() - start:7.1f} s   {summary(sample, jnp.stack(draws))}")

    for moves, leapfrog in [(5, 5), (20, 10), (50, 20)]:
        start = time.perf_counter()
        result = smc(sample, jax.random.key(6), n_particles=1000, n_mcmc_steps=moves, n_leapfrog=leapfrog)
        unique = len(np.unique(np.round(result.particles[:, :2], 10), axis=0))
        print(f"sample SMC {moves:>2}x{leapfrog:<3}  {time.perf_counter() - start:7.1f} s   {summary(sample, result.resample(jax.random.key(7), 4000))}"
              f"   log ML {result.log_marginal_likelihood:.2f}   distinct particles {unique}")
    print(f"kalman log ML {reference.log_marginal_likelihood:.2f}")


if __name__ == "__main__":
    main()
