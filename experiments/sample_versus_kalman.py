"""Risk experiment 1 from PLAN.md: sampling the shocks against averaging them out with the Kalman filter.

On the same data and model, estimate by SMC twice: once with the Kalman route (unknowns: the parameters), once with the shocks
and starting state sampled as unknowns too. Report time, number of unknowns, and the parameter posteriors side by side.

Run: .venv/bin/python experiments/sample_versus_kalman.py
"""

import time

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

CASES = {
    "hidden_level": {"level.sigma_eta": 0.3, "y.sigma_eps": 0.5},
    "forward_looking": {"demand.rho": 0.8, "demand.sigma_x": 0.7, "price.a": 0.5},
}


def main(n_periods=200, n_particles=1000):
    for name, truth in CASES.items():
        model = load_model(open(f"tests/models/{name}.yaml").read())
        system = compile_model(model)
        solve = solver(system)
        params = {k: jnp.asarray(v) for k, v in truth.items()}
        dataset = simulate_data(system, solve(params), jax.random.key(0), n_periods)
        data = arrange(system, dataset, model.data)
        print(f"\n{name}: {n_periods} periods, {n_particles} particles")
        for route in ("kalman", "sample"):
            problem = build_problem(system, solve, data, route=route, start="model")
            start = time.perf_counter()
            try:
                posterior = smc(problem, jax.random.key(1), n_particles=n_particles)
            except Exception as error:  # report, do not hide: this is the point of the experiment
                print(f"  {route:<7} unknowns {problem.size:>4}  FAILED after {time.perf_counter() - start:6.1f} s: {error}")
                continue
            seconds = time.perf_counter() - start
            draws = jax.vmap(problem.parameters)(posterior.resample(jax.random.key(2), 4000))
            summary = "  ".join(f"{k}={float(jnp.mean(v)):.3f}({float(jnp.std(v)):.3f})" for k, v in draws.items() if k in truth)
            steps = len(posterior.diagnostics["temperatures"]) - 1
            print(f"  {route:<7} unknowns {problem.size:>4}  {seconds:7.1f} s  {steps:>3} temperature steps  "
                  f"log ML {posterior.log_marginal_likelihood:9.2f}  {summary}")
        print(f"  truth: {truth}")


if __name__ == "__main__":
    main()
