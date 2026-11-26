"""How the cost of the forward-looking solve, and of its derivative, grows with the number of variables.

Times, for random models with n variables (built so their unique non-exploding solution is known):

- compiling and running one solve, and one solve with its derivative (reverse mode, as a gradient-based sampler uses it);
- the same for 1,000 slightly different models at once (`jax.vmap`), as SMC does for its particles.

Run: .venv/bin/python experiments/forward_looking_solve_cost.py [n ...] [--draws 1000] [--skip-batch-above 40]
Used for decision 34 (docs/design/decisions.md): the derivative's matrix equation was first solved by forming an n^2 x n^2
matrix, whose cost grows with n^6; it is now solved by a doubling iteration that uses only n x n products.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.linear_re import solve_forward_looking
from tests.numerics import reference


def model(n, seed=0):
    rng = np.random.default_rng(seed)
    inside, outside = rng.uniform(-0.95, 0.95, n), rng.choice([-1, 1], n) * rng.uniform(1.05, 3.0, n)
    A, B, C, D, _ = reference.model_with_known_solution(n, inside, outside, rng, k=max(1, n // 4))
    return tuple(map(jnp.asarray, (A, B, C, D))), rng


def timed(fn, *args):
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(*args))
    runs, start = 0, time.perf_counter()
    while runs < 3 or time.perf_counter() - start < 0.5:
        jax.block_until_ready(compiled(*args))
        runs += 1
    return compile_seconds, (time.perf_counter() - start) / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int, default=[5, 10, 20, 40, 80])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--skip-batch-above", type=int, default=80)
    args = parser.parse_args()
    print(f"{'n':>4} | {'one solve':>22} | {'one solve + gradient':>22} | {f'{args.draws} solves':>22} | {f'{args.draws} solves + gradients':>24}")
    print(f"{'':>4} | {'compile s':>10} {'run ms':>11} | {'compile s':>10} {'run ms':>11} | {'compile s':>10} {'run ms':>11} | {'compile s':>10} {'run ms':>13}")
    for n in args.sizes:
        (A, B, C, D), rng = model(n)
        W1, W2 = jnp.asarray(rng.normal(size=(n, n))), jnp.asarray(rng.normal(size=D.shape))

        def summary(A, B, C, D):
            solution = solve_forward_looking(A, B, C, D)
            return jnp.sum(W1 * solution.P) + jnp.sum(W2 * solution.Q)

        solve = lambda A, B, C, D: solve_forward_looking(A, B, C, D).P
        gradient = jax.grad(summary, argnums=(0, 1, 2))
        row = [timed(solve, A, B, C, D), timed(gradient, A, B, C, D)]
        if n <= args.skip_batch_above:
            noise = 0.01 * jax.random.normal(jax.random.key(n), (args.draws, n, n))
            batch = lambda f: lambda noise: jax.vmap(lambda e: f(A + e, B, C, D))(noise)
            row += [timed(batch(solve), noise), timed(batch(gradient), noise)]
        cells = " | ".join(f"{c:10.2f} {1000 * r:11.2f}" for c, r in row)
        print(f"{n:4d} | {cells}", flush=True)


if __name__ == "__main__":
    main()
