"""Independent NUTS chains: the sampler for posterior problems with many unknowns, such as when shocks are sampled.

Each chain starts from its own prior draw, tunes its step size and scales during a warm-up (BlackJAX's window adaptation),
then draws. Chains are compared with the split R-hat statistic: values near 1 mean they agree; above about 1.01 they have not.

Used where sequential Monte Carlo is not reliable (risk experiment 1 in PLAN.md: with about 200 unknowns, SMC's posterior was
wrong and its marginal likelihood off by up to 20). NUTS explores one region well but does not move between separated peaks,
and does not give the marginal likelihood, so none is reported.
"""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.posterior import Posterior


class NUTSError(RuntimeError):
    pass


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split R-hat per unknown, from draws of shape (chains, draws, unknowns) (Gelman and others, Bayesian Data Analysis, 3rd ed.)."""
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    n = split.shape[1]
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = n * split.mean(axis=1).var(axis=0, ddof=1)
    pooled = (n - 1) / n * within + between / n
    return np.sqrt(pooled / np.maximum(within, np.finfo(float).tiny))


def nuts(problem, key, n_chains: int = 4, n_warmup: int = 1000, n_draws: int = 1000) -> Posterior:
    if not getattr(problem, "gradient_ready", True):
        raise NUTSError("NUTS moves every unknown by gradients, but this problem has unknowns it cannot move (a rotation, or no prior)")
    logdensity = jax.jit(lambda z: problem.log_prior(z) + problem.log_likelihood(z))  # contract 2 only
    chains, divergences = [], 0
    for chain_key in jax.random.split(key, n_chains):
        key_start, key_warmup, key_draws = jax.random.split(chain_key, 3)
        start = problem.draw_prior(key_start)
        (state, parameters), _ = blackjax.window_adaptation(blackjax.nuts, logdensity).run(key_warmup, start, num_steps=n_warmup)
        step = blackjax.nuts(logdensity, **parameters).step

        def one(state, k):
            state, info = step(k, state)
            return state, (state.position, info.is_divergent)

        _, (positions, divergent) = jax.lax.scan(one, state, jax.random.split(key_draws, n_draws))
        chains.append(np.asarray(positions))
        divergences += int(jnp.sum(divergent))
    stacked = np.stack(chains)
    if not np.all(np.isfinite(stacked)):
        raise NUTSError("some draws are not finite")
    rhat = split_rhat(stacked)
    draws = stacked.reshape(-1, stacked.shape[-1])
    return Posterior(draws, np.full(len(draws), 1.0 / len(draws)), None, "nuts",
                     {"chains": n_chains, "max_rhat": float(np.max(rhat)), "divergences": divergences})
