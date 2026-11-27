"""Sequential Monte Carlo with tempering: the default sampler.

Thousands of draws ("particles") start from the prior. The data are brought in gradually: at each step the likelihood is
raised to a power ("temperature") that climbs from 0 to 1, each step as large as possible while the particles' weights stay
usable (effective sample size at `target_ess` of the particles). After each step the particles are resampled and moved by
Hamiltonian Monte Carlo, which uses gradients. Because the particles start spread across the whole prior, they settle wherever
the posterior has weight, including on several peaks. The sum of each step's average weight estimates the log marginal
likelihood, which comparing versions needs.

Between steps the moves are retuned: the step size from the acceptance rate, and the scale of each unknown from the particles'
spread. Built on BlackJAX's adaptive tempered SMC; this module drives its loop and does the retuning.

Results carry weights. Summaries must use them (`Posterior.mean`, `Posterior.sd`), or `Posterior.resample` for equally weighted draws.
"""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc import adaptive_tempered, resampling

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.posterior import Posterior


class SMCError(RuntimeError):
    pass


def smc(problem, key, n_particles: int = 2000, n_mcmc_steps: int = 5, n_leapfrog: int = 5, target_ess: float = 0.5,
        target_acceptance: float = 0.65, initial_step_size: float = 0.5, max_steps: int = 1000) -> Posterior:
    """Sample a posterior problem (contract 2: `draw_prior`, `log_prior`, `log_likelihood`)."""
    if not getattr(problem, "gradient_ready", True):
        raise SMCError("SMC moves every unknown by gradients, but this problem has unknowns it cannot move (a rotation, or no prior)")
    key_prior, key_steps = jax.random.split(key)
    particles = jax.vmap(problem.draw_prior)(jax.random.split(key_prior, n_particles))
    if not bool(jnp.all(jnp.isfinite(particles))):
        raise SMCError("some prior draws are not finite")
    kernel = adaptive_tempered.build_kernel(problem.log_prior, problem.log_likelihood, blackjax.hmc.build_kernel(), blackjax.hmc.init,
                                            resampling.systematic, target_ess)
    step = jax.jit(lambda k, state, params: kernel(k, state, n_mcmc_steps, params))
    state = adaptive_tempered.init(particles)
    step_size, log_evidence, temperatures, acceptance = initial_step_size, 0.0, [0.0], []
    for _ in range(max_steps):
        if float(state.tempering_param) >= 1.0:
            break
        key_steps, subkey = jax.random.split(key_steps)
        spread = _weighted_variance(state.particles, state.weights)
        params = dict(step_size=jnp.array([step_size]), inverse_mass_matrix=jnp.maximum(spread, 1e-12)[None],
                      num_integration_steps=jnp.array([n_leapfrog]))
        state, info = step(subkey, state, params)
        increment = float(info.log_likelihood_increment)
        if not np.isfinite(increment):
            raise SMCError(f"the likelihood is not finite for any particle at temperature {temperatures[-1]:.3g}")
        log_evidence += increment
        rate = float(jnp.mean(info.update_info.acceptance_rate))
        step_size *= float(np.exp(rate - target_acceptance))
        temperatures.append(float(state.tempering_param))
        acceptance.append(rate)
    else:
        raise SMCError(f"the temperature reached only {float(state.tempering_param):.3g} after {max_steps} steps")
    stages = len(temperatures) - 1  # each: a gradient at every particle to start its moves, then the moves' leapfrog steps
    return Posterior(np.asarray(state.particles), np.asarray(state.weights), log_evidence, "smc",
                     {"temperatures": tuple(temperatures), "acceptance": tuple(acceptance),
                      "gradient_evaluations": stages * n_particles * (1 + n_mcmc_steps * n_leapfrog)})


def _weighted_variance(particles, weights):
    mean = jnp.sum(weights[:, None] * particles, axis=0)
    return jnp.sum(weights[:, None] * (particles - mean) ** 2, axis=0)
