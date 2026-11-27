"""Choosing a sampler for a posterior problem.

- Improper priors (flat or Jeffreys), or parameters with no prior: the exact regression formula, if the model qualifies.
- Shocks sampled as unknowns: NUTS chains. SMC is not reliable with that many unknowns (risk experiment 1, PLAN.md).
- Otherwise: sequential Monte Carlo, which copes with several peaks and gives the evidence.

A method can also be forced: any name in `SAMPLERS`, including `mcmc`, the general sampler (step rule, tempering ladder,
replicas, checks; see `mutatis.infer.mcmc`).
"""

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.mcmc import mcmc, nuts
from mutatis.infer.posterior import Posterior
from mutatis.infer.regression import NotApplicable, regression_posterior
from mutatis.infer.smc import smc

SAMPLERS = {"smc": smc, "nuts": nuts, "mcmc": mcmc, "exact_regression": regression_posterior}


def choose_method(problem) -> str:
    return "exact_regression" if not problem.proper else "nuts" if problem.route == "sample" else "smc"


def estimate(problem, key, method: str = "auto", **options) -> Posterior:
    method = choose_method(problem) if method == "auto" else method
    if method not in SAMPLERS:
        raise ValueError(f"unknown method {method!r}: use auto or one of {', '.join(SAMPLERS)}")
    if method == "exact_regression":
        try:
            return regression_posterior(problem, key, **options)
        except NotApplicable as reason:
            raise NotApplicable(f"improper priors need the exact regression formula, which does not apply here: {reason}") from None
    return SAMPLERS[method](problem, key, **options)
