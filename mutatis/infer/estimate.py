"""Choosing a sampler for a posterior problem.

- Improper priors (flat or Jeffreys), or parameters with no prior: the exact regression formula, if the model qualifies.
- A rotation with no prior that the likelihood ignores, with proper priors on everything else, in a model the formula does not
  cover: the default sampler for the rest, then ranges over the rotation for each draw (`mutatis.infer.restricted`).
- Shocks sampled as unknowns: NUTS chains. SMC is not reliable with that many unknowns (risk experiment 1, PLAN.md).
- Otherwise: sequential Monte Carlo, which copes with several peaks and gives the evidence.

A method can also be forced: any name in `SAMPLERS`, including `mcmc`, the general sampler (step rule, tempering ladder,
replicas, checks; see `mutatis.infer.mcmc`).
"""

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.mcmc import mcmc, nuts
from mutatis.infer.posterior import Posterior
from mutatis.infer.regression import NotApplicable, regression_posterior
from mutatis.infer.restricted import ranges_posterior
from mutatis.infer.smc import smc

SAMPLERS = {"smc": smc, "nuts": nuts, "mcmc": mcmc, "exact_regression": regression_posterior}


def choose_method(problem) -> str:
    return "exact_regression" if not problem.proper else "nuts" if problem.route == "sample" else "smc"


def _ranges_apply(problem) -> bool:
    return bool(problem.no_prior_unknowns) and all(u.prior.proper for u in problem.parameter_unknowns)


def estimate(problem, key, method: str = "auto", **options) -> Posterior:
    chosen = choose_method(problem) if method == "auto" else method
    if chosen not in SAMPLERS:
        raise ValueError(f"unknown method {chosen!r}: use auto or one of {', '.join(SAMPLERS)}")
    if chosen == "exact_regression":
        try:
            return regression_posterior(problem, key, **options)
        except NotApplicable as reason:
            if method == "auto" and _ranges_apply(problem):
                return ranges_posterior(problem, key, mcmc, **options)
            raise NotApplicable(f"improper priors need the exact regression formula, which does not apply here: {reason}") from None
    if _ranges_apply(problem):
        return ranges_posterior(problem, key, SAMPLERS[chosen], **options)
    return SAMPLERS[chosen](problem, key, **options)
