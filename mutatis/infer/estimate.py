"""Choosing a sampler for a posterior problem.

- Improper priors (flat or Jeffreys), or parameters with no prior: the exact regression formula, if the model qualifies. SMC
  cannot start from an improper prior, because it draws from it.
- Shocks sampled as unknowns: NUTS chains. SMC is not reliable with that many unknowns (risk experiment 1, PLAN.md).
- Otherwise: sequential Monte Carlo, which copes with several peaks and gives the marginal likelihood.

A method can also be forced.
"""

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.nuts import nuts
from mutatis.infer.posterior import Posterior
from mutatis.infer.regression import NotApplicable, regression_posterior
from mutatis.infer.smc import smc


def estimate(problem, key, method: str = "auto", **options) -> Posterior:
    if method == "auto":
        method = "exact_regression" if not problem.proper else "nuts" if problem.route == "sample" else "smc"
    if method == "smc":
        return smc(problem, key, **options)
    if method == "nuts":
        return nuts(problem, key, **options)
    if method == "exact_regression":
        try:
            return regression_posterior(problem, key, **options)
        except NotApplicable as reason:
            raise NotApplicable(f"improper priors need the exact regression formula, which does not apply here: {reason}") from None
    raise ValueError(f"unknown method {method!r}: use auto, smc, nuts or exact_regression")
