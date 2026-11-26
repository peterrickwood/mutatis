"""Choosing a sampler for a posterior problem.

With proper priors, sequential Monte Carlo, which applies to anything. With improper priors (flat or Jeffreys) SMC cannot start,
because it draws from the prior; then the exact regression formula, if the model qualifies. A method can also be forced.
"""

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.posterior import Posterior
from mutatis.infer.regression import NotApplicable, regression_posterior
from mutatis.infer.smc import smc


def estimate(problem, key, method: str = "auto", **options) -> Posterior:
    if method == "auto":
        method = "smc" if problem.proper else "exact_regression"
    if method == "smc":
        return smc(problem, key, **options)
    if method == "exact_regression":
        try:
            return regression_posterior(problem, key, **options)
        except NotApplicable as reason:
            raise NotApplicable(f"improper priors need the exact regression formula, which does not apply here: {reason}") from None
    raise ValueError(f"unknown method {method!r}: use auto, smc or exact_regression")
