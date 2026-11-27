"""The evidence (marginal likelihood), p(data | M): the likelihood averaged over the prior, which is the normalising constant
in Bayes' theorem. A separate, optional step after sampling, by one of two methods:

- `stepping_stones`: from a tempering ladder's record (`mutatis.infer.mcmc`). The ratio of the normalising constants of
  neighbouring rungs k and k+1 is estimated by a geometric bridge: the average of exp(+dbeta/2 x ell) over rung k's draws,
  divided by the average of exp(-dbeta/2 x ell) over rung k+1's draws, where ell = log(prior x likelihood) - log(reference).
  Both averages have finite variance whatever the rungs' spacing (the one-sided "backward" average, exp(-dbeta x ell) over
  rung k+1, often does not), and both see only places where the likelihood is
  positive (decision 40). The product of the ratios along the ladder runs from the reference (normalised, constant 1) to
  the posterior.
- `bridge`: from posterior draws alone (bridge sampling; Meng and Wong 1996, as in Gronau and others 2017). A normal
  distribution is fitted to half the draws; the evidence follows from comparing the posterior with it at the other half and
  at draws from the normal distribution, by the iteration that minimises the estimator's error.

Standard errors come from the variance of the averaged quantities, with each one's effective sample size (for stepping
stones, the delta method summed over the ladder; for bridge sampling, Fruhwirth-Schnatter's 2004 approximation), not from
the spread of a handful of replicas, which is itself too noisy to rely on.

The evidence is defined only for proper priors; with flat or Jeffreys priors it is arbitrary, so both methods refuse.
"""

from dataclasses import dataclass

import blackjax.diagnostics
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from mutatis import _jax_config  # noqa: F401


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    log_value: float
    standard_error: float  # of log_value; nan with a single replica
    method: str


def evidence(problem, posterior, key, method: str = "auto") -> Evidence:
    if not getattr(problem, "proper", True):
        raise EvidenceError("the evidence is not defined with an improper (flat or Jeffreys) prior")
    ladder = posterior.diagnostics.get("ladder")
    if method == "auto":
        method = "stepping_stones" if ladder is not None else "bridge"
    if method == "stepping_stones":
        if ladder is None:
            raise EvidenceError("stepping stones need a tempering ladder; this posterior has none (use bridge)")
        return stepping_stones(ladder.betas, ladder.ell)
    if method == "bridge":
        chains = posterior.diagnostics.get("chains")
        if chains is None:
            chains = posterior.resample(key, len(posterior.weights))[None]
        log_target = jax.jit(jax.vmap(lambda z: problem.log_prior(z) + problem.log_likelihood(z)))
        return bridge(log_target, np.asarray(chains), key)
    raise EvidenceError(f"unknown method {method!r}: use auto, stepping_stones or bridge")


def _log_mean_exp(x, axis=None):
    x = jnp.asarray(x)
    n = x.size if axis is None else x.shape[axis]
    return logsumexp(x, axis=axis) - np.log(n)


def _relative_variance(log_w: np.ndarray) -> float:
    """Variance of the mean of exp(log_w) relative to its square, from draws (chains, draws) with their autocorrelation."""
    log_w = np.asarray(log_w, dtype=float)
    w = np.exp(log_w - np.max(log_w))
    mean = w.mean()
    if mean == 0.0:
        return np.inf
    ess = float(blackjax.diagnostics.effective_sample_size(jnp.asarray(w))) if np.var(w) > 0 else float(w.size)
    return float(np.var(w) / mean**2 / max(min(ess, w.size), 1.0))


def stepping_stones(betas, ell) -> Evidence:
    """betas: (rungs,) from 0 to 1; ell: (replicas, draws, rungs), log(prior x likelihood) - log(reference) at each rung's draws."""
    betas, ell = np.asarray(betas), np.asarray(ell, dtype=float)
    if betas[0] != 0.0 or betas[-1] != 1.0:
        raise EvidenceError("the ladder must run from beta = 0 (the reference) to beta = 1 (the posterior)")
    log_value, variance = 0.0, 0.0
    for k, dbeta in enumerate(np.diff(betas)):
        up, down = 0.5 * dbeta * ell[:, :, k], -0.5 * dbeta * ell[:, :, k + 1]
        log_value += float(_log_mean_exp(up) - _log_mean_exp(down))
        variance += _relative_variance(up) + _relative_variance(down)
    return Evidence(log_value, float(np.sqrt(variance)), "stepping_stones")


def _bridge_one(log_target, draws: np.ndarray, key, tolerance=1e-10, max_iterations=1000):
    """Bridge sampling from posterior draws (chains, draws, unknowns): the normal distribution is fitted to the first half of
    every chain, and the second halves are used to estimate. Returns the log evidence and its standard error."""
    R, n, d = draws.shape
    fit, use = draws[:, : n // 2].reshape(-1, d), draws[:, n // 2:]
    mean = np.mean(fit, axis=0)
    cov = np.atleast_2d(np.cov(fit, rowvar=False)) + 1e-9 * np.eye(d)
    chol = np.linalg.cholesky(cov)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))

    def log_normal(z):
        u = jax.scipy.linalg.solve_triangular(jnp.asarray(chol), (z - mean).T, lower=True)
        return -0.5 * jnp.sum(u**2, axis=0) - 0.5 * log_det - 0.5 * d * np.log(2 * np.pi)

    flat = jnp.asarray(use.reshape(-1, d))
    n1 = n2 = flat.shape[0]
    proposal = jnp.asarray(mean) + jax.random.normal(key, (n2, d)) @ jnp.asarray(chol).T
    l1 = np.asarray(log_target(flat) - log_normal(flat))              # posterior draws: log(target / normal)
    l2 = np.asarray(log_target(proposal) - log_normal(proposal))      # normal draws
    l1, l2 = np.where(np.isnan(l1), -np.inf, l1), np.where(np.isnan(l2), -np.inf, l2)
    ess = float(np.min(np.asarray(jax.vmap(blackjax.diagnostics.effective_sample_size, in_axes=2)(jnp.asarray(use)))))
    n1_eff = min(ess, n1)
    log_s1, log_s2 = np.log(n1_eff / (n1_eff + n2)), np.log(n2 / (n1_eff + n2))
    shift = float(np.median(l1))                                    # keeps the exponentials in range
    l1, l2 = l1 - shift, l2 - shift
    log_r = 0.0
    for _ in range(max_iterations):
        numerator = _log_mean_exp(l2 - np.logaddexp(log_s1 + l2, log_s2 + log_r))
        denominator = _log_mean_exp(-np.logaddexp(log_s1 + l1, log_s2 + log_r))
        new = float(numerator - denominator)
        if abs(new - log_r) < tolerance:
            log_r = new
            break
        log_r = new
    # Fruhwirth-Schnatter (2004): relative mean squared error of the estimated ratio
    f2 = np.exp(l2 - np.logaddexp(log_s1 + l2, log_s2 + log_r))
    f1 = np.exp(-np.logaddexp(log_s1 + l1, log_s2 + log_r))
    rel_mse = np.var(f2) / np.mean(f2) ** 2 / n2 + (n1 / n1_eff) * np.var(f1) / np.mean(f1) ** 2 / n1
    return log_r + shift, float(np.sqrt(rel_mse))


def bridge(log_target, chains: np.ndarray, key) -> Evidence:
    """chains: (replicas, draws, unknowns) of posterior draws; log_target: vectorised log(prior x likelihood)."""
    log_value, se = _bridge_one(log_target, np.asarray(chains), key)
    return Evidence(log_value, se, "bridge")
