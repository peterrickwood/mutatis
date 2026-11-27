"""The samplers compared, each a named setting of mutatis's own samplers, and how each one's evidence is computed.

- "default": `mutatis.infer.mcmc.mcmc` at its defaults, which choose from the data and the hardware (decisions 42 and 43): a
  peak search by optimisation, always a tempering ladder from the prior, a full mass matrix up to 200 unknowns, replicas from
  the CPUs available, and sampling until the convergence tests are met.
- "ladder": always a tempering ladder from the prior, Pathfinder starts and no peak search (the default of decision 41), with
  the same convergence-based stopping.
- "adaptive" (a variant, no longer main): the try-out of decision 42, a ladder only if the data call for one.
- "nuts": one rung, 4 chains from prior draws, the same stopping.
- "smc": `mutatis.infer.smc.smc` at its defaults (a fixed amount of work; it has no convergence test of its own).

Every MCMC method stops on its convergence tests or at the suite's time limit, never after a fixed number of draws.
Evidence: stepping stones and bridge sampling for runs with a ladder, bridge sampling for one-rung runs, SMC's own estimate
for SMC.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    kind: str                     # "mcmc" or "smc"
    options: dict = field(default_factory=dict)


METHODS = {m.name: m for m in [
    Method("default", "Default: peak search, then always a ladder from the prior, run to convergence", "mcmc"),
    Method("ladder", "Ladder always, no peak search (decision 41's default): Pathfinder starts, run to convergence", "mcmc",
           {"rungs": "auto", "starts": "pathfinder"}),
    Method("nuts", "NUTS: one rung, 4 chains from prior draws, run to convergence", "mcmc", {"rungs": 1, "starts": "prior", "replicas": 4}),
    Method("smc", "SMC: tempered, 2,000 particles, 5 HMC moves of 5 steps", "smc"),
    Method("adaptive", "Adaptive (decision 42's try-out): peak search, a ladder only if needed, run to convergence", "mcmc", {"rungs": "decide"}),
    Method("adaptive_prior_starts", "Adaptive with prior-draw starts (no peak search; the ladder only by escalation)", "mcmc",
           {"rungs": "decide", "starts": "prior"}),
    Method("adaptive_hmc", "Adaptive with HMC steps (random length) instead of NUTS", "mcmc", {"rungs": "decide", "step": "hmc"}),
    Method("adaptive_random_walk", "Adaptive with random-walk steps instead of NUTS", "mcmc", {"rungs": "decide", "step": "random_walk"}),
    Method("ladder_fitted_reference", "Default, but the ladder from a normal fitted by Pathfinder instead of the prior", "mcmc",
           {"reference": "fitted"}),
]}
MAIN = ("default", "ladder", "nuts", "smc")


def run(method: Method, problem, key, max_seconds: float):
    """One run. MCMC methods stop at their convergence tests or after `max_seconds` (then flagged as not converged)."""
    if method.kind == "smc":
        from mutatis.infer.smc import smc
        return smc(problem, key, **method.options)
    from mutatis.infer.mcmc import mcmc
    return mcmc(problem, key, max_seconds=max_seconds, **method.options)


def evidence_estimates(method: Method, problem, posterior, key) -> dict:
    """Each applicable evidence estimate as {name: (log value, standard error, seconds)}, or {name: error message}."""
    import time

    if method.kind == "smc":
        return {"smc": (posterior.log_marginal_likelihood, None, 0.0)}
    from mutatis.infer.evidence import evidence
    names = ["stepping_stones", "bridge"] if (posterior.diagnostics.get("rungs") or 1) > 1 else ["bridge"]
    out = {}
    for name in names:
        start = time.perf_counter()
        try:
            result = evidence(problem, posterior, key, method=name)
            out[name] = (result.log_value, result.standard_error, time.perf_counter() - start)
        except Exception as error:  # an estimate that cannot be made is a result, recorded as such
            out[name] = f"{type(error).__name__}: {error}"
    return out
