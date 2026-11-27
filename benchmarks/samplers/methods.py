"""The samplers compared, each a named setting of mutatis's own samplers, and how each one's evidence is computed.

The default and its variants are settings of `mutatis.infer.mcmc.mcmc`; "nuts" is its one-rung setting; "smc" is
`mutatis.infer.smc.smc` with its defaults. Evidence: stepping stones and bridge sampling for runs with a ladder, bridge
sampling for one-rung runs, and SMC's own estimate for SMC.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    kind: str                     # "mcmc" or "smc"
    options: dict = field(default_factory=dict)


METHODS = {m.name: m for m in [
    Method("default", "Default: NUTS steps, tempering ladder, 4 replicas, Pathfinder starts", "mcmc"),
    Method("nuts", "NUTS: one rung, 4 chains started from prior draws", "mcmc", {"rungs": 1, "starts": "prior"}),
    Method("smc", "SMC: tempered, 2,000 particles, 5 HMC moves of 5 steps", "smc"),
    Method("prior_starts", "Default with prior-draw starts", "mcmc", {"starts": "prior"}),
    Method("fitted_reference", "Default with a fitted normal at the ladder's hot end", "mcmc", {"reference": "fitted"}),
    Method("hmc", "Default with HMC steps (random length) instead of NUTS", "mcmc", {"step": "hmc"}),
    Method("random_walk", "Default with random-walk steps instead of NUTS", "mcmc", {"step": "random_walk"}),
]}
MAIN = ("default", "nuts", "smc")


def run(method: Method, problem, key):
    if method.kind == "smc":
        from mutatis.infer.smc import smc
        return smc(problem, key, **method.options)
    from mutatis.infer.mcmc import mcmc
    return mcmc(problem, key, **method.options)


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
