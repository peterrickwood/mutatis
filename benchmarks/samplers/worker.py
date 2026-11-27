"""One benchmark run: one method on one target with one seed, in its own process. Prints one line, "RESULT {json}".

    python -m benchmarks.samplers.worker --target normal_means_d10 --method adaptive --seed 0 --max-seconds 600
    python -m benchmarks.samplers.worker --target forward_looking --reference-only     (build and cache the reference)
"""

import argparse
import json
import os
import platform
import time
import traceback

# Pin this process to the CPUs the scheduler gave it, before JAX starts: XLA sizes its thread pool from the CPU affinity mask
# (see benchmarks/samplers/run.py for why the XLA thread flags are not used).
if os.environ.get("MUTATIS_BENCHMARK_CPUS"):
    os.sched_setaffinity(0, {int(c) for c in os.environ["MUTATIS_BENCHMARK_CPUS"].split(",")})
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jax.config.update("jax_compilation_cache_dir", os.path.join(REPO, ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)

from benchmarks.samplers import methods as m  # noqa: E402
from benchmarks.samplers.metrics import accuracy, evidence_accuracy  # noqa: E402
from benchmarks.samplers.targets import build  # noqa: E402

DIAGNOSTICS = ("max_rhat", "min_ess", "min_tail_ess", "divergences", "rungs", "barrier", "round_trips", "replicas", "processes",
               "gradient_evaluations", "density_evaluations", "converged", "status", "stopped_by", "phase", "findings", "reference",
               "step", "draws_per_chain", "starts_used", "peaks_found", "ladder_used", "ladder_reason", "escalated", "mass_matrix",
               "warmup", "timings", "draws_kept_per_chain", "thinned_by", "search", "peak_search")


def _plain(value):
    """Plain JSON values: numpy scalars and arrays as Python numbers and lists, non-finite numbers as None."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def target_record(target) -> dict:
    ref = target.reference
    return {"target": target.name, "group": target.group, "description": target.description, "size": target.size,
            "n_quantities": len(ref.names), "reference_source": ref.source, "true_log_evidence": ref.log_evidence,
            "has_peaks": ref.peak_weights is not None}


def run_one(target_name: str, method_name: str, seed: int, max_seconds: float = 600.0) -> dict:
    record = {"method": method_name, "seed": seed, "host": platform.node(), "cpus": sorted(os.sched_getaffinity(0))}
    target = build(target_name)
    record |= target_record(target)
    method = m.METHODS[method_name]
    key_run, key_evidence = jax.random.split(jax.random.key(seed))
    start = time.perf_counter()
    try:
        posterior = m.run(method, target, key_run, max_seconds)
    except Exception as error:
        return record | {"status": "error", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=4),
                         "seconds": time.perf_counter() - start, "posterior_ok": False, "confidently_wrong": False}
    record["seconds"] = time.perf_counter() - start
    diagnostics = posterior.diagnostics
    record["diagnostics"] = {k: _plain(diagnostics[k]) for k in DIAGNOSTICS if k in diagnostics}
    if "swap_acceptance" in diagnostics and diagnostics["swap_acceptance"] is not None and len(diagnostics["swap_acceptance"]):
        record["diagnostics"]["min_swap_acceptance"] = float(np.min(diagnostics["swap_acceptance"]))
    if method.kind == "smc":
        record["diagnostics"]["stages"] = len(diagnostics.get("temperatures", ())) - 1
    record["accuracy"] = accuracy(target, np.asarray(posterior.particles), np.asarray(posterior.weights))
    record["evidence"] = {}
    for name, estimate in m.evidence_estimates(method, target, posterior, key_evidence).items():
        if isinstance(estimate, str):
            record["evidence"][name] = {"error_message": estimate, "ok": False}
        else:
            value, se, seconds = estimate
            record["evidence"][name] = evidence_accuracy(value, se, target.reference.log_evidence) | {"seconds": seconds}
    # converged: the method's own convergence tests were met (None: the method has none, as SMC). A run that stopped at its
    # time limit is "not converged": its results are shown, flagged as suspect, and never counted as confidently wrong.
    checks_passed = record["diagnostics"].get("converged")
    ok = record["accuracy"]["posterior_ok"]
    record |= {"status": "ok", "posterior_ok": ok, "checks_passed": checks_passed, "not_converged": checks_passed is False,
               "confidently_wrong": (not ok) and (checks_passed is not False)}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="the MCMC methods' time limit")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.reference_only:
        start = time.perf_counter()
        target = build(args.target)
        print("RESULT " + json.dumps(_plain(target_record(target) | {"seconds": time.perf_counter() - start})), flush=True)
        return
    print("RESULT " + json.dumps(_plain(run_one(args.target, args.method, args.seed, args.max_seconds))), flush=True)


if __name__ == "__main__":
    main()
