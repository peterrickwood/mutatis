"""Running a version of a model, storing the result, and comparing results.

A run is keyed by everything that determines it: the specification (after edits), the data, the method settings and the random
seed. Its folder holds:

    spec.yaml       the specification that was estimated
    run.json        the key, the edits applied, a fingerprint of the data, the settings, and the method used
    posterior.npz   the weighted draws of the unconstrained unknowns
    summary.json    per-parameter posterior summaries, and the log marginal likelihood when defined
"""

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.estimate import estimate
from mutatis.infer.posterior import Posterior
from mutatis.infer.regression import regression_posterior
from mutatis.infer.smc import smc
from mutatis.measure.data import Dataset
from mutatis.spec.diff import diff
from mutatis.spec.edits import Edit, apply_edits
from mutatis.spec.model import Model
from mutatis.spec.yaml_io import dump_model, load_model, model_to_dict
from mutatis.versions.pipeline import prepare

SUMMARY_DRAWS = 4000


@dataclass(frozen=True)
class Result:
    folder: Path
    model: Model
    run: dict
    summary: dict


def data_fingerprint(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(dataset.dates)).encode())
    for name in sorted(dataset.series):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(dataset.series[name], dtype=float).tobytes())
    return digest.hexdigest()


def run_key(model: Model, dataset: Dataset, settings: dict, seed: int) -> str:
    payload = json.dumps({"model": model_to_dict(model), "data": data_fingerprint(dataset), "settings": settings, "seed": seed}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def summarise(problem, posterior: Posterior, key) -> dict:
    draws = posterior.resample(key, SUMMARY_DRAWS)
    values = jax.vmap(problem.parameters)(draws)
    parameters = {}
    for name, info in problem.system.parameters.items():
        if info.kind == "fixed":
            parameters[name] = {"fixed": np.asarray(info.fixed_value).tolist()}
            continue
        samples = np.asarray(values[name]).reshape(SUMMARY_DRAWS, -1)
        labels = _element_labels(name, info.axes, problem.system.sets)
        for i, label in enumerate(labels):
            column = samples[:, i]
            parameters[label] = {"mean": float(column.mean()), "sd": float(column.std()), "q05": float(np.quantile(column, 0.05)),
                                 "q50": float(np.quantile(column, 0.5)), "q95": float(np.quantile(column, 0.95)), "prior": info.spec.prior}
    return {"parameters": parameters, "log_marginal_likelihood": posterior.log_marginal_likelihood, "method": posterior.method,
            "route": problem.route, "start": problem.start, "draws": int(len(posterior.weights))}


def _element_labels(name, axes, sets):
    combos = [()]
    for axis in axes:
        combos = [c + (str(e),) for c in combos for e in sets[axis]]
    return [f"{name}[{', '.join(c)}]" if c else name for c in combos]


def run(model: Model, dataset: Dataset, store: str | Path = "results", edits: list[Edit] = (), seed: int = 0, route: str = "auto",
        method: str = "auto", **options) -> Result:
    edited = apply_edits(model, edits)
    problem = prepare(edited, dataset, route=route)
    method = ("smc" if problem.proper else "exact_regression") if method == "auto" else method
    sampler = {"smc": smc, "exact_regression": regression_posterior}[method]
    defaults = {k: v.default for k, v in inspect.signature(sampler).parameters.items() if v.default is not inspect.Parameter.empty}
    settings = {"route": problem.route, "start": problem.start, "method": method, **defaults, **options}  # every setting, defaults included
    folder = Path(store) / run_key(edited, dataset, settings, seed)
    if (folder / "summary.json").exists():
        return load_result(folder)
    key_estimate, key_summary = jax.random.split(jax.random.key(seed))
    posterior = estimate(problem, key_estimate, method=method, **options)
    summary = summarise(problem, posterior, key_summary)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.yaml").write_text(dump_model(edited))
    record = {"key": folder.name, "model": edited.name, "base_model": model.name, "edits": [e.to_dict() for e in edits],
              "data": data_fingerprint(dataset), "data_dates": [dataset.dates[0], dataset.dates[-1]], "settings": settings, "seed": seed,
              "method": posterior.method, "route": problem.route}
    (folder / "run.json").write_text(json.dumps(record, indent=2, default=str))
    np.savez(folder / "posterior.npz", particles=posterior.particles, weights=posterior.weights)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return Result(folder, edited, record, summary)


def load_result(folder: str | Path) -> Result:
    folder = Path(folder)
    return Result(folder, load_model((folder / "spec.yaml").read_text()), json.loads((folder / "run.json").read_text()),
                  json.loads((folder / "summary.json").read_text()))


def compare(a: Result, b: Result) -> str:
    """A plain-text comparison: the edits that turn a's specification into b's, then each parameter's posterior side by side."""
    lines = [f"Comparing {a.folder.name} ({a.run['method']}, {a.run['route']}) with {b.folder.name} ({b.run['method']}, {b.run['route']})", ""]
    try:
        edits = diff(a.model, b.model)
        lines.append("Edits from the first to the second:" if edits else "The specifications are the same.")
        lines += [f"  - {json.dumps(e.to_dict(), default=str)}" for e in edits]
    except ValueError as reason:
        lines.append(f"The specifications are not versions of one model: {reason}")
    lines.append("")
    if a.run["data"] != b.run["data"]:
        lines += ["Note: the two runs used different data.", ""]
    lines.append(f"{'parameter':<28} {'first: mean (sd)':>22} {'second: mean (sd)':>22} {'change':>10}")
    pa, pb = a.summary["parameters"], b.summary["parameters"]
    for name in list(pa) + [n for n in pb if n not in pa]:
        left, right = pa.get(name), pb.get(name)
        lines.append(f"{name:<28} {_cell(left):>22} {_cell(right):>22} {_change(left, right):>10}")
    la, lb = a.summary.get("log_marginal_likelihood"), b.summary.get("log_marginal_likelihood")
    lines.append("")
    if la is not None and lb is not None:
        lines.append(f"Log marginal likelihood: {la:.2f} versus {lb:.2f}. The data favour the {'second' if lb > la else 'first'} by {abs(lb - la):.2f} "
                     "(depends on how wide the priors are; see README, Open questions).")
    else:
        lines.append("Log marginal likelihood: not defined for at least one run (an improper prior).")
    return "\n".join(lines)


def _cell(entry) -> str:
    if entry is None:
        return "absent"
    if "fixed" in entry:
        return f"fixed {entry['fixed']}"
    return f"{entry['mean']:.3f} ({entry['sd']:.3f})"


def _change(left, right) -> str:
    if left is None or right is None or "mean" not in left or "mean" not in right:
        return ""
    return f"{right['mean'] - left['mean']:+.3f}"
