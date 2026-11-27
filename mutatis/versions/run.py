"""Running a version of a model, storing the result, and comparing results.

A run is keyed by everything that determines it: the specification (after edits), the data, the method settings and the random
seed. Its folder holds:

    spec.yaml       the specification that was estimated
    run.json        the key, the edits applied, a fingerprint of the data, the settings, and the method used
    posterior.npz   the weighted draws of the unconstrained unknowns
    data.npz        the data it was estimated on, exactly (dates, series names, values)
    summary.json    per-parameter posterior summaries, and the log marginal likelihood when defined
    quantities.npz  the draws (or, for ranges, the bounds) of each reported quantity
"""

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.estimate import SAMPLERS, choose_method, estimate
from mutatis.infer.intervals import bootstrap_standard_errors, robust_credible_interval, shortest_centred_interval
from mutatis.infer.posterior import Posterior
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


def summarise(problem, posterior: Posterior, key, report=()) -> dict:
    summary = _summarise_parameters(problem, posterior, key)
    summary["quantities"] = _summarise_quantities(problem, posterior, key, report)
    for name in ("plausibility", "max_rhat", "min_ess", "divergences", "round_trips", "converged", "status", "findings", "peak_search"):
        if name in posterior.diagnostics:
            summary[name] = posterior.diagnostics[name]
    return summary


def _summarise_quantities(problem, posterior, key, report) -> dict:
    """For each reported quantity, per element: the posterior mean, sd and 68% credible interval; or, when the rotation has no
    prior, the set of posterior means and the 68% robust credible interval, from the ranges computed with each draw."""
    out = {}
    bounds = posterior.diagnostics.get("bounds", {})
    for quantity in report:
        if quantity in bounds:
            lower, upper = (np.asarray(b) for b in bounds[quantity])
            robust = [robust_credible_interval(lower[:, i], upper[:, i]) for i in range(lower.shape[1])]
            out[quantity] = {"set_mean_lower": lower.mean(axis=0).tolist(), "set_mean_upper": upper.mean(axis=0).tolist(),
                             "robust68_lower": [r[0] for r in robust], "robust68_upper": [r[1] for r in robust],
                             "lower_sd": lower.std(axis=0).tolist(), "upper_sd": upper.std(axis=0).tolist(), "draws": int(len(lower))}
            continue
        values = np.asarray(jax.vmap(lambda z: _quantity_at(problem, z, quantity))(_equal_weight_draws(posterior, key)))
        values = values.reshape(len(values), -1)
        intervals = [shortest_centred_interval(values[:, i]) for i in range(values.shape[1])]
        out[quantity] = {"mean": values.mean(axis=0).tolist(), "sd": values.std(axis=0).tolist(), "ci68_lower": [c[0] for c in intervals],
                         "ci68_upper": [c[1] for c in intervals], "draws": int(len(values))}
    return out


def _equal_weight_draws(posterior, key):
    if np.allclose(posterior.weights, posterior.weights[0]):
        return jnp.asarray(posterior.particles)
    return jnp.asarray(posterior.resample(key, SUMMARY_DRAWS))


def _quantity_at(problem, z, quantity):
    from mutatis.quantities.expressions import Context, evaluate_quantity

    values, _ = problem.layout.constrain(z)
    params = {k: v for k, v in values.items() if k in problem.system.parameters}
    data = problem.data
    first = data.start.first if problem.start == "data" else 0
    series = {o.series: jnp.asarray(data.observed[first:, i]) for i, o in enumerate(problem.system.observations)}
    series.update({name: jnp.asarray(v[first:]) for name, v in data.auxiliary.items()})
    return evaluate_quantity(quantity, Context(problem.system, problem.solve(params), problem.observed, problem.outside, problem.start_state, series))


def compare_with_reference(result: "Result", path: str | Path, reference_draws: int = 1000, standard_errors: float = 5.0,
                           absolute_floor: float = 1e-6) -> list[dict]:
    """Each statistic in a reference file (quantity, element, statistic, value) against this result's.

    Both are Monte Carlo estimates. The standard error of each of this result's statistics is estimated by a bootstrap over its
    own stored draws; the reference's is the same scaled to its number of draws. The tolerance is `standard_errors` times the
    standard error of the difference, never below `absolute_floor` (a statistic pinned exactly by a restriction has no spread).
    """
    import csv

    report = result.run["settings"]["report"]
    stored = np.load(result.folder / "quantities.npz")
    errors_cache = {}

    def errors(quantity, statistic):
        key = (quantity, statistic)
        if key not in errors_cache:
            i = report.index(quantity)
            if f"q{i}" in stored:
                draws = stored[f"q{i}"]
                n = len(draws)
                fn = {"mean": lambda d: d.mean(axis=0),
                      "ci68_lower": lambda d: [shortest_centred_interval(d[:, j])[0] for j in range(d.shape[1])],
                      "ci68_upper": lambda d: [shortest_centred_interval(d[:, j])[1] for j in range(d.shape[1])]}[statistic]
                errors_cache[key] = (bootstrap_standard_errors(fn, [draws]), n)
            else:
                lower, upper = stored[f"q{i}_lower"], stored[f"q{i}_upper"]
                n = len(lower)
                fn = {"set_mean_lower": lambda lo, up: lo.mean(axis=0), "set_mean_upper": lambda lo, up: up.mean(axis=0),
                      "robust68_lower": lambda lo, up: [robust_credible_interval(lo[:, j], up[:, j])[0] for j in range(lo.shape[1])],
                      "robust68_upper": lambda lo, up: [robust_credible_interval(lo[:, j], up[:, j])[1] for j in range(lo.shape[1])]}[statistic]
                errors_cache[key] = (bootstrap_standard_errors(fn, [lower, upper], reps=50), n)
        return errors_cache[key]

    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            quantity, element, statistic, theirs = row["quantity"], int(row["element"]), row["statistic"], float(row["value"])
            ours_all = result.summary["quantities"].get(quantity, {})
            if statistic not in ours_all:
                continue
            ours = ours_all[statistic][element]
            se, n = errors(quantity, statistic)
            difference_se = se[element] * np.sqrt(1.0 + n / reference_draws)   # ours, plus the reference's at its number of draws
            tolerance = max(standard_errors * difference_se, absolute_floor)
            rows.append({"quantity": quantity, "element": element, "statistic": statistic, "ours": ours, "theirs": theirs,
                         "tolerance": tolerance, "ok": bool(abs(ours - theirs) <= tolerance)})
    return rows


def _summarise_parameters(problem, posterior: Posterior, key) -> dict:
    draws = posterior.resample(key, SUMMARY_DRAWS)
    values = jax.vmap(problem.parameters)(draws)
    parameters = {}
    for name, info in problem.system.parameters.items():
        if info.kind == "fixed":
            parameters[name] = {"fixed": np.asarray(info.fixed_value).tolist()}
            continue
        if info.kind == "no_prior":
            parameters[name] = {"no_prior": True}  # its draws are placeholders; its effect is reported as ranges
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
        method: str = "auto", report: tuple[str, ...] = (), **options) -> Result:
    edited = apply_edits(model, edits)
    problem = prepare(edited, dataset, route=route)
    method = choose_method(problem) if method == "auto" else method
    sampler = SAMPLERS[method]
    defaults = {k: v.default for k, v in inspect.signature(sampler).parameters.items() if v.default is not inspect.Parameter.empty}
    if method == "exact_regression":
        options = {**options, "report": tuple(report)}  # ranges are computed draw by draw, so the sampler needs the report
    settings = {"route": problem.route, "start": problem.start, "method": method, **defaults, **options, "report": list(report)}  # every setting
    folder = Path(store) / run_key(edited, dataset, settings, seed)
    if (folder / "summary.json").exists():
        return load_result(folder)
    posterior = estimate(problem, jax.random.split(jax.random.key(seed))[0], method=method, **options)
    return save_result(folder, model, edits, dataset, problem, posterior, settings, seed, report)


def save_result(folder: str | Path, model: Model, edits: list[Edit] | tuple[Edit, ...], dataset: Dataset, problem, posterior: Posterior, settings: dict, seed: int,
                report: tuple[str, ...] = ()) -> Result:
    """Store a posterior for the problem built from `model` with `edits` applied, on `dataset`: the files listed at the top."""
    folder = Path(folder)
    key_summary = jax.random.split(jax.random.key(seed))[1]
    summary = summarise(problem, posterior, key_summary, report)
    edited = apply_edits(model, edits)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.yaml").write_text(dump_model(edited))
    record = {"key": folder.name, "model": edited.name, "base_model": model.name, "edits": [e.to_dict() for e in edits],
              "data": data_fingerprint(dataset), "data_dates": [dataset.dates[0], dataset.dates[-1]], "settings": settings, "seed": seed,
              "method": posterior.method, "route": problem.route}
    (folder / "run.json").write_text(json.dumps(record, indent=2, default=str))
    np.savez(folder / "posterior.npz", particles=posterior.particles, weights=posterior.weights)
    np.savez(folder / "data.npz", dates=np.array(dataset.dates), names=np.array(list(dataset.series)),
             values=np.stack([np.asarray(v, dtype=float) for v in dataset.series.values()], axis=1))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(folder / "quantities.npz", **_quantity_draws(problem, posterior, key_summary, report))
    return Result(folder, edited, record, summary)


def _quantity_draws(problem, posterior, key, report) -> dict:
    """Per reported quantity (by position in the report): its draws, or the lower and upper bounds of each draw."""
    out = {}
    bounds = posterior.diagnostics.get("bounds", {})
    for i, quantity in enumerate(report):
        if quantity in bounds:
            out[f"q{i}_lower"], out[f"q{i}_upper"] = (np.asarray(b) for b in bounds[quantity])
        else:
            values = np.asarray(jax.vmap(lambda z: _quantity_at(problem, z, quantity))(_equal_weight_draws(posterior, key)))
            out[f"q{i}"] = values.reshape(len(values), -1)
    return out


def load_result(folder: str | Path) -> Result:
    folder = Path(folder)
    return Result(folder, load_model((folder / "spec.yaml").read_text()), json.loads((folder / "run.json").read_text()),
                  json.loads((folder / "summary.json").read_text()))


def load_data(result: Result) -> Dataset | None:
    """The data a run was estimated on, checked against its fingerprint; None for a run stored before runs kept their data."""
    path = result.folder / "data.npz"
    if not path.exists():
        return None
    with np.load(path) as stored:
        dates, names, values = stored["dates"], stored["names"], stored["values"]
    dataset = Dataset(tuple(str(d) for d in dates), {str(n): values[:, i].copy() for i, n in enumerate(names)})
    if data_fingerprint(dataset) != result.run["data"]:
        raise ValueError(f"{path} does not hold the data this run was estimated on (its fingerprint differs)")
    return dataset


def load_posterior(result: Result) -> Posterior:
    """The stored draws, as the sampler returned them (no diagnostics)."""
    with np.load(result.folder / "posterior.npz") as stored:
        return Posterior(stored["particles"], stored["weights"], result.summary.get("log_marginal_likelihood"), result.run["method"])


def problem_of(result: Result, dataset: Dataset):
    """The posterior problem a run was estimated on: its specification, its data, and the route and start it used."""
    return prepare(result.model, dataset, route=result.run["route"], start=result.run["settings"].get("start", "auto"))


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
    if "no_prior" in entry:
        return "no prior: ranges"
    if "fixed" in entry:
        return f"fixed {entry['fixed']}"
    return f"{entry['mean']:.3f} ({entry['sd']:.3f})"


def _change(left, right) -> str:
    if left is None or right is None or "mean" not in left or "mean" not in right:
        return ""
    return f"{right['mean'] - left['mean']:+.3f}"
