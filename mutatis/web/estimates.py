"""What a stored run found, as plain data for display: each parameter's prior against its posterior, each data series against
what the model predicted, and the reported quantities. Everything comes from the run's folder in the results store (its
specification, data, draws and summary); nothing here knows what kind of model it is.

Prior against posterior, per element of each parameter, on one axis:

- the posterior is a histogram of the stored draws, mapped to the parameter's own values;
- the prior is its density on a grid when the prior acts element by element and needs no other parameter's value; otherwise
  (a prior over a whole matrix, such as the even distribution over rotations, or one that depends on other parameters) a
  histogram of draws from it. Improper priors (flat, Jeffreys) have no density to draw. Fixed parameters are listed with
  their values; a parameter with no prior is reported as ranges, not draws.

The axis covers the posterior's draws and, unless it is much wider, the bulk of the prior.

Data against fit: each observed series against the model's one-step-ahead prediction of it (`infer.states.one_step_predictions`)
at a set of posterior draws, as a median and a 90% band. Series that feed the model or are used only by restrictions are shown
as data alone.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.system import compile_model
from mutatis.dist.domains import Interval, Orthogonal, Positive, PositiveDefinite
from mutatis.infer.problem import PosteriorProblem
from mutatis.infer.states import one_step_predictions
from mutatis.measure.data import Dataset
from mutatis.quantities.expressions import QuantityError, _horizons
from mutatis.spec.expr import Call, ParseError, parse_expression, walk
from mutatis.spec.latex import distribution_latex, text_to_latex
from mutatis.versions.run import Result, _element_labels, _equal_weight_draws, load_data, load_posterior, problem_of

BINS = 40
GRID = 121
PRIOR_DRAWS = 4000
FIT_DRAWS = 200
BAND = (0.05, 0.95)
WIDER = 12.0   # the prior's bulk is left off the axis when it is this many times wider than the posterior's


@dataclass
class Loaded:
    """A stored run with what is needed to read its draws: its data, its posterior problem rebuilt, and equally weighted draws."""

    result: Result
    dataset: Dataset | None
    problem: PosteriorProblem | None
    draws: np.ndarray | None          # (draws, unknowns), unconstrained, equally weighted
    missing: str | None = None        # why there are no draws to read, if so


def load(result: Result) -> Loaded:
    dataset = load_data(result)
    if dataset is None:
        return Loaded(result, None, None, None, "this run was stored before runs kept their data: only its summary can be shown; run it again to see more")
    problem = problem_of(result, dataset)
    draws = np.asarray(_equal_weight_draws(load_posterior(result), jax.random.key(0)))
    return Loaded(result, dataset, problem, draws)


# Prior against posterior ------------------------------------------------------------------------------------------------------

def parameter_panels(loaded: Loaded) -> dict:
    result = loaded.result
    system = loaded.problem.system if loaded.problem is not None else compile_model(result.model)
    values = jax.vmap(loaded.problem.parameters)(jnp.asarray(loaded.draws)) if loaded.problem is not None else {}
    fixed = system.fixed_parameters()
    out = []
    for index, (key, info) in enumerate(system.parameters.items()):
        entry = {"key": key, "owner": key.rpartition(".")[0].split("[")[0] or None, "kind": info.kind, "axes": list(info.axes),
                 "shape": list(info.shape), "prior": info.spec.prior, "prior_latex": _distribution(info.spec.prior), "elements": []}
        if info.kind == "fixed":
            entry["fixed"] = np.asarray(info.fixed_value).tolist()
            out.append(entry)
            continue
        if info.kind == "no_prior":
            entry["note"] = "restricted, with no prior: what the data cannot decide is reported as ranges, not draws"
            out.append(entry)
            continue
        prior, note = _prior(key, info, fixed, loaded.problem, jax.random.fold_in(jax.random.key(1), index))
        entry["prior_note"] = note
        labels = _element_labels(key, info.axes, system.sets)
        draws = np.asarray(values[key]).reshape(len(loaded.draws), -1) if key in values else None
        for i, label in enumerate(labels):
            entry["elements"].append(_panel(label, None if draws is None else draws[:, i], prior, i, info, result.summary["parameters"].get(label)))
        out.append(entry)
    return {"parameters": out, "draws": None if loaded.draws is None else len(loaded.draws), "note": loaded.missing}


def _distribution(text):
    if text is None:
        return None
    try:
        return distribution_latex(text)
    except (ParseError, TypeError):
        return None


def _prior(key, info, fixed, problem, rng):
    """('density', distribution, draws) for an element-by-element prior that needs no estimated parameter (the draws only set
    the axis); ('draws', draws) for any other proper prior; or None, with the reason. Draws are (draws, elements)."""
    spec = info.prior
    if not spec.proper:
        return None, f"{spec.family}: not a probability distribution, so there is no prior density to draw"
    try:
        dist = spec.build(fixed)
    except KeyError:   # its arguments include parameters that are estimated: draw it with them
        if problem is None or not problem.proper:
            return None, "this prior depends on other parameters, which cannot all be drawn from their priors"
        z = jax.vmap(problem.draw_prior)(jax.random.split(rng, PRIOR_DRAWS))
        return ("draws", np.asarray(jax.vmap(problem.parameters)(z)[key]).reshape(PRIOR_DRAWS, -1)), None
    if dist.event_shape:
        return ("draws", np.asarray(dist.sample(rng, (PRIOR_DRAWS,))).reshape(PRIOR_DRAWS, -1)), None
    sample = dist.sample(rng, (PRIOR_DRAWS,))
    if tuple(sample.shape[1:]) != info.shape:
        sample = dist.sample(rng, (PRIOR_DRAWS, *info.shape))
    return ("density", dist, np.asarray(sample).reshape(PRIOR_DRAWS, -1)), None


def _panel(label, posterior, prior, i, info, summary) -> dict:
    prior_sample = None if prior is None else prior[-1][:, i]
    lower, upper, prior_wider = _window(posterior, prior_sample, _bounds(info, i))
    edges = np.linspace(lower, upper, BINS + 1)
    panel = {"label": label, "window": [lower, upper], "bins": edges.tolist(), "prior_wider": prior_wider, **(summary or {})}
    panel.pop("prior", None)
    if posterior is not None:
        posterior = posterior[np.isfinite(posterior)]
        counts, _ = np.histogram(posterior, bins=edges)
        panel["posterior"] = (counts / (len(posterior) * (edges[1] - edges[0]))).tolist()
        panel["outside"] = float(np.mean((posterior < lower) | (posterior > upper)))
    if prior is not None and prior[0] == "density":
        grid = np.linspace(lower, upper, GRID)
        log_density = prior[1].log_prob(jnp.asarray(grid).reshape((-1,) + (1,) * len(info.shape)))
        density = np.exp(np.asarray(jnp.broadcast_to(log_density, (GRID, *info.shape))).reshape(GRID, -1)[:, i])
        panel["prior"] = {"x": grid.tolist(), "density": np.where(np.isfinite(density), density, 0.0).tolist()}
    elif prior is not None:
        counts, _ = np.histogram(prior_sample, bins=edges)
        panel["prior"] = {"bins": edges.tolist(), "density": (counts / (len(prior_sample) * (edges[1] - edges[0]))).tolist()}
    return panel


def _bounds(info, i) -> tuple[float, float]:
    """The values one element of a parameter can take: from its domain (a variance on a covariance matrix's diagonal is positive)."""
    domain = info.domain
    if isinstance(domain, Positive):
        return 0.0, np.inf
    if isinstance(domain, Interval):
        return float(domain.low), float(domain.high)
    if isinstance(domain, Orthogonal):
        return -1.0, 1.0
    if isinstance(domain, PositiveDefinite) and len(info.shape) == 2 and i // info.shape[1] == i % info.shape[1]:
        return 0.0, np.inf
    return -np.inf, np.inf


def _window(posterior, prior_sample, bounds) -> tuple[float, float, bool]:
    """The axis: the posterior's draws with a margin, widened to the central 96% of the prior unless that is much wider, and kept
    within the values the element can take."""
    prior_wider = False
    spread = None
    if posterior is not None and np.isfinite(posterior).any():
        lower, upper = np.quantile(posterior[np.isfinite(posterior)], [0.001, 0.999])
        spread = upper - lower if upper > lower else max(abs(lower), 1.0) * 0.1
        lower, upper = lower - 0.15 * spread, upper + 0.15 * spread
    if prior_sample is not None and np.isfinite(prior_sample).any():
        p_lower, p_upper = np.quantile(prior_sample[np.isfinite(prior_sample)], [0.02, 0.98])
        if spread is None:
            lower, upper = p_lower, p_upper
        elif p_upper - p_lower <= WIDER * spread:
            lower, upper = min(lower, p_lower), max(upper, p_upper)
        else:
            prior_wider = True
    if spread is None and prior_sample is None:
        lower, upper = -1.0, 1.0
    lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper), prior_wider


# Data against fit -------------------------------------------------------------------------------------------------------------

BASIS = {
    "kalman": "the model's prediction of each period from the data up to the period before (the Kalman filter)",
    "read_off": "the model's prediction of each period from the data up to the period before (read exactly from the data)",
    "sample": "the model's prediction of each period from the state one period before, as the sampled shocks imply it (drawn given all the data)",
}


def fit_panels(loaded: Loaded) -> dict:
    if loaded.problem is None:
        return {"note": loaded.missing, "observed": [], "fed": [], "auxiliary": [], "dates": []}
    problem, system = loaded.problem, loaded.problem.system
    chosen = loaded.draws[np.unique(np.linspace(0, len(loaded.draws) - 1, FIT_DRAWS).astype(int))]
    predicted = np.asarray(jax.jit(jax.vmap(lambda z: one_step_predictions(problem, z)))(jnp.asarray(chosen)))
    usable = np.all(np.isfinite(predicted), axis=(1, 2))
    first = problem.data.start.first if problem.start == "data" else 0
    dates = list(problem.data.dates[first:])
    links = loaded.result.model.data
    observed = []
    for column, observation in enumerate(system.observations):
        entry = {"series": observation.series, "exact": observation.error is None, "transform": links[observation.series].transform,
                 "data": _list(problem.observed[:, column])}
        if usable.any():
            low, middle, high = np.quantile(predicted[usable][:, :, column], [BAND[0], 0.5, BAND[1]], axis=0)
            entry.update(median=_list(middle), lower=_list(low), upper=_list(high))
        observed.append(entry)
    fed = []
    for info in system.outside.values():
        for position in system.layout.outside[info.key].positions.reshape(-1):
            label = system.outside_labels[position]
            if label in info.series:
                fed.append({"series": info.series[label], "variable": label, "transform": links[info.series[label]].transform,
                            "data": _list(problem.outside[:, position])})
    auxiliary = [{"series": name, "transform": links[name].transform, "data": _list(values[first:])} for name, values in problem.data.auxiliary.items()]
    return {"dates": dates, "route": problem.route, "basis": BASIS[problem.route], "band": list(BAND), "draws": int(usable.sum()),
            "draws_tried": len(chosen), "observed": observed, "fed": fed, "auxiliary": auxiliary, "note": None}


def _list(values) -> list:
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


# Reported quantities ----------------------------------------------------------------------------------------------------------

def quantity_panels(result: Result) -> list[dict]:
    """The reported quantities from the run's summary: per element, the mean and 68% interval; or, for ranges, the set of
    posterior means and the 68% robust interval."""
    return [quantity_panel(text, stats) for text, stats in result.summary.get("quantities", {}).items()]


def quantity_panel(text: str, stats: dict) -> dict:
    n = len(stats.get("mean", stats.get("set_mean_lower", [])))
    axis, label = list(range(n)), "element"
    try:
        expr = parse_expression(text)
        latex = text_to_latex(text)
        for node in walk(expr):
            if isinstance(node, Call) and "horizons" in dict(node.kwargs):
                horizons = _horizons(dict(node.kwargs)["horizons"])
                if len(horizons) == n:
                    axis, label = horizons, "horizon"
                break
    except (ParseError, QuantityError, TypeError):
        latex = None
    entry = {"quantity": text, "latex": latex, "axis": axis, "axis_label": label, "draws": stats.get("draws")}
    if "mean" in stats:
        entry.update(kind="draws", mean=stats["mean"], lower=stats["ci68_lower"], upper=stats["ci68_upper"])
    else:
        entry.update(kind="ranges", lower=stats["set_mean_lower"], upper=stats["set_mean_upper"], robust_lower=stats["robust68_lower"],
                     robust_upper=stats["robust68_upper"])
    return entry
