"""Importing a Dynare model file (.mod) as a specification, through Dynare's own preprocessor.

The preprocessor is run as a separate program (it is GPL; nothing of it is included here) with `json=parse`, and its JSON is
read: variables, shocks, parameters and their values, the model's equations, the steady-state model (or initial values), the
shocks' standard deviations, the observed variables and their measurement errors, and the estimated parameters' priors.

How Dynare's file becomes a specification:
- One block, named after the file, holds every equation: a Dynare file does not say which equation determines which variable.
- Dynare's leads, x(1), become expectations, E(x(+1)): a Dynare equation holds in expectation, and to first order the
  expectation of a function of next period's values is the function of their expectations, so the first-order solution is
  unchanged. Leads of more than one period are not supported yet.
- Parameters are fixed at the values the file gives them, except those in `estimated_params`, which get its priors (beta,
  gamma, normal, uniform, and the inverse gamma for standard deviations).
- `steady_state_model` (or `initval`) gives where finding the steady state starts.
- Each observed variable (`varobs`) is observed by a data series of the same name, with the measurement error the `shocks`
  block declares for it, if any.

The program is found from $MUTATIS_DYNARE_PREPROCESSOR, or as `dynare-preprocessor` on the path.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mutatis.spec.expr import Binary, Call, Name, Num, Shift, Unary, parse_expression, to_text
from mutatis.spec.model import Block, DataLink, Model, Parameter, Shock

PRIORS = {1: "beta", 2: "gamma", 3: "normal", 4: "inverse_gamma_sd", 5: "uniform"}   # Dynare's prior shape codes


class DynareImportError(ValueError):
    pass


def preprocessor() -> str:
    program = os.environ.get("MUTATIS_DYNARE_PREPROCESSOR") or shutil.which("dynare-preprocessor")
    if not program:
        raise DynareImportError("Dynare's preprocessor was not found: set MUTATIS_DYNARE_PREPROCESSOR to its path")
    return program


def read_json(mod_file: str | Path) -> dict:
    """Run the preprocessor on a copy of the file in a scratch folder and return its JSON (with the steady-state model)."""
    mod_file = Path(mod_file)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / mod_file.name
        copy.write_text(mod_file.read_text())
        done = subprocess.run([preprocessor(), copy.name, "json=parse", "onlyjson"], cwd=scratch, capture_output=True, text=True)
        folder = Path(scratch) / mod_file.stem / "model" / "json"
        if done.returncode != 0 or not (folder / "modfile.json").exists():
            raise DynareImportError(f"Dynare's preprocessor failed on {mod_file.name}:\n{done.stdout[-2000:]}{done.stderr[-2000:]}")
        content = json.loads((folder / "modfile.json").read_text())
        steady = folder / "steady_state_model.json"
        if steady.exists():
            content["steady_state_model"] = json.loads(steady.read_text())["steady_state_model"]
        return content


def _expectations(expr):
    """Dynare's x(1) as E(x(+1)), everywhere in an expression tree."""
    if isinstance(expr, Shift) and _lead(expr) > 0:
        if _lead(expr) > 1:
            raise DynareImportError(f"leads of more than one period ({to_text(expr)}) are not supported yet")
        return Call("E", (expr,))
    if hasattr(expr, "__dataclass_fields__"):
        changes = {}
        for name in expr.__dataclass_fields__:
            value = getattr(expr, name)
            if isinstance(value, tuple):
                changes[name] = tuple(_expectations(v) if hasattr(v, "__dataclass_fields__") else v for v in value)
            elif hasattr(value, "__dataclass_fields__"):
                changes[name] = _expectations(value)
        if changes:
            from dataclasses import replace
            return replace(expr, **changes)
    return expr


def _lead(shift: Shift) -> int:
    lag = shift.lag
    if isinstance(lag, Num):
        return int(lag.value)
    if isinstance(lag, Unary) and lag.op == "-" and isinstance(lag.operand, Num):
        return -int(lag.operand.value)
    return 0


def _text(dynare_expression: str) -> str:
    return to_text(_expectations(parse_expression(dynare_expression)))


def _number(text: str) -> float | None:
    value = float(text)
    return None if value != value else value      # NaN means "not given"


def _inverse_gamma_sd(mean: float, sd: float) -> tuple[float, float]:
    """(s, nu) of the inverse gamma for a standard deviation with this mean and standard deviation, as Dynare specifies it."""
    from scipy.optimize import brentq
    from scipy.special import gammaln

    def mean_of(s, nu):
        return s * math.sqrt(nu / 2) * math.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))

    def gap(nu):   # sd / mean depends on nu alone
        m = mean_of(1.0, nu)
        return math.sqrt(nu / (nu - 2) - m**2) / m - sd / mean
    nu = brentq(gap, 2.0 + 1e-9, 1e6)
    return mean / mean_of(1.0, nu), nu


def _prior(entry: dict) -> str:
    shape = PRIORS.get(entry["prior_distribution"])
    mean, sd, p3, p4 = (_number(entry[k]) for k in ("mean", "std", "p3", "p4"))
    if shape is None:
        raise DynareImportError(f"{entry['param']}: Dynare prior shape {entry['prior_distribution']} is not supported yet")
    if shape == "uniform":
        low, high = (p3, p4) if p3 is not None else (mean - 3**0.5 * sd, mean + 3**0.5 * sd)
        return f"uniform({low!r}, {high!r})"
    if shape == "normal":
        return f"normal({mean!r}, {sd!r})"
    if shape == "inverse_gamma_sd":
        s, nu = _inverse_gamma_sd(mean, sd)
        return f"inverse_gamma_sd(s={s!r}, nu={nu!r})"
    if p3 is not None or p4 is not None:
        raise DynareImportError(f"{entry['param']}: shifted or bounded {shape} priors (p3, p4) are not supported yet")
    return f"{shape}(mean={mean!r}, sd={sd!r})"


def import_dynare(mod_file: str | Path, name: str | None = None) -> Model:
    """A specification from a Dynare model file."""
    content = read_json(mod_file)
    if content.get("aux_vars"):
        raise DynareImportError("the file needs auxiliary variables (leads or lags beyond one period, or expectation operators); not supported yet")
    endogenous = [v["name"] for v in content["endogenous"]]
    exogenous = [v["name"] for v in content["exogenous"]]
    statements = content["statements"]
    values = {s["name"]: s["value"] for s in statements if s.get("statementName") == "param_init"}
    estimated = {p["param"]: p for s in statements if s.get("statementName") == "estimated_params" for p in s["params"]}
    stderr = {}
    for s in statements:
        if s.get("statementName") == "shocks":
            if s["variance"] or s["covariance"] or s["correlation"]:
                raise DynareImportError("shocks given by variances, covariances or correlations are not supported yet; give standard deviations (stderr)")
            stderr.update({e["name"]: e["stderr"] for e in s["stderr"]})

    parameters = {}
    for p in (v["name"] for v in content["parameters"]):
        if p in estimated:
            parameters[p] = Parameter(prior=_prior(estimated[p]))
        elif p in values:
            parameters[p] = Parameter(fixed=_constant(parse_expression(values[p]), values, p))
        else:
            raise DynareImportError(f"parameter {p} has neither a value nor a prior")

    missing = [e for e in exogenous if e not in stderr]
    if missing:
        raise DynareImportError(f"no standard deviation for the shocks {missing} (give `var {missing[0]}; stderr ...;` in the shocks block)")
    shocks = {e: Shock(f"normal(0, {_text(stderr[e])})") for e in exogenous}
    equations = tuple(f"{_text(eq['lhs'])} = {_text(eq['rhs'])}" for eq in content["model"])
    steady = {}
    for entry in content.get("steady_state_model") or []:
        steady[entry["lhs"]] = _text(entry["rhs"])
    if not steady:
        for s in statements:
            if s.get("statementName") == "initval":
                steady.update({v["name"]: _text(v["value"]) for v in s["vals"]})
    block = Block(outputs=tuple(endogenous), shocks=shocks, equations=equations, steady_state=steady)

    observed = content.get("varobs") or []
    errors = {v: stderr[v] for v in observed if v in stderr and v not in exogenous}
    data = {v: DataLink(observes=v, error=None if v not in errors else f"normal(0, {_text(errors[v])})") for v in observed}
    model_name = name or Path(mod_file).stem
    return Model(name=model_name, time="quarterly", sets={}, expectations="rational", parameters=parameters, blocks={model_name: block},
                 data=data)


def _constant(expr, values: dict, name: str) -> float:
    """A parameter's value, written as a number or an expression in numbers and other parameters' values."""
    if isinstance(expr, Num):
        return float(expr.value)
    if isinstance(expr, Name) and expr.id in values:
        return _constant(parse_expression(values[expr.id]), values, expr.id)
    if isinstance(expr, Unary) and expr.op == "-":
        return -_constant(expr.operand, values, name)
    if isinstance(expr, Binary) and expr.op in "+-*/^":
        a, b = _constant(expr.left, values, name), _constant(expr.right, values, name)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "^": a ** b}[expr.op]
    if isinstance(expr, Call) and expr.func in ("log", "exp", "sqrt") and len(expr.args) == 1:
        return float(getattr(math, expr.func)(_constant(expr.args[0], values, name)))
    raise DynareImportError(f"the value of {name}, {to_text(expr)}, is not a number or an expression in other parameters' values")
