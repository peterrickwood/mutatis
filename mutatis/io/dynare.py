"""Importing a Dynare model file (.mod) as a specification, through Dynare's own preprocessor.

The preprocessor is run as a separate program (it is GPL; nothing of it is included here) with `json=parse`, and its JSON is
read: variables, shocks, parameters and their values, the model's equations, the steady-state model (or initial values), the
shocks' standard deviations, the observed variables and their measurement errors, and the estimated parameters' priors.

How Dynare's file becomes a specification:
- One block, named after the file, holds every equation: a Dynare file does not say which equation determines which variable.
- Dynare's leads, x(1), become expectations, E(x(+1)): a Dynare equation holds in expectation, and to first order the
  expectation of a function of next period's values is the function of their expectations, so the first-order solution is
  unchanged. Leads of more than one period are not supported yet.
- Parameters are fixed at the values the file gives them, worked out in the file's order, except those in
  `estimated_params`, which get its priors (beta, gamma, normal, uniform, and the inverse gamma for standard deviations).
  Values the file computes in MATLAB code are supplied by the caller (`values`), e.g. from the authors' stored results.
- Estimated shock standard deviations (`stderr e, ...` in `estimated_params`) become parameters named `stderr_e`.
- Model-local variables (`#`) become the block's definitions. A declared shock that no equation uses is left out.
- `steady_state_model` (or `initval`) gives where finding the steady state starts.
- Each observed variable (`varobs`) is observed by a data series of the same name, with the measurement error the `shocks`
  block declares for it, if any.
- The model is read as it stands at the `estimation` statement: statements after it are ignored.
- Files included with `@#include` are read from the model file's folder; macro variables can be set (`defines`).

The program is found from $MUTATIS_DYNARE_PREPROCESSOR, or as `dynare-preprocessor` on the path.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mutatis.spec.expr import Binary, Call, Name, Num, Shift, Unary, parse_expression, to_text, walk
from mutatis.spec.model import Block, DataLink, Model, Parameter, Shock

PRIORS = {1: "beta", 2: "gamma", 3: "normal", 4: "inverse_gamma_sd", 5: "uniform"}   # Dynare's prior shape codes


class DynareImportError(ValueError):
    pass


def preprocessor() -> str:
    program = os.environ.get("MUTATIS_DYNARE_PREPROCESSOR") or shutil.which("dynare-preprocessor")
    if not program:
        raise DynareImportError("Dynare's preprocessor was not found: set MUTATIS_DYNARE_PREPROCESSOR to its path")
    return program


def read_json(mod_file: str | Path, defines: dict | None = None) -> dict:
    """Run the preprocessor on a copy of the file in a scratch folder and return its JSON (with the steady-state model). Files
    the model includes (`@#include`) are read from the file's own folder; `defines` sets macro variables (`-D`)."""
    mod_file = Path(mod_file)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / mod_file.name
        copy.write_text(mod_file.read_text())
        command = [preprocessor(), copy.name, "json=parse", "onlyjson", f"-I{mod_file.resolve().parent}", *(f"-D{k}={v}" for k, v in (defines or {}).items())]
        done = subprocess.run(command, cwd=scratch, capture_output=True, text=True)
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


def import_dynare(mod_file: str | Path, name: str | None = None, defines: dict | None = None, values: dict | None = None) -> Model:
    """A specification from a Dynare model file.

    The model is taken as it stands at the file's `estimation` statement, if it has one: later statements (calibration checks,
    simulations) are ignored. Parameter values are worked out in the file's order, as Dynare does; a parameter whose value the
    file computes outside Dynare's language (in MATLAB code) takes its number from `values`, for example the values stored in
    the authors' results file (`stored_parameter_values`)."""
    content = read_json(mod_file, defines)
    if content.get("aux_vars"):
        raise DynareImportError("the file needs auxiliary variables (leads or lags beyond one period, or expectation operators); not supported yet")
    endogenous = [v["name"] for v in content["endogenous"]]
    exogenous = [v["name"] for v in content["exogenous"]]
    statements = content["statements"]
    stop = next((i for i, s in enumerate(statements) if s.get("statementName") == "estimation"), len(statements))
    statements = statements[:stop]
    estimated, estimated_sd = {}, {}
    for s in statements:
        if s.get("statementName") == "estimated_params":
            for p in s["params"]:
                if "param" in p:
                    estimated[p["param"]] = p
                elif "var" in p and "var2" not in p:
                    estimated_sd[p["var"]] = p
                else:
                    raise DynareImportError(f"estimated correlations ({p}) are not supported yet")
    stderr = {}
    for s in statements:
        if s.get("statementName") == "shocks":
            if s["variance"] or s["covariance"] or s["correlation"]:
                raise DynareImportError("shocks given by variances, covariances or correlations are not supported yet; give standard deviations (stderr)")
            stderr.update({e["name"]: e["stderr"] for e in s["stderr"]})

    known = _parameter_values(statements, values or {}, set(estimated))
    parameters = {}
    for p in (v["name"] for v in content["parameters"]):
        if p in estimated:
            parameters[p] = Parameter(prior=_prior(estimated[p]))
        elif p in known:
            parameters[p] = Parameter(fixed=known[p])
        else:
            raise DynareImportError(f"parameter {p} has neither a value nor a prior")
    sd_name = {}
    for v, entry in estimated_sd.items():
        sd_name[v] = f"stderr_{v}"
        if sd_name[v] in parameters:
            raise DynareImportError(f"cannot name the standard deviation of {v}: a parameter is already called {sd_name[v]}")
        parameters[sd_name[v]] = Parameter(prior=_prior({**entry, "param": v}))

    equations = tuple(f"{_text(eq['lhs'])} = {_text(eq['rhs'])}" for eq in content["model"])
    definitions = {v["variable"]: _text(v["value"]) for v in content.get("model_local_variables") or []}
    used = {node.id for text in (*equations, *definitions.values()) for node in walk(parse_expression(text)) if isinstance(node, Name)}
    exogenous = [e for e in exogenous if e in used]          # a declared shock that no equation uses is left out, as Dynare's nostrict does
    missing = [e for e in exogenous if e not in stderr and e not in sd_name]
    if missing:
        raise DynareImportError(f"no standard deviation for the shocks {missing} (give `var {missing[0]}; stderr ...;` in the shocks block)")
    shocks = {e: Shock(f"normal(0, {sd_name[e] if e in sd_name else _text(stderr[e])})") for e in exogenous}
    steady = {}
    for entry in content.get("steady_state_model") or []:
        steady[entry["lhs"]] = _text(entry["rhs"])
    if not steady:
        for s in statements:
            if s.get("statementName") == "initval":
                steady.update({v["name"]: _text(v["value"]) for v in s["vals"] if v["name"] in endogenous})
    block = Block(outputs=tuple(endogenous), shocks=shocks, definitions=definitions, equations=equations, steady_state=steady)

    observed = content.get("varobs") or []
    errors = {v: (sd_name[v] if v in sd_name else _text(stderr[v])) for v in observed if (v in stderr or v in sd_name) and v not in exogenous}
    data = {v: DataLink(observes=v, error=None if v not in errors else f"normal(0, {errors[v]})") for v in observed}
    model_name = name or Path(mod_file).stem
    return Model(name=model_name, time="quarterly", sets={}, expectations="rational", parameters=parameters, blocks={model_name: block},
                 data=data)


def _parameter_values(statements: list, values: dict, estimated: set) -> dict:
    """Every parameter's value, worked out statement by statement in the file's order (a parameter set twice takes the second
    value from then on). A value that is not a number or an expression in numbers and earlier values (one computed by MATLAB
    code) is taken from `values`. Estimated parameters' values are worked out too, since other values may use them."""
    known = {}
    for s in statements:
        if s.get("statementName") != "param_init":
            continue
        name = s["name"]
        try:
            known[name] = _constant(parse_expression(s["value"]), known, name)
        except DynareImportError:
            if name in values:
                known[name] = float(values[name])
            elif name not in estimated:
                raise DynareImportError(f"the value of {name}, {s['value']}, is computed outside Dynare's language: give it in `values` "
                                        "(for example from the authors' stored results, `stored_parameter_values`)") from None
    return known


def stored_parameter_values(results_file) -> dict:
    """The parameter values a Dynare results file (`<model>_results.mat`, a path or bytes) stores in M_.params."""
    import io

    from scipy.io import loadmat

    source = io.BytesIO(results_file) if isinstance(results_file, bytes) else results_file
    M = loadmat(source, variable_names=["M_"], squeeze_me=True, struct_as_record=False)["M_"]
    names = [str(n).strip() for n in (M.param_names if not isinstance(M.param_names, str) else [M.param_names])]
    return dict(zip(names, (float(v) for v in M.params)))


def _constant(expr, known: dict, name: str) -> float:
    """A parameter's value, written as a number or an expression in numbers and the values of parameters set before it."""
    if isinstance(expr, Num):
        return float(expr.value)
    if isinstance(expr, Name) and expr.id in known:
        return known[expr.id]
    if isinstance(expr, Unary) and expr.op == "-":
        return -_constant(expr.operand, known, name)
    if isinstance(expr, Binary) and expr.op in "+-*/^":
        a, b = _constant(expr.left, known, name), _constant(expr.right, known, name)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "^": a ** b}[expr.op]
    if isinstance(expr, Call) and expr.func in ("log", "exp", "sqrt") and len(expr.args) == 1:
        return float(getattr(math, expr.func)(_constant(expr.args[0], known, name)))
    raise DynareImportError(f"the value of {name}, {to_text(expr)}, is not a number or an expression in other parameters' values")


def import_dynare_source(path: str | Path) -> Model:
    """A specification from a Dynare model that is published in an archive, described by a small YAML file:

        source:
          url: https://.../replication-files.zip
          sha256: "<checksum of the archive>"
          member: Folder/model.mod          # the file to import; the files it includes are read from the same folder
        values: Folder/model_results.mat    # optional: parameters the file computes in MATLAB code take the values stored here
        defines: {flag: 1}                  # optional: macro variables

    The archive is downloaded once into the data cache (`mutatis.measure.manifest.fetch`) and its folder's model files are
    unpacked there; nothing of it goes into the repository."""
    import zipfile

    import yaml

    from mutatis.measure.manifest import cache_folder, fetch

    description = yaml.safe_load(Path(path).read_text())
    source = description["source"]
    archive = fetch(source["url"], source["sha256"])
    member = Path(source["member"])
    unpacked = cache_folder() / f"{source['sha256']}-files"
    with zipfile.ZipFile(archive) as files:
        for item in files.namelist():
            if Path(item).parent == member.parent and item.endswith((".mod", ".inc")) and not (unpacked / item).exists():
                files.extract(item, unpacked)
        values = stored_parameter_values(files.read(description["values"])) if "values" in description else None
    return import_dynare(unpacked / member, name=description.get("name"), defines=description.get("defines"), values=values)

