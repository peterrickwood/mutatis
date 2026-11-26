"""Compiling a specification into an equation system.

The result, `System`, lists every scalar variable, shock and outside variable in a fixed order, describes every parameter,
and evaluates all the equations at once:

    residual(lead, now, lag, shocks, outside, params) = 0

where lead, now and lag are the variables one period ahead (inside expectations), now, and one period back. Data links
become observation expressions in terms of the current period: observe(now, params). Nothing here solves anything.
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.distributions import DistributionSpec, parse_distribution, value_domain
from mutatis.compile.evaluate import Layout, Slot, Values, evaluate
from mutatis.compile.expand import CompileError, Scope, expand_differences, inline_definitions, resolve, substitute, unroll, element_node
from mutatis.compile.expectations import apply_scheme, parse_scheme
from mutatis.compile.lags import Auxiliary, LagReducer
from mutatis.compile.symbols import PARAMETER, Ref
from mutatis.dist.domains import Domain
from mutatis.spec.checks import check, instances
from mutatis.spec.expr import Binary, Call, Expr, Index, Num, Shift, Unary, parse_expression
from mutatis.spec.model import Model, Parameter, port


@dataclass(frozen=True)
class ParameterInfo:
    key: str
    spec: Parameter
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    prior: DistributionSpec | None      # None when fixed, or when it has no prior
    domain: Domain

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def fixed_value(self):
        return None if self.spec.fixed is None else jnp.broadcast_to(jnp.asarray(self.spec.fixed, dtype=float), self.shape)


@dataclass(frozen=True)
class ShockInfo:
    key: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    distribution: DistributionSpec


@dataclass(frozen=True)
class OutsideInfo:
    key: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    series: dict[str, str]          # flat label -> data series that feeds it
    calendar: str | None = None     # or the calendar expression that defines it


@dataclass(frozen=True)
class ObservationInfo:
    series: str
    expr: Expr                       # in terms of the current period only
    error: DistributionSpec | None   # None: observed exactly


@dataclass(frozen=True)
class EquationInfo:
    block: str
    expr: Expr                       # the residual: zero when the equation holds
    text: str


@dataclass(frozen=True)
class System:
    name: str
    sets: dict
    layout: Layout
    variable_labels: tuple[str, ...]
    shock_labels: tuple[str, ...]
    outside_labels: tuple[str, ...]
    parameters: dict[str, ParameterInfo]
    shocks: dict[str, ShockInfo]
    outside: dict[str, OutsideInfo]
    equations: tuple[EquationInfo, ...]
    outputs_by_block: dict[str, tuple[str, ...]]
    observations: tuple[ObservationInfo, ...]
    auxiliary_series: tuple[str, ...]
    initial: dict[str, DistributionSpec]
    auxiliary: dict[str, Auxiliary]
    has_expectations: bool
    linear: bool
    presample: int                   # periods of data before the first one explained, when starting from the data itself
    model_variables: tuple[str, ...] = field(default=())

    def residual(self, lead, now, lag, shocks, outside, params):
        values = Values(lead, now, lag, shocks, outside, params)
        return jnp.concatenate([jnp.ravel(evaluate(e.expr, self.layout, values)) for e in self.equations])

    def observe(self, now, params):
        values = Values(None, now, None, None, None, params)
        return jnp.stack([jnp.reshape(evaluate(o.expr, self.layout, values), ()) for o in self.observations]) if self.observations else jnp.zeros(0)

    def fixed_parameters(self) -> dict:
        return {k: p.fixed_value for k, p in self.parameters.items() if p.spec.fixed is not None}

    @property
    def n_variables(self) -> int:
        return len(self.variable_labels)


def compile_model(model: Model) -> System:
    findings = check(model)
    if findings:
        raise CompileError("the specification has problems:\n  " + "\n  ".join(map(str, findings)))
    scheme, factor = parse_scheme(model.expectations)
    sets = model.sets
    variable_axes, outside_axes, block_outputs = _variables(model)
    parameters = _parameters(model)
    if factor is not None and factor not in model.parameters:
        raise CompileError(f"the discounting factor {factor!r} must be a model-level parameter")

    reducer = LagReducer()
    equations, shocks, initial = [], {}, {}
    for name, block in model.blocks.items():
        for label, instance, binding in instances(name, block, model):
            element = binding.get(block.over[0]) if block.over else None
            parameter_keys = {**{p: p for p in model.parameters}, **{p: _block_parameter_key(name, p, block.parameters[p], element) for p in block.parameters}}
            if block.over and instance is not block:  # a replaced instance: its own parameters
                parameter_keys.update({p: f"{label}.{p}" for p in instance.parameters})
            shock_keys = {s: (f"{label}.{s}" if instance is not block else f"{name}.{s}") for s in instance.shocks}
            scope = Scope(parameter_keys, shock_keys, set(variable_axes), set(outside_axes), sets)
            bindings = {k: element_node(_element_value(sets, block.over[1], v)) for k, v in binding.items()} if block.over else {}
            definitions = {k: substitute(parse_expression(v), bindings) for k, v in instance.definitions.items()}
            for text in instance.equations:
                parsed = parse_expression(text)
                residual = Binary("-", parsed.left, parsed.right)
                residual = unroll(inline_definitions(substitute(residual, bindings), definitions), sets)
                residual = apply_scheme(expand_differences(resolve(residual, scope)), scheme, factor)
                equations.append(EquationInfo(label, reducer.equation(residual), text))
            for s, info in instance.shocks.items():
                key = shock_keys[s]
                if key in shocks:
                    continue
                shape = tuple(len(sets[a]) for a in info.shape)
                shocks[key] = ShockInfo(key, info.shape, shape, parse_distribution(info.distribution, parameter_keys, shape))
            for variable, text in instance.initial.items():
                shape = tuple(len(sets[a]) for a in variable_axes[variable])
                initial[variable] = parse_distribution(text, parameter_keys, shape)

    observations, auxiliary_series = [], []
    for series, link in model.data.items():
        if link.observes is None:
            if link.feeds is None:
                auxiliary_series.append(series)
            continue
        keys = {**{p: p for p in model.parameters}, **{p: f"{series}.{p}" for p in link.parameters}}
        scope = Scope(keys, {}, set(variable_axes), set(outside_axes), sets)
        expr = reducer.observation(expand_differences(resolve(unroll(parse_expression(link.observes), sets), scope)))
        error = None if link.error is None else parse_distribution(link.error, keys, ())
        observations.append(ObservationInfo(series, expr, error))

    for aux in reducer.auxiliary.values():
        if aux.base in variable_axes:
            variable_axes[aux.key] = variable_axes[aux.base]
        elif aux.base in outside_axes:
            variable_axes[aux.key] = outside_axes[aux.base]
        else:
            variable_axes[aux.key] = shocks[aux.base].axes
        equations.append(EquationInfo("auxiliary", aux.equation, f"{aux.key} copies {aux.base}"))

    variable_slots, variable_labels = _slots(variable_axes, sets)
    outside_info = _outside(model, outside_axes, sets)
    outside_slots, outside_labels = _slots(outside_axes, sets)
    shock_slots, shock_labels = _slots({k: s.axes for k, s in shocks.items()}, sets)
    layout = Layout(variable_slots, shock_slots, outside_slots, {k: p.axes for k, p in parameters.items()}, sets)
    model_variables = tuple(v for v in variable_axes if v not in reducer.auxiliary)
    system = System(model.name, sets, layout, variable_labels, shock_labels, outside_labels, parameters, shocks, outside_info, tuple(equations),
                    {label: tuple(_labels(outputs, sets)) for label, outputs in block_outputs.items()}, tuple(observations), tuple(auxiliary_series),
                    initial, dict(reducer.auxiliary), any(_has_expectation(e.expr) for e in equations),
                    all(_degree(e.expr) <= 1 for e in equations) and all(_degree(o.expr) <= 1 for o in observations),
                    _presample(reducer.auxiliary), model_variables)
    _count(system)
    return system


# Variables, parameters, outside variables -------------------------------------------------------------------------------------

def _element_value(sets, set_name, text):
    for e in sets[set_name]:
        if str(e) == str(text):
            return e
    return text


def _set_of_element(sets, element) -> str:
    owners = [name for name, values in sets.items() if any(str(e) == str(element) for e in values)]
    if len(owners) != 1:
        raise CompileError(f"element {element!r} belongs to {len(owners)} sets; it must belong to exactly one")
    return owners[0]


def _port_axes(reference, sets, binding, template_set) -> tuple[str, tuple]:
    name, items = port(reference)
    axes = []
    for item in items:
        if item in binding:
            axes.append(template_set)
        elif item in sets:
            axes.append(item)
        else:
            axes.append(_set_of_element(sets, item))
    return name, tuple(axes)


def _variables(model: Model):
    variable_axes: dict[str, tuple] = {}
    block_outputs: dict[str, list[str]] = {}
    for name, block in model.blocks.items():
        for label, instance, binding in instances(name, block, model):
            outputs = []
            for reference in instance.outputs:
                variable, axes = _port_axes(reference, model.sets, binding, block.over[1] if block.over else None)
                if variable_axes.setdefault(variable, axes) != axes:
                    raise CompileError(f"{variable!r} runs over {variable_axes[variable]} in one place and {axes} in another")
                bound = port(reference)[1]
                outputs.append((variable, tuple(binding.get(i, i) for i in bound)))
            block_outputs[label] = outputs
    outside_axes: dict[str, tuple] = {}
    for series, link in model.data.items():
        if link.feeds is not None:
            variable, axes = _port_axes(link.feeds, model.sets, {}, None)
            outside_axes.setdefault(variable, axes)
    for name, block in model.blocks.items():
        for _, instance, binding in instances(name, block, model):
            for reference in instance.inputs:
                variable, axes = _port_axes(reference, model.sets, binding, block.over[1] if block.over else None)
                if variable not in variable_axes:
                    outside_axes[variable] = axes if variable not in outside_axes or len(axes) >= len(outside_axes[variable]) else outside_axes[variable]
    for name in model.calendar:
        outside_axes[name] = ()
    return variable_axes, outside_axes, block_outputs


def _labels(outputs, sets) -> list[str]:
    labels = []
    for variable, items in outputs:
        combos = [()]
        for item in items:
            options = [str(e) for e in sets[item]] if item in sets else [str(item)]
            combos = [c + (o,) for c in combos for o in options]
        labels += [f"{variable}[{', '.join(c)}]" if c else variable for c in combos]
    return labels


def _slots(axes_by_key: dict[str, tuple], sets) -> tuple[dict[str, Slot], tuple[str, ...]]:
    slots, labels, position = {}, [], 0
    for key, axes in axes_by_key.items():
        shape = tuple(len(sets[a]) for a in axes)
        size = math.prod(shape)
        slots[key] = Slot(np.arange(position, position + size).reshape(shape), axes)
        combos = [()]
        for axis in axes:
            combos = [c + (str(e),) for c in combos for e in sets[axis]]
        labels += [f"{key}[{', '.join(c)}]" if c else key for c in combos]
        position += size
    return slots, tuple(labels)


def _block_parameter_key(block: str, name: str, parameter: Parameter, element) -> str:
    if element is not None and element in parameter.separate:
        return f"{block}.{name}[{element}]"
    return f"{block}.{name}"


def _parameter_info(key: str, spec: Parameter, sets, keys) -> ParameterInfo:
    shape = tuple(len(sets[a]) for a in spec.shape)
    prior = None
    if spec.prior is not None:
        prior = parse_distribution(spec.prior, keys, shape)
    return ParameterInfo(key, spec, tuple(spec.shape), shape, prior, value_domain(spec.domain, prior))


def _parameters(model: Model) -> dict[str, ParameterInfo]:
    model_keys = {p: p for p in model.parameters}
    out = {k: _parameter_info(k, p, model.sets, model_keys) for k, p in model.parameters.items()}
    for name, block in model.blocks.items():
        keys = {**model_keys, **{p: f"{name}.{p}" for p in block.parameters}}
        for p, spec in block.parameters.items():
            out[f"{name}.{p}"] = _parameter_info(f"{name}.{p}", spec, model.sets, keys)
            for element, own in spec.separate.items():
                out[f"{name}.{p}[{element}]"] = _parameter_info(f"{name}.{p}[{element}]", own, model.sets, keys)
        for element, override in block.overrides.items():
            for p, spec in override.parameters.items():
                key = f"{name}[{element}].{p}"
                out[key] = _parameter_info(key, spec, model.sets, {**model_keys, **{q: f"{name}[{element}].{q}" for q in override.parameters}})
    for series, link in model.data.items():
        keys = {**model_keys, **{p: f"{series}.{p}" for p in link.parameters}}
        for p, spec in link.parameters.items():
            out[f"{series}.{p}"] = _parameter_info(f"{series}.{p}", spec, model.sets, keys)
    return out


def _outside(model: Model, outside_axes, sets) -> dict[str, OutsideInfo]:
    out = {}
    for key, axes in outside_axes.items():
        shape = tuple(len(sets[a]) for a in axes)
        if key in model.calendar:
            out[key] = OutsideInfo(key, axes, shape, {}, model.calendar[key])
            continue
        series = {}
        for name, link in model.data.items():
            if link.feeds is not None and port(link.feeds)[0] == key:
                for label in _labels([(key, port(link.feeds)[1])], sets):
                    series[label] = name
        out[key] = OutsideInfo(key, axes, shape, series)
    return out


def _presample(auxiliary: dict[str, Auxiliary]) -> int:
    deepest = 0
    for key in auxiliary:
        if "@lag" in key:
            deepest = max(deepest, int(key.rsplit("@lag", 1)[1]))
    return deepest + 1


# Properties of the equations --------------------------------------------------------------------------------------------------

def _has_expectation(expr: Expr) -> bool:
    if isinstance(expr, Call) and expr.func == "E":
        return True
    return any(_has_expectation(getattr(expr, f)) for f in ("left", "right", "operand", "base") if isinstance(getattr(expr, f, None), Expr)) or \
        any(_has_expectation(a) for a in getattr(expr, "args", ()))


_INFINITE = 10**6


def _degree(expr: Expr) -> int:
    """The polynomial degree of an expression in the variables, shocks and outside variables; _INFINITE if not a polynomial."""
    match expr:
        case Num():
            return 0
        case Ref(kind=kind):
            return 0 if kind == PARAMETER else 1
        case Shift(base=base) | Index(base=base):
            return _degree(base)
        case Unary(operand=operand):
            return _degree(operand)
        case Binary(op=op, left=left, right=right):
            a, b = _degree(left), _degree(right)
            if op in "+-":
                return max(a, b)
            if op in ("*", "@"):
                return a + b
            if op == "/":
                return a if b == 0 else _INFINITE
            if op == "^":
                return 0 if a == 0 and b == 0 else _INFINITE
        case Call(func="E", args=(inner,)):
            return _degree(inner)
        case Call(args=args):
            return 0 if all(_degree(a) == 0 for a in args) else _INFINITE
    return _INFINITE


def _count(system: System) -> None:
    """Each block must have exactly as many equation rows as scalar outputs."""
    params = {k: jax.ShapeDtypeStruct(p.shape, jnp.float64) for k, p in system.parameters.items()}
    n, k, m = len(system.variable_labels), len(system.shock_labels), len(system.outside_labels)
    rows_by_block: dict[str, int] = {}
    for equation in system.equations:
        def one(lead, now, lag, shocks, outside, params, expr=equation.expr):
            return jnp.ravel(evaluate(expr, system.layout, Values(lead, now, lag, shocks, outside, params)))
        shape = jax.eval_shape(one, *(jax.ShapeDtypeStruct((size,), jnp.float64) for size in (n, n, n, k, m)), params)
        rows_by_block[equation.block] = rows_by_block.get(equation.block, 0) + shape.shape[0]
    problems = []
    for label, outputs in system.outputs_by_block.items():
        rows = rows_by_block.get(label, 0)
        if rows != len(outputs):
            problems.append(f"{label}: {rows} equation rows for {len(outputs)} outputs ({', '.join(outputs)})")
    total = sum(rows_by_block.values())
    if not problems and total != n:
        problems.append(f"{total} equation rows for {n} variables")
    if problems:
        raise CompileError("each block needs one equation per output:\n  " + "\n  ".join(problems))
