"""Quantities: named results computed from a solved model and the data, written in the specification's expression language.

The same expressions serve restrictions ("this must be at least zero"), reports ("show me this") and comparisons. Available:

    response(y[gdp], eps[policy], horizons=0..20)   response of a variable to a one-unit shock, at each horizon
    shock_equation(eps[policy], y[cpi])             the coefficient on a variable in the equation that defines a shock: the
                                                    entry of the inverse of the impact matrix (needs every series observed exactly)
    shock_estimate(eps[policy])                     that shock in each period, read off the data
    data(proxy)                                     a data series, over the periods the model explains
    correlation(a, b)                               the correlation of two series, over periods where both are present
    abs, max, min, + - * /, and max(j in shocks except policy: ...)

Variables are named as in the model (y[gdp]); shocks by their name without the block (eps[policy]), which must be unique.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.expand import unroll
from mutatis.compile.system import System
from mutatis.quantities.recover import recover_shocks
from mutatis.quantities.responses import impulse_responses
from mutatis.solve.contract import SolvedModel
from mutatis.spec.expr import Binary, Call, Expr, Index, Name, Num, Range, Unary, parse_expression, parse_requirement, to_text


class QuantityError(ValueError):
    pass


@dataclass
class Context:
    """Everything a quantity may need: the compiled system, a solved model, and the data the model explains."""

    system: System
    solved: SolvedModel
    observed: jnp.ndarray
    outside: jnp.ndarray
    start_state: jnp.ndarray | None
    series: dict  # every data series by name, over the periods the model explains


def _label(expr: Expr) -> str:
    if isinstance(expr, Name):
        return expr.id
    if isinstance(expr, Index) and isinstance(expr.base, Name) and all(isinstance(i, (Name, Num)) for i in expr.items):
        return f"{expr.base.id}[{', '.join(str(i.id if isinstance(i, Name) else i.value) for i in expr.items)}]"
    raise QuantityError(f"expected a variable or shock such as y[gdp] or eps[policy], not {to_text(expr)}")


def variable_index(system: System, expr: Expr) -> int:
    label = _label(expr)
    if label not in system.variable_labels:
        raise QuantityError(f"{label} is not a variable of the model")
    return system.variable_labels.index(label)


def shock_index(system: System, expr: Expr) -> int:
    label = _label(expr)
    matches = [i for i, full in enumerate(system.shock_labels) if full == label or full.split(".", 1)[-1] == label]
    if len(matches) != 1:
        raise QuantityError(f"{label} matches {len(matches)} shocks of the model; it must match exactly one")
    return matches[0]


def observation_index(system: System, variable: int) -> int:
    """The data series that observes a variable exactly and alone."""
    from mutatis.measure.data import _identity_label  # the same test the data layer uses to read the starting state

    label = system.variable_labels[variable]
    for i, observation in enumerate(system.observations):
        if observation.error is None and _identity_label(observation.expr, system) == label:
            return i
    raise QuantityError(f"no data series observes {label} exactly")


def _horizons(expr: Expr | None) -> list[int]:
    if expr is None:
        return [0]
    if isinstance(expr, Range) and isinstance(expr.start, Num) and isinstance(expr.stop, Num):
        return list(range(int(expr.start.value), int(expr.stop.value) + 1))
    if isinstance(expr, Num):
        return [int(expr.value)]
    raise QuantityError("horizons is a whole number or a range such as 0..20")


def evaluate_quantity(text_or_expr, context: Context):
    expr = parse_expression(text_or_expr) if isinstance(text_or_expr, str) else text_or_expr
    return _evaluate(unroll(expr, context.system.sets), context)


def _evaluate(expr: Expr, ctx: Context):
    system = ctx.system
    match expr:
        case Num(value=value):
            return jnp.asarray(float(value))
        case Unary(op=op, operand=operand):
            value = _evaluate(operand, ctx)
            return -value if op == "-" else value
        case Binary(op=op, left=left, right=right) if op in "+-*/":
            a, b = _evaluate(left, ctx), _evaluate(right, ctx)
            return {"+": jnp.add, "-": jnp.subtract, "*": jnp.multiply, "/": jnp.divide}[op](a, b)
        case Call(func="response", args=(variable, shock), kwargs=kwargs):
            horizons = _horizons(dict(kwargs).get("horizons"))
            responses = impulse_responses(ctx.solved, system.n_variables, len(system.shock_labels), len(system.outside_labels), max(horizons))
            return responses[jnp.array(horizons), variable_index(system, variable), shock_index(system, shock)]
        case Call(func="shock_equation", args=(shock, variable)):
            form = ctx.solved.linear_form()
            ZQ = form.Z @ form.Q
            if ZQ.shape[0] != ZQ.shape[1]:
                raise QuantityError("shock_equation needs as many series observed exactly as there are shocks")
            return jnp.linalg.inv(ZQ)[shock_index(system, shock), observation_index(system, variable_index(system, variable))]
        case Call(func="shock_estimate", args=(shock,)):
            if ctx.start_state is None:
                raise QuantityError("shock_estimate needs the starting state read from the data")
            shocks, _ = recover_shocks(ctx.solved, ctx.observed, ctx.outside, ctx.start_state)
            return shocks[:, shock_index(system, shock)]
        case Call(func="data", args=(Name(id=series),)):
            if series not in ctx.series:
                raise QuantityError(f"no data series {series!r}")
            return jnp.asarray(ctx.series[series])
        case Call(func="correlation", args=(a, b)):
            return _correlation(_evaluate(a, ctx), _evaluate(b, ctx))
        case Call(func="abs", args=(inner,)):
            return jnp.abs(_evaluate(inner, ctx))
        case Call(func=func, args=args) if func in ("max", "min") and args:
            values = [_evaluate(a, ctx) for a in args]
            out = values[0]
            for v in values[1:]:
                out = jnp.maximum(out, v) if func == "max" else jnp.minimum(out, v)
            return out
    raise QuantityError(f"cannot compute {to_text(expr)} as a quantity")


def _correlation(a, b):
    both = ~(jnp.isnan(a) | jnp.isnan(b))
    a, b = jnp.where(both, a, 0.0), jnp.where(both, b, 0.0)
    n = jnp.sum(both)
    ma, mb = jnp.sum(a) / n, jnp.sum(b) / n
    da, db = jnp.where(both, a - ma, 0.0), jnp.where(both, b - mb, 0.0)
    return jnp.sum(da * db) / jnp.sqrt(jnp.sum(da**2) * jnp.sum(db**2))


_COMPARE = {">=": jnp.greater_equal, ">": jnp.greater, "<=": jnp.less_equal, "<": jnp.less, "==": jnp.equal, "!=": jnp.not_equal}


def restriction_holds(quantity: str, require: str, context: Context):
    """Whether a restriction holds: every element of the quantity meets the requirement."""
    op, bound = parse_requirement(require)
    value = evaluate_quantity(quantity, context)
    return jnp.all(_COMPARE[op](value, _evaluate(bound, context)))
