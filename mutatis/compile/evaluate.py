"""Evaluating a compiled syntax tree with JAX.

After compiling, an equation refers to variables now, one period back (`Shift(-1)`) and one period ahead inside E(...)
(`Shift(+1)`); to shocks and outside variables now; and to parameters. `Layout` says where each of them sits in the flat
arrays passed in, and which set each axis of a vector runs over.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.expand import CompileError
from mutatis.compile.symbols import OUTSIDE, PARAMETER, SHOCK, VARIABLE, Element, Ref, Whole
from mutatis.spec.expr import Binary, Call, Expr, Index, Num, Shift, Unary


@dataclass(frozen=True)
class Slot:
    """Where one named quantity sits in a flat array: `positions` has the quantity's own shape."""

    positions: np.ndarray
    axes: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    variables: dict[str, Slot]
    shocks: dict[str, Slot]
    outside: dict[str, Slot]
    parameter_axes: dict[str, tuple[str, ...]]
    sets: dict[str, tuple]

    def element_position(self, set_name: str, value) -> int:
        elements = self.sets[set_name]
        for i, e in enumerate(elements):
            if e == value or str(e) == str(value):
                return i
        raise CompileError(f"{value!r} is not an element of {set_name!r}")


@dataclass
class Values:
    """The arrays an expression is evaluated on. Slots an expression does not need may be None (an observation needs only `now`)."""

    lead: jnp.ndarray | None
    now: jnp.ndarray | None
    lag: jnp.ndarray | None
    shocks: jnp.ndarray | None
    outside: jnp.ndarray | None
    params: dict


_FUNCTIONS = {
    "log": jnp.log, "exp": jnp.exp, "sqrt": jnp.sqrt, "abs": jnp.abs, "chol": jnp.linalg.cholesky, "inv": jnp.linalg.inv,
    "transpose": lambda x: jnp.swapaxes(x, -1, -2),
}


def evaluate(expr: Expr, layout: Layout, values: Values):
    match expr:
        case Num(value=value):
            return jnp.asarray(float(value))
        case Ref():
            return _ref(expr, 0, layout, values)
        case Shift(base=base, lag=Num(value=lag)):
            if isinstance(base, Ref):
                return _ref(base, int(lag), layout, values)
            return _index(base, int(lag), layout, values)
        case Index():
            return _index(expr, 0, layout, values)
        case Unary(op=op, operand=operand):
            value = evaluate(operand, layout, values)
            return -value if op == "-" else value
        case Binary(op=op, left=left, right=right):
            a, b = evaluate(left, layout, values), evaluate(right, layout, values)
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "^":
                return jnp.power(a, b)
            if op == "@":
                return jnp.matmul(a, b)
        case Call(func="E", args=(inner,)):
            return evaluate(inner, layout, values)
        case Call(func=func, args=args) if func in ("max", "min"):
            items = [evaluate(a, layout, values) for a in args]
            combine = jnp.maximum if func == "max" else jnp.minimum
            out = items[0]
            for item in items[1:]:
                out = combine(out, item)
            return out
        case Call(func="sum", args=(inner,)):
            return jnp.sum(evaluate(inner, layout, values))
        case Call(func=func, args=(inner,)) if func in _FUNCTIONS:
            return _FUNCTIONS[func](evaluate(inner, layout, values))
    raise CompileError(f"cannot evaluate {type(expr).__name__} {getattr(expr, 'func', '')} in an equation")


def _ref(ref: Ref, lag: int, layout: Layout, values: Values):
    if ref.kind == PARAMETER:
        return values.params[ref.key]
    if ref.kind == VARIABLE:
        source = values.now if lag == 0 else values.lag if lag == -1 else values.lead if lag == 1 else None
        if source is None:
            raise CompileError(f"{ref.key} shifted by {lag}: the compiler should have removed this")
        return source[layout.variables[ref.key].positions]
    if lag != 0:
        raise CompileError(f"{ref.key} shifted by {lag}: the compiler should have removed this")
    if ref.kind == SHOCK:
        return values.shocks[layout.shocks[ref.key].positions]
    if ref.kind == OUTSIDE:
        return values.outside[layout.outside[ref.key].positions]
    raise CompileError(f"unknown reference {ref!r}")


def axes_of(ref: Ref, layout: Layout) -> tuple[str, ...]:
    if ref.kind == PARAMETER:
        return layout.parameter_axes[ref.key]
    table = {VARIABLE: layout.variables, SHOCK: layout.shocks, OUTSIDE: layout.outside}[ref.kind]
    return table[ref.key].axes


def _index(expr: Index, lag: int, layout: Layout, values: Values):
    if not isinstance(expr.base, Ref):
        raise CompileError("only a named variable, shock or parameter can be indexed")
    value = _ref(expr.base, lag, layout, values)
    axes = axes_of(expr.base, layout)
    if len(expr.items) > len(axes):
        raise CompileError(f"{expr.base.key} has {len(axes)} index(es), but {len(expr.items)} were given")
    index = []
    for item, axis in zip(expr.items, axes):
        if isinstance(item, Whole):
            if item.set != axis:
                raise CompileError(f"{expr.base.key} runs over {axis!r}, not {item.set!r}")
            index.append(slice(None))
        elif isinstance(item, Element):
            index.append(layout.element_position(axis, item.value))
        else:
            raise CompileError(f"bad index {item!r}")
    return value[tuple(index)]
