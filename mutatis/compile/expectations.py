"""Expectation schemes: how every marked expectation term E(...) is computed. Each is a rewrite of the syntax tree.

- rational: E(...) is kept. The model must then be solved for the one rule that makes these expectations come true.
- naive: people expect any future value to equal its most recent past value. Inside E(...), x(+k) becomes x(-1), and the
  E(...) is dropped. What is left looks only backward, so no forward-looking solve is needed.
- discounted(m): Gabaix's cognitive discounting. E(...) becomes m * E(...), with m a model-level parameter between 0 and 1.
  This is meant for linear models written in deviations from a resting point; on levels it would also shrink the resting point.
"""

from mutatis.compile.expand import CompileError, map_children, shift
from mutatis.compile.symbols import PARAMETER, Ref
from mutatis.spec.expr import Binary, Call, Expr, Name, Num, Shift, parse_expression


def parse_scheme(text: str) -> tuple[str, str | None]:
    expr = parse_expression(text, shifts=False)
    if isinstance(expr, Name) and expr.id in ("rational", "naive"):
        return expr.id, None
    if isinstance(expr, Call) and expr.func == "discounted":
        args = [*expr.args, *(v for k, v in expr.kwargs if k == "factor")]
        if len(args) == 1 and isinstance(args[0], Name):
            return "discounted", args[0].id
    raise CompileError(f"unknown expectation scheme {text!r}: use rational, naive, or discounted(<parameter name>)")


def apply_scheme(expr: Expr, scheme: str, factor_key: str | None) -> Expr:
    if isinstance(expr, Call) and expr.func == "E":
        if len(expr.args) != 1 or expr.kwargs:
            raise CompileError("E(...) takes one argument")
        inner = apply_scheme(expr.args[0], scheme, factor_key)
        if scheme == "naive":
            return _future_to_last_past(inner)
        if scheme == "discounted":
            return Binary("*", Ref(PARAMETER, factor_key), Call("E", (inner,)))
        return Call("E", (inner,))
    return map_children(expr, lambda child: apply_scheme(child, scheme, factor_key))


def _future_to_last_past(expr: Expr) -> Expr:
    if isinstance(expr, Shift) and isinstance(expr.lag, Num) and expr.lag.value > 0:
        return shift(expr.base, -1)
    return map_children(expr, _future_to_last_past)
