"""Rewriting syntax trees on the way from a specification to an equation system.

The steps, in the order the compiler applies them to each equation:

1. `substitute`: a template's index name becomes the instance's element.
2. `inline_definitions`: names of definitions become the expressions they stand for.
3. `unroll`: sums, maxima and minima over sets become explicit expressions.
4. `resolve`: every remaining name becomes a `Ref` (variable, outside variable, parameter or shock), an `Element` or a `Whole` set.
5. `expand_differences`: d(x) becomes x - x(-1), shifting every variable inside.
"""

from dataclasses import fields, replace

from mutatis.compile.symbols import OUTSIDE, PARAMETER, SHOCK, VARIABLE, Element, Ref, Whole
from mutatis.spec.expr import Binary, Call, Expr, Index, Name, Num, Over, Shift, Unary


class CompileError(ValueError):
    pass


def map_children(expr: Expr, fn) -> Expr:
    """The same node with `fn` applied to each child expression."""
    changes = {}
    for f in fields(expr):
        if f.name == "span":
            continue
        value = getattr(expr, f.name)
        if isinstance(value, Expr):
            changes[f.name] = fn(value)
        elif isinstance(value, tuple) and value and all(isinstance(v, Expr) for v in value):
            changes[f.name] = tuple(fn(v) for v in value)
        elif isinstance(value, tuple) and value and all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], Expr) for v in value):
            changes[f.name] = tuple((k, fn(v)) for k, v in value)
    return replace(expr, **changes) if changes else expr


def substitute(expr: Expr, bindings: dict[str, Expr]) -> Expr:
    """Replace names by expressions. A name bound by an inner sum is left alone inside that sum."""
    if isinstance(expr, Name):
        return bindings.get(expr.id, expr)
    if isinstance(expr, Over) and expr.index in bindings:
        inner = {k: v for k, v in bindings.items() if k != expr.index}
        return replace(expr, body=substitute(expr.body, inner))
    return map_children(expr, lambda child: substitute(child, bindings))


def element_node(value) -> Expr:
    return Num(value, str(value)) if isinstance(value, int) else Name(str(value))


def inline_definitions(expr: Expr, definitions: dict[str, Expr], seen: tuple[str, ...] = ()) -> Expr:
    if isinstance(expr, Name) and expr.id in definitions:
        if expr.id in seen:
            raise CompileError(f"definitions refer to each other in a circle: {' -> '.join((*seen, expr.id))}")
        return inline_definitions(definitions[expr.id], definitions, (*seen, expr.id))
    return map_children(expr, lambda child: inline_definitions(child, definitions, seen))


def unroll(expr: Expr, sets: dict[str, tuple]) -> Expr:
    if isinstance(expr, Over):
        if expr.set not in sets:
            raise CompileError(f"{expr.func} over unknown set {expr.set!r}")
        terms = [unroll(substitute(expr.body, {expr.index: element_node(e)}), sets) for e in sets[expr.set] if str(e) not in expr.exclude]
        if expr.func == "sum":
            if not terms:
                return Num(0, "0")
            total = terms[0]
            for term in terms[1:]:
                total = Binary("+", total, term)
            return total
        if not terms:
            raise CompileError(f"{expr.func} over an empty set")
        return Call(expr.func, tuple(terms))
    return map_children(expr, lambda child: unroll(child, sets))


def static_integer(expr: Expr) -> int:
    """The value of a time shift, which must be a whole number known before any data or parameters."""
    match expr:
        case Num(value=value) if float(value).is_integer():
            return int(value)
        case Unary(op=op, operand=operand):
            return -static_integer(operand) if op == "-" else static_integer(operand)
        case Binary(op=op, left=left, right=right) if op in "+-":
            a, b = static_integer(left), static_integer(right)
            return a + b if op == "+" else a - b
    raise CompileError("a time shift must be a whole number, such as x(-1), or a set element that is one, such as y(-l)")


class Scope:
    """What each name means inside one block instance."""

    def __init__(self, parameters: dict[str, str], shocks: dict[str, str], variables: set[str], outside: set[str], sets: dict[str, tuple]):
        self.parameters, self.shocks, self.variables, self.outside, self.sets = parameters, shocks, variables, outside, sets
        self.elements = {str(e) for values in sets.values() for e in values}

    def name(self, name: str) -> Expr:
        if name in self.parameters:
            return Ref(PARAMETER, self.parameters[name])
        if name in self.shocks:
            return Ref(SHOCK, self.shocks[name])
        if name in self.variables:
            return Ref(VARIABLE, name)
        if name in self.outside:
            return Ref(OUTSIDE, name)
        raise CompileError(f"{name!r} does not refer to anything here")

    def item(self, expr: Expr) -> Expr:
        if isinstance(expr, Name) and expr.id in self.sets:
            return Whole(expr.id)
        if isinstance(expr, Name) and expr.id in self.elements:
            return Element(expr.id)
        if isinstance(expr, Num):
            return Element(expr.value)
        raise CompileError(f"an index must be a set, an element of a set, or a whole number, not {expr!r}")


def resolve(expr: Expr, scope: Scope) -> Expr:
    match expr:
        case Name(id=name):
            return scope.name(name)
        case Index(base=base, items=items):
            return Index(resolve(base, scope), tuple(scope.item(i) for i in items))
        case Shift(base=base, lag=lag):
            resolved = resolve(base, scope)
            target = resolved.base if isinstance(resolved, Index) else resolved
            if not (isinstance(target, Ref) and target.kind in (VARIABLE, OUTSIDE, SHOCK)):
                raise CompileError("only variables and shocks can be shifted in time")
            return shift(resolved, static_integer(lag))
    return map_children(expr, lambda child: resolve(child, scope))


def shift(expr: Expr, k: int) -> Expr:
    """The expression k periods later (k > 0) or earlier (k < 0): every variable, outside variable and shock inside moves by k."""
    if k == 0:
        return expr
    match expr:
        case Shift(base=base, lag=Num(value=lag)):
            total = lag + k
            return base if total == 0 else Shift(base, Num(total, str(total)))
        case Ref(kind=kind) if kind in (VARIABLE, OUTSIDE, SHOCK):
            return Shift(expr, Num(k, str(k)))
        case Index(base=Ref(kind=kind)) if kind in (VARIABLE, OUTSIDE, SHOCK):
            return Shift(expr, Num(k, str(k)))
        case Index():
            return expr  # a parameter's entry does not move in time
    return map_children(expr, lambda child: shift(child, k))


def expand_differences(expr: Expr) -> Expr:
    if isinstance(expr, Call) and expr.func == "d":
        if len(expr.args) != 1 or expr.kwargs:
            raise CompileError("d(x) takes one argument: d(x) = x - x(-1)")
        inner = expand_differences(expr.args[0])
        return Binary("-", inner, shift(inner, -1))
    return map_children(expr, expand_differences)
