"""Rewriting a system so that every equation looks at most one period back and one period ahead.

A lag longer than one period gets a chain of extra variables: v@lag1 = v(-1), v@lag2 = v@lag1(-1), and so on, so that
v(-3) becomes v@lag2(-1). A shock or outside variable that appears with a lag first gets a copy that is part of the state,
b@now = b, whose lags then work the same way. Observation expressions may only look at the current period, so their lags are
replaced by the extra variables directly: v(-2) in an observation becomes v@lag2.

Leads of more than one period, and leads of shocks or outside variables, are not supported yet.
"""

from dataclasses import dataclass

from mutatis.compile.expand import CompileError, map_children
from mutatis.compile.symbols import OUTSIDE, SHOCK, VARIABLE, Ref
from mutatis.spec.expr import Binary, Call, Expr, Index, Num, Shift


@dataclass(frozen=True)
class Auxiliary:
    key: str        # the new variable
    base: str       # the variable, shock or outside variable it copies
    base_kind: str
    equation: Expr  # residual: zero when the copy holds


class LagReducer:
    def __init__(self):
        self.auxiliary: dict[str, Auxiliary] = {}

    def _now_copy(self, kind: str, key: str) -> str:
        """A state variable equal to the current value of a shock or outside variable."""
        if kind == VARIABLE:
            return key
        copy = f"{key}@now"
        if copy not in self.auxiliary:
            self.auxiliary[copy] = Auxiliary(copy, key, kind, Binary("-", Ref(VARIABLE, copy), Ref(kind, key)))
        return copy

    def _lagged(self, key: str, depth: int) -> str:
        """The variable equal to `key` `depth` periods ago (depth >= 1), creating the chain as needed."""
        previous = key
        for j in range(1, depth + 1):
            name = f"{key}@lag{j}"
            if name not in self.auxiliary:
                self.auxiliary[name] = Auxiliary(name, key, VARIABLE, Binary("-", Ref(VARIABLE, name), Shift(Ref(VARIABLE, previous), Num(-1, "-1"))))
            previous = name
        return previous

    def equation(self, expr: Expr, inside_expectation: bool = False) -> Expr:
        if isinstance(expr, Call) and expr.func == "E":
            return Call("E", tuple(self.equation(a, True) for a in expr.args))
        if isinstance(expr, Shift):
            base, items, ref = _parts(expr.base)
            k = int(expr.lag.value)
            if k > 0:
                if not inside_expectation:
                    raise CompileError("a value from the future must be inside E(...)")
                if k > 1 or ref.kind != VARIABLE:
                    raise CompileError("only variables one period ahead, x(+1), are supported inside E(...) so far")
                return expr
            key = self._now_copy(ref.kind, ref.key)
            if k == -1:
                return Shift(_rebuild(Ref(VARIABLE, key), items), Num(-1, "-1"))
            return Shift(_rebuild(Ref(VARIABLE, self._lagged(key, -k - 1)), items), Num(-1, "-1"))
        return map_children(expr, lambda child: self.equation(child, inside_expectation))

    def observation(self, expr: Expr) -> Expr:
        if isinstance(expr, Call) and expr.func == "E":
            raise CompileError("an observation cannot contain an expectation term")
        if isinstance(expr, Shift):
            base, items, ref = _parts(expr.base)
            k = int(expr.lag.value)
            if k > 0:
                raise CompileError("an observation cannot look into the future")
            return _rebuild(Ref(VARIABLE, self._lagged(self._now_copy(ref.kind, ref.key), -k)), items)
        if isinstance(expr, Ref) and expr.kind in (SHOCK, OUTSIDE):
            return Ref(VARIABLE, self._now_copy(expr.kind, expr.key))
        if isinstance(expr, Index) and isinstance(expr.base, Ref) and expr.base.kind in (SHOCK, OUTSIDE):
            return Index(Ref(VARIABLE, self._now_copy(expr.base.kind, expr.base.key)), expr.items)
        return map_children(expr, self.observation)


def _parts(base: Expr):
    if isinstance(base, Index):
        return base, base.items, base.base
    return base, None, base


def _rebuild(ref: Ref, items):
    return ref if items is None else Index(ref, items)
