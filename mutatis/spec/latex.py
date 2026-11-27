r"""Expressions written as mathematics (LaTeX), for display.

`to_latex(expr, timed)` writes a syntax tree the way it would be printed in a paper:

- names as symbols: `alpha` as \alpha; digits after a Greek or one-letter name, and an underscore after one, as a subscript
  (`psi1` as \psi_{1}, `sigma_eta` as \sigma_{\eta}); longer names in italics (`cash_rate` as \mathit{cash\_rate});
- the names in `timed` (variables and shocks) with the period as a subscript: `x` as x_{t}, `x(-1)` as x_{t-1}, `y[gdp]` as
  y_{\mathit{gdp},t}; a shifted name always carries its period;
- E(...) as \mathbb{E}_{t}[...], d(x) as \Delta x, sums and maxima over sets with the set below the sign, divisions as fractions;
- brackets only where the order of operations needs them.

The text a person wrote stays the specification; this is only how it is shown.
"""

import re
from collections.abc import Collection

from mutatis.spec.expr import Binary, Call, Compare, Date, Expr, Index, Name, Num, Over, Range, Shift, Unary, parse_expression

GREEK = {name: "\\" + name for name in (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega")}
GREEK["eps"] = r"\varepsilon"

_ATOM, _POWER, _UNARY, _PRODUCT, _OVER, _SUM, _RANGE, _COMPARE = 8, 7, 6, 5, 4, 3, 2, 1
_RELATIONS = {"=": "=", "==": "=", "!=": r"\neq", "<": "<", "<=": r"\le", ">": ">", ">=": r"\ge"}
_PLAIN_CALLS = {"log": r"\log", "exp": r"\exp", "max": r"\max", "min": r"\min"}


def to_latex(expr: Expr, timed: Collection[str] = ()) -> str:
    return _Writer(frozenset(timed)).write(expr)


def text_to_latex(text: str, timed: Collection[str] = ()) -> str:
    """Parse text in the expression language (an equation or an expression) and write it."""
    return to_latex(parse_expression(text), timed)


def distribution_latex(text: str) -> str:
    """A distribution as written in a specification, such as normal(0, sigma) or uniform (a bare name is a family with no arguments)."""
    expr = parse_expression(text, shifts=False)
    return to_latex(Call(expr.id) if isinstance(expr, Name) else expr)


def symbol(name: str) -> str:
    body, subscripts = _symbol_parts(name)
    return _attach(body, subscripts)


def _symbol_parts(name: str) -> tuple[str, list[str]]:
    if name in GREEK:
        return GREEK[name], []
    numbered = re.fullmatch(r"([A-Za-z]+?)(\d+)", name)
    if numbered and _short(numbered.group(1)):
        return _symbol_parts(numbered.group(1))[0], [numbered.group(2)]
    head, underscore, tail = name.partition("_")
    if underscore and tail and _short(head):
        return _symbol_parts(head)[0], [symbol(tail)]
    if len(name) == 1:
        return name, []
    return r"\mathit{" + name.replace("_", r"\_") + "}", []


def _short(name: str) -> bool:
    return name in GREEK or len(name) == 1


def _attach(body: str, subscripts: list[str]) -> str:
    return f"{body}_{{{','.join(subscripts)}}}" if subscripts else body


def _word(name: str) -> str:
    return r"\mathrm{" + name.replace("_", r"\_") + "}"


def _number(text: str) -> str:
    mantissa, e, exponent = text.lower().partition("e")
    if not e:
        return text
    power = f"10^{{{int(exponent)}}}"
    return power if mantissa in ("1", "1.", "1.0") else f"{mantissa} \\times {power}"


class _Writer:
    def __init__(self, timed: frozenset[str]):
        self.timed = timed

    def write(self, expr: Expr) -> str:
        return self._write(expr)[0]

    def _write(self, expr: Expr) -> tuple[str, int]:
        """The LaTeX for a node, and how tightly it binds (higher binds tighter), so a parent knows when to add brackets."""
        leaf = _leaf(expr)
        if leaf is not None:
            return self._leaf(*leaf), _ATOM
        match expr:
            case Num(value=value, text=text):
                return _number(text or repr(value)), _ATOM
            case Date(year=year, quarter=quarter):
                return _word(f"{year}Q{quarter}"), _ATOM
            case Index(base=base, items=items):
                return f"{self._tight(base, _ATOM)}\\left[{', '.join(map(self.write, items))}\\right]", _ATOM
            case Shift(base=base, lag=lag):
                return f"{self._tight(base, _ATOM)}_{{{self._period(lag)}}}", _ATOM
            case Call():
                return self._call(expr)
            case Unary(op=op, operand=operand):
                return f"{op}{self._operand(operand, _UNARY)}", _UNARY
            case Binary(op="/", left=left, right=right):   # binds like a sign: bracketed only as the base of a power
                return f"\\frac{{{self.write(left)}}}{{{self.write(right)}}}", _UNARY
            case Binary(op="^", left=left, right=right):
                return f"{self._tight(left, _ATOM, calls=False)}^{{{self.write(right)}}}", _POWER
            case Binary(op=op, left=left, right=right) if op in ("*", "@"):
                a, b = self._tight(left, _PRODUCT), self._operand(right, _PRODUCT)
                joiner = r" \cdot " if isinstance(right, Num) or b[:1].isdigit() else r"\, "
                return f"{a}{joiner}{b}", _PRODUCT
            case Binary(op=op, left=left, right=right):   # + and -
                right_text = self._operand(right, _SUM + (1 if op == "-" else 0))
                return f"{self._tight(left, _SUM)} {op} {right_text}", _SUM
            case Compare(op=op, left=left, right=right):
                return f"{self._tight(left, _RANGE)} {_RELATIONS[op]} {self._tight(right, _RANGE)}", _COMPARE
            case Range(start=start, stop=stop):
                return f"{self._tight(start, _SUM)}, \\dots, {self._tight(stop, _SUM)}", _RANGE
            case Over(func=func, index=index, set=set_name, exclude=exclude, body=body):
                below = f"{symbol(index)} \\in {symbol(set_name)}"
                if exclude:
                    below += r" \setminus \{" + ", ".join(map(symbol, exclude)) + r"\}"
                sign = {"sum": r"\sum", "max": r"\max", "min": r"\min"}[func]
                return f"{sign}_{{{below}}} {self._tight(body, _PRODUCT)}", _OVER
        raise TypeError(f"not an expression of the specification language: {expr!r}")

    def _tight(self, expr: Expr, needed: int, calls: bool = True) -> str:
        """The node's LaTeX, in brackets if it binds less tightly than `needed` (or, with calls=False, if it is a function call)."""
        text, strength = self._write(expr)
        if strength < needed or (not calls and isinstance(expr, Call)):
            return f"\\left({text}\\right)"
        return text

    def _operand(self, expr: Expr, needed: int) -> str:
        """A right-hand operand: a sign in front of it always gets brackets, so that a + (-b) is not printed as a + -b."""
        return f"\\left({self.write(expr)}\\right)" if isinstance(expr, Unary) else self._tight(expr, needed)

    def _leaf(self, name: str, items: list[Expr], lag: Expr | None) -> str:
        body, subscripts = _symbol_parts(name)
        subscripts = subscripts + [self.write(i) for i in items]
        if lag is not None:
            subscripts.append(self._period(lag))
        elif name in self.timed:
            subscripts.append("t")
        return _attach(body, subscripts)

    def _period(self, lag: Expr) -> str:
        match lag:
            case Num(value=0):
                return "t"
            case Num():
                return f"t+{self.write(lag)}"
            case Unary(op=op, operand=Num(value=0)):
                return "t"
            case Unary(op=op, operand=operand):
                return f"t{op}{self._tight(operand, _UNARY)}"
        return f"t+{self._tight(lag, _UNARY)}"

    def _call(self, expr: Call) -> tuple[str, int]:
        func, args = expr.func, expr.args
        if func == "E" and len(args) == 1 and not expr.kwargs:
            return f"\\mathbb{{E}}_{{t}}\\left[{self.write(args[0])}\\right]", _ATOM
        if func == "d" and len(args) == 1:
            return f"\\Delta {self._tight(args[0], _ATOM)}", _UNARY
        if func == "sqrt" and len(args) == 1:
            return f"\\sqrt{{{self.write(args[0])}}}", _ATOM
        if func == "abs" and len(args) == 1:
            return f"\\left|{self.write(args[0])}\\right|", _ATOM
        if func == "inv" and len(args) == 1:
            return f"{self._tight(args[0], _ATOM, calls=False)}^{{-1}}", _POWER
        if func == "transpose" and len(args) == 1:
            return f"{self._tight(args[0], _ATOM, calls=False)}^{{\\top}}", _POWER
        name = _PLAIN_CALLS.get(func, r"\operatorname{" + func.replace("_", r"\_") + "}")
        listed = [self.write(a) for a in args] + [f"{_word(k)}={self.write(v)}" for k, v in expr.kwargs]
        return (f"{name}\\left({', '.join(listed)}\\right)" if listed else name), _ATOM


def _leaf(expr: Expr) -> tuple[str, list[Expr], Expr | None] | None:
    """A name with any indexes and at most one time shift, such as inve[j](-1), as (name, index items, lag); otherwise None."""
    items: list[Expr] = []
    lag = None
    while True:
        if isinstance(expr, Index):
            items = list(expr.items) + items
            expr = expr.base
        elif isinstance(expr, Shift) and lag is None:
            lag, expr = expr.lag, expr.base
        elif isinstance(expr, Name):
            return expr.id, items, lag
        else:
            return None
