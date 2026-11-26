"""The expression language used in equations, distributions, restrictions and calendar definitions.

Text is parsed into a small syntax tree of frozen dataclasses. Every node records where it came from in the text (`span`), so
that tools can point at, or rewrite, an exact piece of an equation (naming a number, for example) without disturbing the rest.

Syntax, by example:

    c(-1), p(+1), inve[j](-1)        a variable last period, next period; an element of a variable, last period
    E(p(+1))                         an expectation term
    d(x)                             x - x(-1)
    y[series], y[a], B[l]            a whole vector, one element, one slice
    B @ y                            matrix product
    sum(l in lags: B[l] @ y(-l))     a sum over the elements of a set
    max(j in shocks except policy: abs(x[j]))
    normal(0, sigma), gamma(mean=1.5, sd=0.4)
    horizons=0..3                    a range of whole numbers
    date < 2008Q2                    comparisons and dates, for calendar variables and restrictions

`a(k)` is read as "a, shifted k periods" when `a` is not a known function name and k is a single argument; otherwise it is a
function call. Parsing with `shifts=False` turns this off, for contexts such as distributions where `student_t(4)` is a call.
"""

import re
from dataclasses import dataclass, field

# Function names that are calls, never time shifts. Which of them are allowed where is checked later, not by the parser.
FUNCTIONS = frozenset({
    "E", "d", "log", "exp", "sqrt", "abs", "chol", "inv", "transpose", "sum", "max", "min",
    "response", "shock_equation", "shock_estimate", "correlation", "data",
})


@dataclass(frozen=True)
class Expr:
    span: tuple[int, int] = field(default=(0, 0), compare=False, repr=False, kw_only=True)


@dataclass(frozen=True)
class Num(Expr):
    value: float | int
    text: str = field(default="", compare=False)


@dataclass(frozen=True)
class Name(Expr):
    id: str


@dataclass(frozen=True)
class Date(Expr):
    year: int
    quarter: int


@dataclass(frozen=True)
class Index(Expr):
    base: Expr
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Shift(Expr):
    base: Expr
    lag: Expr  # a whole number, possibly through a set element: x(-1), y(-l)


@dataclass(frozen=True)
class Call(Expr):
    func: str
    args: tuple[Expr, ...] = ()
    kwargs: tuple[tuple[str, Expr], ...] = ()


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: str  # + - * / ^ @
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Compare(Expr):
    op: str  # = < <= > >= == !=
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Range(Expr):
    start: Expr
    stop: Expr


@dataclass(frozen=True)
class Over(Expr):
    """sum, max or min over the elements of a set: sum(l in lags: body). `index` is the name that runs over the set (l)."""

    func: str
    index: str
    set: str
    exclude: tuple[str, ...]
    body: Expr


class ParseError(ValueError):
    def __init__(self, message: str, text: str, position: int):
        pointer = " " * position + "^"
        super().__init__(f"{message}\n  {text}\n  {pointer}")
        self.position = position


_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<date>\d{4}Q[1-4]\b)
  | (?P<number>(?:\d+(?:\.(?!\.)\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\*\*|\.\.|<=|>=|==|!=|[-+*/^@()\[\],:=<>])
""", re.VERBOSE)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int


def _tokenize(text: str) -> list[_Token]:
    tokens, position = [], 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match:
            raise ParseError(f"unexpected character {text[position]!r}", text, position)
        if match.lastgroup != "space":
            tokens.append(_Token(match.lastgroup, match.group(), match.start(), match.end()))
        position = match.end()
    tokens.append(_Token("end", "", len(text), len(text)))
    return tokens


_BINARY = {"=": 1, "<": 1, "<=": 1, ">": 1, ">=": 1, "==": 1, "!=": 1, "..": 2, "+": 3, "-": 3, "*": 4, "/": 4, "@": 4, "^": 6, "**": 6}
_UNARY_POWER = 5
_COMPARE = {"=", "<", "<=", ">", ">=", "==", "!="}


class _Parser:
    def __init__(self, text: str, shifts: bool):
        self.text, self.shifts = text, shifts
        self.tokens = _tokenize(text)
        self.i = 0

    @property
    def peek(self) -> _Token:
        return self.tokens[self.i]

    def next(self) -> _Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def expect(self, text: str) -> _Token:
        token = self.next()
        if token.text != text:
            raise ParseError(f"expected {text!r}, found {token.text or 'end of text'!r}", self.text, token.start)
        return token

    def fail(self, message: str, token: _Token):
        raise ParseError(message, self.text, token.start)

    def parse(self) -> Expr:
        expr = self.expression(0)
        if self.peek.kind != "end":
            self.fail(f"unexpected {self.peek.text!r}", self.peek)
        return expr

    def expression(self, min_power: int) -> Expr:
        left = self.prefix()
        while True:
            token = self.peek
            power = _BINARY.get(token.text) if token.kind == "op" else None
            if power is None or power < min_power:
                return left
            self.next()
            right_power = power if token.text in ("^", "**") else power + 1  # powers group to the right
            right = self.expression(right_power)
            span = (left.span[0], right.span[1])
            if token.text in _COMPARE:
                left = Compare(token.text, left, right, span=span)
            elif token.text == "..":
                left = Range(left, right, span=span)
            else:
                left = Binary("^" if token.text == "**" else token.text, left, right, span=span)

    def prefix(self) -> Expr:
        token = self.next()
        if token.text in ("-", "+"):
            operand = self.expression(_UNARY_POWER)
            return self.postfix(Unary(token.text, operand, span=(token.start, operand.span[1])))
        if token.kind == "number":
            value = float(token.text) if any(c in token.text for c in ".eE") else int(token.text)
            return self.postfix(Num(value, token.text, span=(token.start, token.end)))
        if token.kind == "date":
            return Date(int(token.text[:4]), int(token.text[5]), span=(token.start, token.end))
        if token.kind == "name":
            return self.postfix(Name(token.text, span=(token.start, token.end)))
        if token.text == "(":
            inner = self.expression(0)
            close = self.expect(")")
            return self.postfix(_respan(inner, (token.start, close.end)))
        self.fail(f"unexpected {token.text or 'end of text'!r}", token)

    def postfix(self, base: Expr) -> Expr:
        while True:
            if self.peek.text == "[":
                self.next()
                items = [self.expression(0)]
                while self.peek.text == ",":
                    self.next()
                    items.append(self.expression(0))
                close = self.expect("]")
                base = Index(base, tuple(items), span=(base.span[0], close.end))
            elif self.peek.text == "(" and isinstance(base, (Name, Index)):
                base = self.call(base)
            else:
                return base

    def call(self, base: Expr) -> Expr:
        self.expect("(")
        func = base.id if isinstance(base, Name) else None
        if func in ("sum", "max", "min") and self.tokens[self.i + 1].text == "in":
            return self.over(func, base)
        args, kwargs = [], []
        while self.peek.text != ")":
            if self.peek.kind == "name" and self.tokens[self.i + 1].text == "=":
                name = self.next().text
                self.next()
                kwargs.append((name, self.expression(2)))  # above comparisons, so "=" ends the value
            else:
                if kwargs:
                    self.fail("positional argument after keyword argument", self.peek)
                args.append(self.expression(0))
            if self.peek.text != ")":
                self.expect(",")
        close = self.expect(")")
        span = (base.span[0], close.end)
        if self.shifts and func not in FUNCTIONS and len(args) == 1 and not kwargs:
            return Shift(base, args[0], span=span)
        if func is None:
            self.fail("only a named function can be called", self.tokens[self.i - 1])
        return Call(func, tuple(args), tuple(kwargs), span=span)

    def over(self, func: str, base: Expr) -> Over:
        index = self.next().text
        self.expect("in")
        set_name = self.next()
        if set_name.kind != "name":
            self.fail("expected the name of a set", set_name)
        exclude = []
        if self.peek.text == "except":
            self.next()
            exclude.append(self.next().text)
            while self.peek.text == ",":
                self.next()
                exclude.append(self.next().text)
        self.expect(":")
        body = self.expression(0)
        close = self.expect(")")
        return Over(func, index, set_name.text, tuple(exclude), body, span=(base.span[0], close.end))


def _respan(expr: Expr, span: tuple[int, int]) -> Expr:
    """The same node, with a span widened to include surrounding brackets."""
    return type(expr)(**{**{f: getattr(expr, f) for f in expr.__dataclass_fields__ if f != "span"}, "span": span})


def parse_expression(text: str, shifts: bool = True) -> Expr:
    return _Parser(text, shifts).parse()


@dataclass(frozen=True)
class Equation:
    lhs: Expr
    rhs: Expr
    text: str = field(compare=False, repr=False)


def parse_equation(text: str) -> Equation:
    expr = parse_expression(text)
    if not (isinstance(expr, Compare) and expr.op == "="):
        raise ParseError("an equation needs exactly one '=' between its two sides", text, 0)
    return Equation(expr.left, expr.right, text)


def parse_requirement(text: str) -> tuple[str, Expr]:
    """A requirement such as '>= 0' or '< 2.5': the comparison and the bound."""
    expr = parse_expression("_ " + text)
    if not (isinstance(expr, Compare) and expr.op != "="):
        raise ParseError("a requirement is a comparison such as '>= 0'", text, 0)
    return expr.op, expr.right


def to_text(expr: Expr) -> str:
    """Canonical text for a syntax tree. parse_expression(to_text(e)) == e."""
    match expr:
        case Num(text=text, value=value):
            return text or repr(value)
        case Name(id=name):
            return name
        case Date(year=year, quarter=quarter):
            return f"{year}Q{quarter}"
        case Index(base=base, items=items):
            return f"{to_text(base)}[{', '.join(map(to_text, items))}]"
        case Shift(base=base, lag=lag):
            return f"{to_text(base)}({to_text(lag)})"
        case Call(func=func, args=args, kwargs=kwargs):
            return f"{func}({', '.join([*map(to_text, args), *(f'{k}={to_text(v)}' for k, v in kwargs)])})"
        case Unary(op=op, operand=operand):
            return f"({op}{to_text(operand)})"
        case Binary(op=op, left=left, right=right) | Compare(op=op, left=left, right=right):
            return f"({to_text(left)} {op} {to_text(right)})"
        case Range(start=start, stop=stop):
            return f"{to_text(start)}..{to_text(stop)}"
        case Over(func=func, index=index, set=set_name, exclude=exclude, body=body):
            excluded = f" except {', '.join(exclude)}" if exclude else ""
            return f"{func}({index} in {set_name}{excluded}: {to_text(body)})"
    raise TypeError(f"not an expression: {expr!r}")


def walk(expr: Expr):
    """Every node in the tree, parents before children."""
    yield expr
    match expr:
        case Index(base=base, items=items):
            yield from walk(base)
            for item in items:
                yield from walk(item)
        case Shift(base=base, lag=lag):
            yield from walk(base)
            yield from walk(lag)
        case Call(args=args, kwargs=kwargs):
            for arg in args:
                yield from walk(arg)
            for _, value in kwargs:
                yield from walk(value)
        case Unary(operand=operand):
            yield from walk(operand)
        case Binary(left=left, right=right) | Compare(left=left, right=right):
            yield from walk(left)
            yield from walk(right)
        case Range(start=start, stop=stop):
            yield from walk(start)
            yield from walk(stop)
        case Over(body=body):
            yield from walk(body)
