"""Numbers written directly in equations.

Each is an assumption fixed at a value, even though it has no name. `find_numbers` lists them so that they can be shown,
included in sensitivity checks, and turned into named parameters by the `name_number` edit. Numbers that state structure rather
than size are left out: time shifts such as the 1 in x(-1), element positions, ranges and dates.
"""

from dataclasses import dataclass

from mutatis.spec.expr import Binary, Call, Compare, Expr, Index, Num, Over, Range, Shift, Unary, parse_expression
from mutatis.spec.model import Model


@dataclass(frozen=True)
class NumberSite:
    block: str
    equation: int
    start: int
    end: int
    text: str
    value: float | int


def numbers_in(expr: Expr):
    """Number nodes in an expression, leaving out time shifts, indices and ranges."""
    match expr:
        case Num():
            yield expr
        case Shift(base=base):
            yield from numbers_in(base)
        case Index(base=base):
            yield from numbers_in(base)
        case Range():
            return
        case Call(args=args, kwargs=kwargs):
            for arg in args:
                yield from numbers_in(arg)
            for _, value in kwargs:
                yield from numbers_in(value)
        case Unary(operand=operand):
            yield from numbers_in(operand)
        case Binary(left=left, right=right) | Compare(left=left, right=right):
            yield from numbers_in(left)
            yield from numbers_in(right)
        case Over(body=body):
            yield from numbers_in(body)


def find_numbers(model: Model) -> list[NumberSite]:
    sites = []
    for name, block in model.blocks.items():
        for index, equation in enumerate(block.equations):
            for number in numbers_in(parse_expression(equation)):
                sites.append(NumberSite(name, index, number.span[0], number.span[1], equation[number.span[0]:number.span[1]], number.value))
    return sites
