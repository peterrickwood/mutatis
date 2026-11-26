"""The expression language: parsing, shifts versus calls, spans, canonical text, errors."""

import pytest
import yaml

from mutatis.spec.expr import (Binary, Call, Compare, Date, Index, Name, Num, Over, ParseError, Range, Shift, Unary, parse_equation,
                               parse_expression, parse_requirement, to_text, walk)
from tests.spec.fixtures import MODEL_FILES, SKETCHES

EXAMPLES = [
    "mu = mu(-1) + eta",
    "y = c + B @ y(-1) + u",
    "p = x + a * E(p(+1))",
    "d(spend) = alpha0 + gamma * (spend(-1) - beta1 * income(-1)) + alpha1 * d(income) + e",
    "y = c + sum(l in lags: B[l] @ y(-l)) + sum(l in outside_lags: D[l] @ x(-l)) + chol(Sigma) @ Q @ eps",
    "c**(-sigma) = beta * E(c(+1)**(-sigma) * (1 + i) / (1 + pi(+1)))",
    "lambda_k[j] = phik * mubar^2 * ((1 + beta) * inve[j] - inve[j](-1) - beta * inve[j](+1) - beta * mu(+1) + mu) + lambda - upsilon",
    "RCR = 100 * (1 + NCR / 100) / (PTM / PTM(-4)) - 100",
    "abs(correlation(shock_estimate(eps[policy]), data(proxy))) - max(j in shocks except policy: abs(correlation(shock_estimate(eps[j]), data(proxy))))",
    "response(y[cash_rate], eps[policy], horizons=0..3)",
    "date < 2008Q2",
    "x = 1.5e-3 * .5 - 2.",
]


@pytest.mark.parametrize("text", EXAMPLES)
def test_canonical_text_parses_back_to_the_same_tree(text):
    tree = parse_expression(text)
    assert parse_expression(to_text(tree)) == tree


def test_every_equation_in_the_toy_models_and_sketches_parses():
    count = 0
    for path in MODEL_FILES + SKETCHES:
        for document in path.read_text().split("```yaml")[1:] if path.suffix == ".md" else [path.read_text()]:
            text = document.split("```")[0]
            for equation in _equations(yaml.safe_load(text)):
                parse_equation(equation)
                count += 1
    assert count >= 20


def _equations(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("equations", "equation") and isinstance(value, (list, str)):
                yield from ([value] if isinstance(value, str) else value)
            else:
                yield from _equations(value)
    elif isinstance(node, list):
        for item in node:
            yield from _equations(item)


def test_shift_versus_call():
    assert parse_expression("x(-1)") == Shift(Name("x"), Unary("-", Num(1)))
    assert parse_expression("inve[j](+1)") == Shift(Index(Name("inve"), (Name("j"),)), Unary("+", Num(1)))
    assert parse_expression("y(-l)") == Shift(Name("y"), Unary("-", Name("l")))
    assert parse_expression("log(x)") == Call("log", (Name("x"),))
    assert parse_expression("E(p(+1))") == Call("E", (Shift(Name("p"), Unary("+", Num(1))),))
    assert parse_expression("normal(0, sigma)") == Call("normal", (Num(0), Name("sigma")))
    assert parse_expression("student_t(4)", shifts=False) == Call("student_t", (Num(4),))
    assert isinstance(parse_expression("student_t(4)"), Shift)  # where shifts are allowed, a one-argument unknown name is a shift


def test_structure():
    assert parse_expression("a - b - c") == Binary("-", Binary("-", Name("a"), Name("b")), Name("c"))
    assert parse_expression("a ^ b ^ c") == Binary("^", Name("a"), Binary("^", Name("b"), Name("c")))  # powers group to the right
    assert parse_expression("-x^2") == Unary("-", Binary("^", Name("x"), Num(2)))
    assert parse_expression("a ** b") == Binary("^", Name("a"), Name("b"))
    assert parse_expression("0..3") == Range(Num(0), Num(3))
    assert parse_expression("date < 2008Q2") == Compare("<", Name("date"), Date(2008, 2))
    over = parse_expression("max(j in shocks except policy, other: x[j])")
    assert over == Over("max", "j", "shocks", ("policy", "other"), Index(Name("x"), (Name("j"),)))
    call = parse_expression("gamma(mean=1.5, sd=0.4)")
    assert call == Call("gamma", (), (("mean", Num(1.5)), ("sd", Num(0.4))))


def test_spans_point_at_the_text():
    text = "k = (1 - delta) * k(-1) + 0.25 * invest"
    numbers = [node for node in walk(parse_expression(text)) if isinstance(node, Num)]
    assert [text[n.span[0]:n.span[1]] for n in numbers] == ["1", "1", "0.25"]


def test_equation_and_requirement():
    equation = parse_equation("y = a + b")
    assert equation.lhs == Name("y") and equation.rhs == Binary("+", Name("a"), Name("b"))
    with pytest.raises(ParseError):
        parse_equation("y + a")
    assert parse_requirement(">= 0") == (">=", Num(0))
    with pytest.raises(ParseError):
        parse_requirement("= 3")


@pytest.mark.parametrize("text, position", [("x = (a + b", 10), ("x = a $ b", 6), ("x = a + ", 8), ("f(a=1, 2)", 7)])
def test_errors_point_at_the_problem(text, position):
    with pytest.raises(ParseError) as error:
        parse_expression(text)
    assert error.value.position == position
