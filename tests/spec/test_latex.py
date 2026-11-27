"""Expressions written as LaTeX: symbols, periods, brackets only where needed, and every equation in the repository."""

import random
import re
from pathlib import Path

import pytest
import yaml

from mutatis.spec.expr import parse_expression
from mutatis.spec.latex import distribution_latex, symbol, text_to_latex, to_latex
from tests.spec.fixtures import MODEL_FILES, REPO, SKETCHES

DEMO_FILES = sorted((REPO / "demos").glob("*/model.yaml"))


@pytest.mark.parametrize("name, latex", [
    ("alpha", r"\alpha"), ("Sigma", r"\Sigma"), ("eps", r"\varepsilon"), ("alpha0", r"\alpha_{0}"), ("psi1", r"\psi_{1}"), ("B2", "B_{2}"),
    ("sigma_eta", r"\sigma_{\eta}"), ("rho_R", r"\rho_{R}"), ("x_obs", r"x_{\mathit{obs}}"), ("cash_rate", r"\mathit{cash\_rate}"),
    ("piA", r"\mathit{piA}"), ("other1", r"\mathit{other1}"), ("k", "k"),
])
def test_symbols(name, latex):
    assert symbol(name) == latex


@pytest.mark.parametrize("text, timed, latex", [
    ("x = rho * x(-1) + e", {"x", "e"}, r"x_{t} = \rho\, x_{t-1} + e_{t}"),
    ("p = x + a * E(p(+1))", {"p", "x"}, r"p_{t} = x_{t} + a\, \mathbb{E}_{t}\left[p_{t+1}\right]"),
    ("k[j] = (1 - delta) * k[j](-1) + invest[j]", {"k", "invest"}, r"k_{j,t} = \left(1 - \delta\right)\, k_{j,t-1} + \mathit{invest}_{j,t}"),
    ("y = c + sum(l in lags: B[l] @ y(-l)) + chol(Sigma) @ eps", {"y", "eps"},
     r"y_{t} = c + \sum_{l \in \mathit{lags}} B_{l}\, y_{t-l} + \operatorname{chol}\left(\Sigma\right)\, \varepsilon_{t}"),
    ("d(spend) = alpha0 + e", {"spend", "e"}, r"\Delta \mathit{spend}_{t} = \alpha_{0} + e_{t}"),
    ("y[cash_rate]", {"y"}, r"y_{\mathit{cash\_rate},t}"),
    ("1 / c = beta * E(1 / c(+1))", {"c"}, r"\frac{1}{c_{t}} = \beta\, \mathbb{E}_{t}\left[\frac{1}{c_{t+1}}\right]"),
    ("max(j in shocks except policy: abs(x[j]))", set(), r"\max_{j \in \mathit{shocks} \setminus \{\mathit{policy}\}} \left|x_{j}\right|"),
    ("response(y[gdp], eps[policy], horizons=0..20)", set(),
     r"\operatorname{response}\left(y_{\mathit{gdp}}, \varepsilon_{\mathit{policy}}, \mathrm{horizons}=0, \dots, 20\right)"),
    ("date < 2008Q2", set(), r"\mathit{date} < \mathrm{2008Q2}"),
    ("x >= 1e-6", set(), r"x \ge 10^{-6}"),
    ("2.5e3 * x", set(), r"2.5 \times 10^{3}\, x"),
    ("inv(A) @ transpose(B)", set(), r"A^{-1}\, B^{\top}"),
    ("sqrt(x) + log(y) + exp(-z)", set(), r"\sqrt{x} + \log\left(y\right) + \exp\left(-z\right)"),
])
def test_equations(text, timed, latex):
    assert text_to_latex(text, timed) == latex


@pytest.mark.parametrize("text, latex", [
    ("a - (b - c)", r"a - \left(b - c\right)"), ("a - (b + c)", r"a - \left(b + c\right)"), ("a + (b - c)", "a + b - c"),
    ("(a + b) * c", r"\left(a + b\right)\, c"), ("a * (b * c)", r"a\, b\, c"), ("a + (-b)", r"a + \left(-b\right)"), ("a * -b", r"a\, \left(-b\right)"),
    ("-x ^ 2", "-x^{2}"), ("(-x) ^ 2", r"\left(-x\right)^{2}"), ("(a / b) ^ c", r"\left(\frac{a}{b}\right)^{c}"), ("(a * b) ^ 2", r"\left(a\, b\right)^{2}"),
    ("log(x) ^ 2", r"\left(\log\left(x\right)\right)^{2}"), ("a ^ b ^ c", "a^{b^{c}}"), ("(a ^ b) ^ c", r"\left(a^{b}\right)^{c}"), ("2 * 3", r"2 \cdot 3"),
    ("x * 0.5", r"x \cdot 0.5"), ("(a / b) * c", r"\frac{a}{b}\, c"), ("2 * sum(i in s: x[i])", r"2\, \left(\sum_{i \in s} x_{i}\right)"),
])
def test_brackets_only_where_needed(text, latex):
    assert text_to_latex(text) == latex


def test_distributions():
    assert distribution_latex("uniform") == r"\operatorname{uniform}"
    assert distribution_latex("gamma(mean=1.5, sd=0.4)") == r"\operatorname{gamma}\left(\mathrm{mean}=1.5, \mathrm{sd}=0.4\right)"
    assert distribution_latex("student_t(4, 0, sigma)") == r"\operatorname{student\_t}\left(4, 0, \sigma\right)"   # a call, not a shift


def _as_python(latex: str) -> str:
    """Arithmetic LaTeX as written by `to_latex`, back into Python, so the two can be evaluated and compared."""
    def braced(text, start):
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                return text[start + 1:i], i + 1
        raise ValueError(text)
    while r"\frac" in latex:
        at = latex.index(r"\frac")
        top, after = braced(latex, at + 5)
        bottom, end = braced(latex, after)
        latex = f"{latex[:at]}(({top})/({bottom})){latex[end:]}"
    while "^{" in latex:
        at = latex.index("^{")
        power, end = braced(latex, at + 1)
        latex = f"{latex[:at]}**({power}){latex[end:]}"
    return latex.replace(r"\left(", "(").replace(r"\right)", ")").replace(r"\, ", "*").replace(r" \cdot ", "*")


def _random_tree(rng: random.Random, depth: int) -> str:
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(["a", "b", "c", "2", "3"])
    kind = rng.random()
    if kind < 0.15:
        return f"(-{_random_tree(rng, depth - 1)})"
    op = rng.choice(["+", "-", "*", "/", "^"])
    if op == "^":   # keep powers small and positive so every value is a real number
        return f"({_random_tree(rng, depth - 1)}) ^ {rng.choice(['2', '3', '(-1)', '(a - 1)'])}"
    return f"({_random_tree(rng, depth - 1)} {op} {_random_tree(rng, depth - 1)})"


def test_the_brackets_keep_the_meaning():
    """For many random expressions, the LaTeX (read back as arithmetic) has the same value as the expression itself."""
    rng = random.Random(0)
    values = {"a": 1.7, "b": -0.6, "c": 2.3}
    checked = 0
    for _ in range(400):
        text = _random_tree(rng, 4)
        try:
            expected = eval(text.replace("^", "**"), {}, dict(values))
        except (ZeroDivisionError, OverflowError):
            continue
        if isinstance(expected, complex) or abs(expected) > 1e8:
            continue
        written = to_latex(parse_expression(text))
        assert eval(_as_python(written), {}, dict(values)) == pytest.approx(expected, rel=1e-12, abs=1e-12), (text, written)
        checked += 1
    assert checked > 300


def _equations_in(path: Path):
    documents = path.read_text().split("```yaml")[1:] if path.suffix == ".md" else [path.read_text()]
    for document in documents:
        yield from _find(yaml.safe_load(document.split("```")[0]))


def _find(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("equations", "equation") and isinstance(value, (list, str)):
                yield from ([value] if isinstance(value, str) else value)
            else:
                yield from _find(value)
    elif isinstance(node, list):
        for item in node:
            yield from _find(item)


def test_every_equation_in_the_repository_is_written_with_balanced_brackets():
    count = 0
    for path in MODEL_FILES + SKETCHES + DEMO_FILES:
        for equation in _equations_in(path):
            latex = text_to_latex(equation, {"x", "y"})
            assert latex.count("{") == latex.count("}"), (equation, latex)
            assert latex.count(r"\left") == latex.count(r"\right"), (equation, latex)
            assert re.search(r"[=]", latex), (equation, latex)
            count += 1
    assert count >= 25
