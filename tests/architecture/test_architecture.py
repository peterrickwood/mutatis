"""The automatic architecture checks from PLAN.md, run with the fast tests."""

import pytest

from tests.architecture import source_rules as rules


def test_package_follows_import_rules():
    found = [v for m, src, is_pkg in rules.package_modules() for v in rules.import_violations(m, src, is_pkg)]
    assert not found, "\n".join(map(str, found))


def test_core_names_no_model_family():
    found = [v for m, src, _ in rules.package_modules() for v in rules.family_name_violations(m, src)]
    assert not found, "\n".join(map(str, found))


def test_demonstrations_contain_no_code():
    found = rules.code_files_in(rules.DEMOS)
    assert not found, "demonstrations must contain no code: " + ", ".join(str(p.relative_to(rules.REPO)) for p in found)


# The checks above pass vacuously if a checker never fires. These feed each checker source that breaks a rule.

@pytest.mark.parametrize(
    "module, source, is_pkg, expected",
    [
        ("mutatis.numerics.kalman", "from mutatis.solve import contract", False, "numerics must import nothing else"),
        ("mutatis.dist.normal", "import mutatis.spec.model", False, "dist must import nothing else"),
        ("mutatis.spec.model", "import jax.numpy as jnp", False, "spec must not use JAX"),
        ("mutatis.spec.model", "from mutatis import _jax_config", False, "spec must not use JAX"),
        ("mutatis.spec.model", "from mutatis.numerics import rootfind", False, "spec must import nothing else"),
        ("mutatis.infer.posterior", "from mutatis.solve.linear import solve", False, "only through mutatis.solve.contract"),
        ("mutatis.infer.posterior", "from ..solve import linear", False, "only through mutatis.solve.contract"),
        ("mutatis.solve.linear", "from mutatis.infer import posterior", False, "may not import the later layer infer"),
        ("mutatis.compile", "from mutatis.library import household", True, "nothing may import library"),
        ("mutatis.versions.store", "import mutatis.io.dynare", False, "nothing may import io"),
        ("mutatis.versions.run", "from mutatis.web import server", False, "nothing may import web"),
        ("mutatis.library.household", "from mutatis.web.graph import block_graph", False, "nothing may import web"),
        ("mutatis", "from mutatis import numerics", True, "top-level package must not import layers"),
        ("mutatis.extra.thing", "from mutatis.spec import model", False, "not a known layer"),
    ],
)
def test_import_checker_catches(module, source, is_pkg, expected):
    found = rules.import_violations(module, source, is_pkg)
    assert any(expected in v.message for v in found), found


@pytest.mark.parametrize(
    "module, source, is_pkg",
    [
        ("mutatis.infer.posterior", "from mutatis.solve.contract import SolvedModel", False),
        ("mutatis.infer.posterior", "from ..solve import contract", False),
        ("mutatis.solve.linear", "from mutatis import _jax_config\nfrom mutatis.numerics import linear_re", False),
        ("mutatis.library.household", "from mutatis.infer import posterior", False),
        ("mutatis.spec.model", "import sympy\nfrom . import equation", False),
        ("mutatis.web.estimates", "from mutatis.versions.run import load_result\nfrom mutatis.infer.states import one_step_predictions", False),
        ("mutatis.__main__", "from mutatis.web.server import serve", False),
    ],
)
def test_import_checker_allows(module, source, is_pkg):
    assert rules.import_violations(module, source, is_pkg) == []


@pytest.mark.parametrize(
    "module, source, expected_word",
    [
        ("mutatis.infer.posterior", "def svar_prior(): pass", "svar"),
        ("mutatis.solve.linear", "class DsgeSolver: pass", "dsge"),
        ("mutatis.compile.blocks", "n_var = 3", "var"),
        ("mutatis.quantities.response", "def f(martin_block): pass", "martin"),
        ("mutatis.solve.svar", "x = 1", "svar"),
    ],
)
def test_family_name_checker_catches(module, source, expected_word):
    found = rules.family_name_violations(module, source)
    assert any(expected_word in v.message for v in found), found


@pytest.mark.parametrize(
    "module, source",
    [
        ("mutatis.numerics.stats", "variance = jnp.var(x)\nn_variables = 3"),  # using someone else's `var` is fine
        ("mutatis.library.svar", "def svar_block(): pass"),                       # the library may name families
    ],
)
def test_family_name_checker_allows(module, source):
    assert rules.family_name_violations(module, source) == []


def test_code_file_checker_catches(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "spec.yaml").write_text("model: x\n")
    (tmp_path / "demo" / "fudge.py").write_text("x = 1\n")
    assert [p.name for p in rules.code_files_in(tmp_path)] == ["fudge.py"]
