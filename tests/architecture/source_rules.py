"""The automatic architecture rules from PLAN.md, written as plain functions over source text.

Kept separate from the tests so the rules can be checked against deliberately bad source as well as the real package:
a checker that never fires would pass vacuously.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "mutatis"
DEMOS = REPO / "demos"

# Layers in dependency order: a layer may import only layers earlier in this list (and itself).
LAYER_ORDER = ["spec", "numerics", "dist", "compile", "solve", "measure", "quantities", "infer", "versions"]
TOP_LAYERS = {"library", "io", "web", "__main__"}  # may import any layer; nothing imports them, except the command line
ENTRY = "__main__"              # the command line: may import any layer, including the top ones
CONFIG = "_jax_config"          # JAX settings, importable by any layer that uses JAX
SELF_CONTAINED = {"numerics", "dist"}  # import nothing else of ours except CONFIG
SOLVE_CONTRACT = "mutatis.solve.contract"  # the only part of solve that infer may see
JAX_PACKAGES = {"jax", "jaxlib", "equinox", "optimistix", "lineax", "blackjax", "jaxtyping"}

# Model-family words that must not appear in identifiers in the core (PLAN.md, rule 3).
FAMILY_WORDS = {"var", "vars", "svar", "svars", "bvar", "bvars", "dsge", "dsges", "martin"}
FAMILY_EXEMPT_LAYERS = {"library", "io"}

CODE_SUFFIXES = {".py", ".pyi", ".pyx", ".ipynb", ".so"}


@dataclass(frozen=True)
class Violation:
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def module_name(path: Path) -> str:
    parts = path.relative_to(REPO).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def package_modules() -> list[tuple[str, str, bool]]:
    """Every module in the package as (module name, source, is_package)."""
    return [(module_name(p), p.read_text(), p.name == "__init__.py") for p in sorted(PACKAGE.rglob("*.py"))]


def layer_of(name: str) -> str | None:
    """The layer a fully qualified module name belongs to: 'numerics', '_jax_config', '' for the top-level package, or None if not ours."""
    parts = name.split(".")
    if parts[0] != "mutatis":
        return None
    return parts[1] if len(parts) > 1 else ""


def imported_names(module: str, source: str, is_package: bool) -> list[str]:
    """Fully qualified targets of every import in a module. `from a.b import c` gives 'a.b.c'."""
    package = module if is_package else module.rpartition(".")[0]
    targets = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            targets += [f"{base}.{alias.name}" for alias in node.names]
    return targets


def import_violations(module: str, source: str, is_package: bool) -> list[Violation]:
    """Check one module against the layering rules in PLAN.md (Layout)."""
    here = layer_of(module)
    found = []
    for target in imported_names(module, source, is_package):
        there = layer_of(target)
        if there is None:
            if here == "spec" and target.split(".")[0] in JAX_PACKAGES:
                found.append(Violation(module, f"spec must not use JAX, but imports {target}"))
            continue
        if there == here or here == ENTRY:
            continue
        if there in TOP_LAYERS:
            found.append(Violation(module, f"nothing may import {there}, but this imports {target}"))
        elif there == CONFIG:
            if here == "spec":
                found.append(Violation(module, "spec must not use JAX, but imports the JAX settings"))
        elif here == "":
            found.append(Violation(module, f"the top-level package must not import layers, but imports {target}"))
        elif here == "spec" or here in SELF_CONTAINED:
            found.append(Violation(module, f"{here} must import nothing else from mutatis, but imports {target}"))
        elif here == "infer" and there == "solve" and not target.startswith(SOLVE_CONTRACT):
            found.append(Violation(module, f"infer may see solve only through {SOLVE_CONTRACT}, but imports {target}"))
        elif here in TOP_LAYERS:
            continue
        elif here not in LAYER_ORDER:
            found.append(Violation(module, f"{here!r} is not a known layer; add it to LAYER_ORDER or TOP_LAYERS in the architecture rules"))
        elif there == "" or there not in LAYER_ORDER or LAYER_ORDER.index(there) > LAYER_ORDER.index(here):
            found.append(Violation(module, f"{here} may not import the later layer {there or 'mutatis'} ({target})"))
    return found


def _words(identifier: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    return [w.lower() for w in spaced.split("_") if w]


def defined_identifiers(source: str) -> list[str]:
    """Names this source defines: functions, classes, arguments, assignment targets, attributes it sets, import aliases."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.append(node.attr)
        elif isinstance(node, ast.alias) and node.asname:
            names.append(node.asname)
    return names


def family_name_violations(module: str, source: str) -> list[Violation]:
    """Check one module for model-family words in the identifiers it defines, and in its own name (PLAN.md, rule 3)."""
    if layer_of(module) in FAMILY_EXEMPT_LAYERS:
        return []
    found = []
    for name in [module.split(".")[-1], *defined_identifiers(source)]:
        bad = FAMILY_WORDS.intersection(_words(name))
        if bad:
            found.append(Violation(module, f"identifier {name!r} names a model family ({', '.join(sorted(bad))})"))
    return found


def code_files_in(folder: Path) -> list[Path]:
    """Files under a folder that hold program code (PLAN.md, rule 1: a demonstration contains no Python)."""
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in CODE_SUFFIXES)
