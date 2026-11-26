"""Syntax-tree nodes that exist only after a specification is compiled: names resolved to what they refer to."""

from dataclasses import dataclass

from mutatis.spec.expr import Expr

VARIABLE, OUTSIDE, PARAMETER, SHOCK = "variable", "outside", "parameter", "shock"


@dataclass(frozen=True)
class Ref(Expr):
    """A resolved name. `key` is global: a variable's name, `block.parameter`, `block.shock`, or a model-level parameter's name."""

    kind: str
    key: str


@dataclass(frozen=True)
class Element(Expr):
    """One element of a set, used to pick an entry of a vector: the `a` in y[a], or a lag 1 in B[1]."""

    value: str | int


@dataclass(frozen=True)
class Whole(Expr):
    """Every element of a set: the `series` in y[series]."""

    set: str
