"""The specification: a model as pure data.

A model is a set of blocks joined by shared variables, plus data links, restrictions, sets of names that variables can run
over, and the expectations setting. Everything here is plain, immutable data: no numbers are computed in this layer. Equations,
distributions and restrictions are kept as the text the person wrote; `mutatis.spec.expr` parses them when needed.

Names. Inside a block, an equation refers to the block's own parameters, shocks and definitions, and to model-level parameters,
by their bare names. From outside, a parameter is named by a target: `household.beta` (a block's), `y.sigma_eps` (a data
link's) or `beta` (the model's).
"""

from dataclasses import dataclass, field, replace
from typing import Any

from mutatis.spec.expr import Index, Name, parse_expression

NO_PRIOR = "none"  # a parameter restricted with no prior behind it: results are reported as ranges


def freeze(value: Any) -> Any:
    """Lists to tuples, recursively, so fixed matrix values compare and hash like numbers."""
    return tuple(freeze(v) for v in value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True)
class Parameter:
    """Exactly one of `fixed` or `prior` is set. `prior` is distribution text, or "none" (no prior: report ranges), or "flat"."""

    fixed: Any = None
    prior: str | None = None
    shape: tuple[str, ...] = ()            # names of the sets it runs over; empty for a single number
    domain: str | None = None              # positive, unit_interval, positive_definite, orthogonal; None means any real number
    separate: dict[str, "Parameter"] = field(default_factory=dict)  # template instances with a parameter of their own

    def __post_init__(self):
        if (self.fixed is None) == (self.prior is None):
            raise ValueError("a parameter needs exactly one of 'fixed' or 'prior'")
        object.__setattr__(self, "fixed", freeze(self.fixed))
        object.__setattr__(self, "shape", tuple(self.shape))

    @property
    def kind(self) -> str:
        return "fixed" if self.fixed is not None else "no_prior" if self.prior == NO_PRIOR else "prior"


@dataclass(frozen=True)
class Shock:
    distribution: str
    shape: tuple[str, ...] = ()


@dataclass(frozen=True)
class Block:
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: dict[str, Parameter] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    shocks: dict[str, Shock] = field(default_factory=dict)
    initial: dict[str, str] = field(default_factory=dict)
    steady_state: dict[str, str] = field(default_factory=dict)   # resting values of outputs: where finding the steady state starts
    equations: tuple[str, ...] = ()
    code: str | None = None
    over: tuple[str, str] | None = None    # a template: (index name, set name), one instance per element of the set
    overrides: dict[str, "Block"] = field(default_factory=dict)  # template instances replaced by a block of their own

    def __post_init__(self):
        for name in ("inputs", "outputs", "equations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.over is not None:
            object.__setattr__(self, "over", tuple(self.over))
        if self.code is not None and self.equations:
            raise ValueError("a block has either equations or code, not both")


@dataclass(frozen=True)
class DataLink:
    """One data series. `observes`: the model must explain it. `feeds`: it supplies an outside variable. Neither: it is used only
    by restrictions. `error` is distribution text, or None when the series is observed exactly."""

    observes: str | None = None
    feeds: str | None = None
    error: str | None = None
    parameters: dict[str, Parameter] = field(default_factory=dict)
    transform: str | None = None

    def __post_init__(self):
        if self.observes is not None and self.feeds is not None:
            raise ValueError("a data series either is observed or feeds an outside variable, not both")
        if self.feeds is not None and self.error is not None:
            raise ValueError("a series that feeds an outside variable is taken as given, so it has no measurement error")

    @property
    def role(self) -> str:
        return "observes" if self.observes is not None else "feeds" if self.feeds is not None else "auxiliary"


@dataclass(frozen=True)
class Restriction:
    quantity: str
    require: str


@dataclass(frozen=True)
class Model:
    name: str
    time: str = "quarterly"
    sets: dict[str, tuple] = field(default_factory=dict)
    calendar: dict[str, str] = field(default_factory=dict)
    expectations: str = "rational"
    parameters: dict[str, Parameter] = field(default_factory=dict)   # model-level, shared by every block
    blocks: dict[str, Block] = field(default_factory=dict)
    data: dict[str, DataLink] = field(default_factory=dict)
    restrictions: dict[str, Restriction] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "sets", {name: tuple(elements) for name, elements in self.sets.items()})

    # Looking things up by target -------------------------------------------------------------------------------------------

    def parameter(self, target: str) -> Parameter:
        owner, name = split_target(target)
        try:
            return self._parameters_of(owner)[name]
        except KeyError:
            raise KeyError(f"no parameter {target!r}") from None

    def _parameters_of(self, owner: str | None) -> dict[str, Parameter]:
        if owner is None:
            return self.parameters
        if owner in self.blocks:
            return self.blocks[owner].parameters
        if owner in self.data:
            return self.data[owner].parameters
        raise KeyError(f"no block or data series named {owner!r}")

    def with_parameter(self, target: str, parameter: Parameter | None) -> "Model":
        """A copy with one parameter set, added, or (with None) removed."""
        owner, name = split_target(target)
        parameters = dict(self._parameters_of(owner))
        if parameter is None:
            del parameters[name]
        else:
            parameters[name] = parameter
        if owner is None:
            return replace(self, parameters=parameters)
        if owner in self.blocks:
            return replace(self, blocks={**self.blocks, owner: replace(self.blocks[owner], parameters=parameters)})
        return replace(self, data={**self.data, owner: replace(self.data[owner], parameters=parameters)})


def split_target(target: str) -> tuple[str | None, str]:
    owner, _, name = target.rpartition(".")
    return (owner or None), name


def port(reference: str) -> tuple[str, tuple[str, ...]]:
    """A port such as 'y[series]', 'x' or 'lambda_k[j]', as (variable name, the names in brackets)."""
    expr = parse_expression(reference)
    if isinstance(expr, Name):
        return expr.id, ()
    if isinstance(expr, Index) and isinstance(expr.base, Name) and all(isinstance(item, Name) for item in expr.items):
        return expr.base.id, tuple(item.id for item in expr.items)
    raise ValueError(f"a port is a variable name, optionally followed by set or element names in brackets: {reference!r}")
