"""Edits: the fixed list of changes that turn one version of a model into the next.

Each edit is a pure function from model to model (`apply`), and is plain data that can be written out (`to_dict`) and read back
(`edit_from_dict`). An edit checks only its own preconditions (that its target exists, that a replacement block has the same
ports); whether the whole model still makes sense is for `mutatis.spec.checks`.

In YAML an edit list looks like:

    - set_prior: {target: household.sigma, prior: "gamma(mean=1.5, sd=0.4)"}
    - set_prior: {target: economy.Q, prior: none}
    - extend_set: {set: production_sectors, add: [h]}
"""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, ClassVar

from mutatis.spec.expr import parse_expression
from mutatis.spec.model import Block, DataLink, Model, Parameter, Restriction, port, split_target
from mutatis.spec.numbers import numbers_in


class EditError(ValueError):
    pass


_REGISTRY: dict[str, type["Edit"]] = {}


@dataclass(frozen=True)
class Edit:
    op: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.op] = cls

    def apply(self, model: Model) -> Model:
        raise NotImplementedError

    def to_dict(self) -> dict:
        from mutatis.spec import yaml_io  # imported here: the YAML layer imports this module's neighbours

        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Block):
                value = yaml_io.block_to_dict(value)
            elif isinstance(value, DataLink):
                value = yaml_io.data_link_to_dict(value)
            elif isinstance(value, Parameter):
                value = yaml_io.parameter_to_dict(value)
            elif isinstance(value, Restriction):
                value = asdict(value)
            elif isinstance(value, tuple):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            if value is not None:
                out[f.name] = value
        return {self.op: out}


def edit_from_dict(value: dict) -> Edit:
    from mutatis.spec import yaml_io

    if not (isinstance(value, dict) and len(value) == 1):
        raise EditError(f"an edit is a mapping with one key, the edit's name: {value!r}")
    (op, args), = value.items()
    if op not in _REGISTRY:
        raise EditError(f"unknown edit {op!r}; known edits: {sorted(_REGISTRY)}")
    cls = _REGISTRY[op]
    args = dict(args or {})
    names = {f.name for f in fields(cls)}
    unknown = set(args) - names
    if unknown:
        raise EditError(f"{op}: unknown argument(s) {sorted(unknown)}; allowed: {sorted(names)}")
    for f in fields(cls):
        if f.name not in args:
            continue
        if f.type in ("Block", Block):
            args[f.name] = yaml_io.block_from_dict(f"{op}.{f.name}", args[f.name])
        elif f.type in ("DataLink", DataLink):
            args[f.name] = yaml_io.data_link_from_dict(f"{op}.{f.name}", args[f.name])
        elif f.type in ("Parameter", Parameter):
            args[f.name] = yaml_io.parameter_from_dict(f"{op}.{f.name}", args[f.name])
        elif f.type in ("Restriction", Restriction):
            args[f.name] = Restriction(**args[f.name])
        elif isinstance(args[f.name], list):
            args[f.name] = tuple(tuple(v) if isinstance(v, list) else v for v in args[f.name])
    return cls(**args)


def apply_edits(model: Model, edits) -> Model:
    for edit in edits:
        model = edit.apply(model)
    return model


# Parameters -------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SetPrior(Edit):
    """Fix a parameter, give it a prior, or give it no prior ("none"). Its shape and domain are kept."""

    op: ClassVar[str] = "set_prior"
    target: str
    fixed: Any = None
    prior: str | None = None

    def apply(self, model):
        old = model.parameter(self.target)
        return model.with_parameter(self.target, replace(old, fixed=self.fixed, prior=self.prior))


@dataclass(frozen=True)
class AddParameter(Edit):
    op: ClassVar[str] = "add_parameter"
    target: str
    parameter: Parameter

    def apply(self, model):
        try:
            model.parameter(self.target)
        except KeyError:
            return model.with_parameter(self.target, self.parameter)
        raise EditError(f"add_parameter: {self.target!r} already exists")


@dataclass(frozen=True)
class RemoveParameter(Edit):
    op: ClassVar[str] = "remove_parameter"
    target: str

    def apply(self, model):
        model.parameter(self.target)
        return model.with_parameter(self.target, None)


@dataclass(frozen=True)
class Untie(Edit):
    """Give one instance of a template block a parameter of its own, instead of the one all instances share."""

    op: ClassVar[str] = "untie"
    target: str
    element: str
    fixed: Any = None
    prior: str | None = None

    def apply(self, model):
        old = model.parameter(self.target)
        _require_template_element(model, self.target, self.element)
        own = replace(old, fixed=self.fixed, prior=self.prior, separate={})
        return model.with_parameter(self.target, replace(old, separate={**old.separate, self.element: own}))


@dataclass(frozen=True)
class Bind(Edit):
    """Make an instance share the template's parameter again."""

    op: ClassVar[str] = "bind"
    target: str
    element: str

    def apply(self, model):
        old = model.parameter(self.target)
        if self.element not in old.separate:
            raise EditError(f"bind: {self.target!r} has no separate value for {self.element!r}")
        return model.with_parameter(self.target, replace(old, separate={k: v for k, v in old.separate.items() if k != self.element}))


def _require_template_element(model, target, element):
    owner, _ = split_target(target)
    block = model.blocks.get(owner)
    if block is None or block.over is None:
        raise EditError(f"{target!r} does not belong to a template block")
    if element not in map(str, model.sets[block.over[1]]):
        raise EditError(f"{element!r} is not an element of {block.over[1]!r}")


# Shocks, expectations, numbers ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SetShock(Edit):
    op: ClassVar[str] = "set_shock"
    target: str            # block.shock
    distribution: str

    def apply(self, model):
        owner, name = split_target(self.target)
        block = _block(model, owner)
        if name not in block.shocks:
            raise EditError(f"set_shock: block {owner!r} has no shock {name!r}")
        shocks = {**block.shocks, name: replace(block.shocks[name], distribution=self.distribution)}
        return _with_block(model, owner, replace(block, shocks=shocks))


@dataclass(frozen=True)
class SetExpectations(Edit):
    op: ClassVar[str] = "set_expectations"
    scheme: str

    def apply(self, model):
        return replace(model, expectations=self.scheme)


@dataclass(frozen=True)
class NameNumber(Edit):
    """Replace a number written in equations by a new named parameter, fixed at that number.

    `sites` lists (equation index, character position) pairs inside block `block`, as found by `mutatis.spec.numbers`.
    """

    op: ClassVar[str] = "name_number"
    block: str
    sites: tuple[tuple[int, int], ...]
    name: str

    def apply(self, model):
        block = _block(model, self.block)
        if self.name in block.parameters or self.name in model.parameters:
            raise EditError(f"name_number: {self.name!r} is already a parameter")
        equations = list(block.equations)
        value = None
        for index, start in sorted(self.sites, key=lambda s: (s[0], -s[1])):  # right to left keeps earlier positions valid
            number = next((n for n in numbers_in(parse_expression(equations[index])) if n.span[0] == start), None)
            if number is None:
                raise EditError(f"name_number: no nameable number starts at position {start} of equation {index} in {self.block!r}")
            if value is not None and number.value != value:
                raise EditError(f"name_number: the sites hold different numbers ({value} and {number.value})")
            value = number.value
            text = equations[index]
            equations[index] = text[:number.span[0]] + self.name + text[number.span[1]:]
        parameters = {**block.parameters, self.name: Parameter(fixed=value)}
        return _with_block(model, self.block, replace(block, equations=tuple(equations), parameters=parameters))


# Equations and blocks ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplaceEquation(Edit):
    op: ClassVar[str] = "replace_equation"
    block: str
    index: int
    equation: str

    def apply(self, model):
        block = _block(model, self.block)
        if not 0 <= self.index < len(block.equations):
            raise EditError(f"replace_equation: block {self.block!r} has no equation {self.index}")
        equations = tuple(self.equation if i == self.index else e for i, e in enumerate(block.equations))
        return _with_block(model, self.block, replace(block, equations=equations))


@dataclass(frozen=True)
class AddBlock(Edit):
    op: ClassVar[str] = "add_block"
    name: str
    block: Block

    def apply(self, model):
        if self.name in model.blocks:
            raise EditError(f"add_block: {self.name!r} already exists")
        return replace(model, blocks={**model.blocks, self.name: self.block})


@dataclass(frozen=True)
class RemoveBlock(Edit):
    op: ClassVar[str] = "remove_block"
    name: str

    def apply(self, model):
        _block(model, self.name)
        return replace(model, blocks={k: v for k, v in model.blocks.items() if k != self.name})


@dataclass(frozen=True)
class ReplaceBlock(Edit):
    """Replace a block, or one instance of a template block ("name[element]"), by one with the same inputs and outputs."""

    op: ClassVar[str] = "replace_block"
    target: str
    block: Block

    def apply(self, model):
        name, element = _instance(self.target)
        old = _block(model, name)
        if element is None:
            _same_ports(self.target, old, self.block, {})
            return _with_block(model, name, replace(self.block, over=old.over))
        _require_template_element(model, f"{name}.x", element)
        _same_ports(self.target, old, self.block, {old.over[0]: element})
        return _with_block(model, name, replace(old, overrides={**old.overrides, element: self.block}))


@dataclass(frozen=True)
class RestoreInstance(Edit):
    """Undo replace_block on one instance of a template block, so it follows the template again."""

    op: ClassVar[str] = "restore_instance"
    target: str

    def apply(self, model):
        name, element = _instance(self.target)
        old = _block(model, name)
        if element not in old.overrides:
            raise EditError(f"restore_instance: {self.target!r} is not replaced")
        return _with_block(model, name, replace(old, overrides={k: v for k, v in old.overrides.items() if k != element}))


def _instance(target: str) -> tuple[str, str | None]:
    name, items = port(target)
    if len(items) > 1:
        raise EditError(f"{target!r}: a block instance takes one element")
    return name, (items[0] if items else None)


def _same_ports(target, old: Block, new: Block, substitute: dict[str, str]):
    def resolved(ports):
        out = set()
        for reference in ports:
            name, items = port(reference)
            out.add((name, tuple(substitute.get(i, i) for i in items)))
        return out

    for side in ("inputs", "outputs"):
        if resolved(getattr(old, side)) != resolved(getattr(new, side)):
            raise EditError(f"replace_block: {target!r} must keep the same {side}: {sorted(getattr(old, side))} "
                            f"(with {substitute or 'no substitution'}) versus {sorted(getattr(new, side))}")


# Sets ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AddSet(Edit):
    op: ClassVar[str] = "add_set"
    set: str
    elements: tuple = ()

    def apply(self, model):
        if self.set in model.sets:
            raise EditError(f"add_set: {self.set!r} already exists")
        return replace(model, sets={**model.sets, self.set: tuple(self.elements)})


@dataclass(frozen=True)
class RemoveSet(Edit):
    op: ClassVar[str] = "remove_set"
    set: str

    def apply(self, model):
        if self.set not in model.sets:
            raise EditError(f"remove_set: no set {self.set!r}")
        return replace(model, sets={k: v for k, v in model.sets.items() if k != self.set})


@dataclass(frozen=True)
class ExtendSet(Edit):
    """Add elements to a set: every template block over it gains instances, and every sum over it gains terms."""

    op: ClassVar[str] = "extend_set"
    set: str
    add: tuple = ()

    def apply(self, model):
        old = _set(model, self.set)
        clash = set(self.add) & set(old)
        if clash:
            raise EditError(f"extend_set: {sorted(map(str, clash))} already in {self.set!r}")
        return replace(model, sets={**model.sets, self.set: old + tuple(self.add)})


@dataclass(frozen=True)
class ShrinkSet(Edit):
    op: ClassVar[str] = "shrink_set"
    set: str
    remove: tuple = ()

    def apply(self, model):
        old = _set(model, self.set)
        missing = set(self.remove) - set(old)
        if missing:
            raise EditError(f"shrink_set: {sorted(map(str, missing))} not in {self.set!r}")
        return replace(model, sets={**model.sets, self.set: tuple(e for e in old if e not in self.remove)})


# Data and restrictions --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AddData(Edit):
    op: ClassVar[str] = "add_data"
    series: str
    link: DataLink

    def apply(self, model):
        if self.series in model.data:
            raise EditError(f"add_data: {self.series!r} already exists")
        return replace(model, data={**model.data, self.series: self.link})


@dataclass(frozen=True)
class RemoveData(Edit):
    op: ClassVar[str] = "remove_data"
    series: str

    def apply(self, model):
        if self.series not in model.data:
            raise EditError(f"remove_data: no series {self.series!r}")
        return replace(model, data={k: v for k, v in model.data.items() if k != self.series})


@dataclass(frozen=True)
class AddRestriction(Edit):
    op: ClassVar[str] = "add_restriction"
    name: str
    restriction: Restriction

    def apply(self, model):
        if self.name in model.restrictions:
            raise EditError(f"add_restriction: {self.name!r} already exists")
        return replace(model, restrictions={**model.restrictions, self.name: self.restriction})


@dataclass(frozen=True)
class RemoveRestriction(Edit):
    op: ClassVar[str] = "remove_restriction"
    name: str

    def apply(self, model):
        if self.name not in model.restrictions:
            raise EditError(f"remove_restriction: no restriction {self.name!r}")
        return replace(model, restrictions={k: v for k, v in model.restrictions.items() if k != self.name})


# Helpers --------------------------------------------------------------------------------------------------------------------

def _block(model: Model, name: str) -> Block:
    if name not in model.blocks:
        raise EditError(f"no block {name!r}")
    return model.blocks[name]


def _with_block(model: Model, name: str, block: Block) -> Model:
    return replace(model, blocks={**model.blocks, name: block})


def _set(model: Model, name: str) -> tuple:
    if name not in model.sets:
        raise EditError(f"no set {name!r}")
    return model.sets[name]


EDITS = dict(_REGISTRY)  # every edit, by name
