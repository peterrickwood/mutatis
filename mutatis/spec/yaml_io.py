"""Reading and writing specifications as YAML (or any equivalent nested dictionaries, such as JSON).

Unknown keys are errors, not ignored: a typo, or a comma inside an unquoted value in YAML's inline `{...}` form (which YAML
reads as a second key), must not silently change the model.
"""

import re
from dataclasses import fields
from typing import Any

import yaml

from mutatis.spec.model import Block, DataLink, Model, Parameter, Restriction, Shock

_TEMPLATE_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)\s*\]\s*$")
_SHOCK_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\[\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\])?\s*$")
_RANGE = re.compile(r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*$")


class SpecError(ValueError):
    pass


def _check_keys(where: str, given: dict, allowed: set[str]):
    if not isinstance(given, dict):
        raise SpecError(f"{where}: expected a mapping, got {type(given).__name__}")
    unknown = set(given) - allowed
    if unknown:
        hint = " (a comma inside an unquoted value in {...} splits it into two keys: quote the value)" if any(")" in str(k) for k in unknown) else ""
        raise SpecError(f"{where}: unknown key(s) {sorted(map(str, unknown))}; allowed: {sorted(allowed)}{hint}")


def _set_elements(where: str, value) -> tuple:
    if isinstance(value, str):
        match = _RANGE.match(value)
        if not match:
            raise SpecError(f"{where}: a set is a list of names or a range such as 1..4, got {value!r}")
        start, stop = int(match.group(1)), int(match.group(2))
        return tuple(range(start, stop + 1))
    if not isinstance(value, list):
        raise SpecError(f"{where}: a set is a list of names or a range such as 1..4")
    return tuple(value)


def parameter_from_dict(where: str, value: dict) -> Parameter:
    _check_keys(where, value, {"fixed", "prior", "shape", "domain", "separate"})
    separate = {str(element): parameter_from_dict(f"{where}.separate.{element}", v) for element, v in value.get("separate", {}).items()}
    try:
        return Parameter(fixed=value.get("fixed"), prior=_text(value.get("prior")), shape=tuple(value.get("shape", ())),
                         domain=value.get("domain"), separate=separate)
    except ValueError as error:
        raise SpecError(f"{where}: {error}") from None


def _text(value):
    return None if value is None else str(value)


def block_from_dict(where: str, value: dict) -> Block:
    _check_keys(where, value, {"inputs", "outputs", "parameters", "definitions", "shocks", "initial", "equations", "code", "overrides"})
    shocks = {}
    for key, distribution in (value.get("shocks") or {}).items():
        match = _SHOCK_KEY.match(str(key))
        if not match:
            raise SpecError(f"{where}.shocks: a shock is a name, optionally followed by set names in brackets: {key!r}")
        shape = tuple(s.strip() for s in match.group(2).split(",")) if match.group(2) else ()
        shocks[match.group(1)] = Shock(str(distribution), shape)
    overrides = {str(element): block_from_dict(f"{where}.overrides.{element}", v) for element, v in (value.get("overrides") or {}).items()}
    try:
        return Block(inputs=tuple(value.get("inputs") or ()), outputs=tuple(value.get("outputs") or ()),
                     parameters={k: parameter_from_dict(f"{where}.parameters.{k}", v) for k, v in (value.get("parameters") or {}).items()},
                     definitions={k: str(v) for k, v in (value.get("definitions") or {}).items()}, shocks=shocks,
                     initial={k: str(v) for k, v in (value.get("initial") or {}).items()},
                     equations=tuple(str(e) for e in (value.get("equations") or ())), code=value.get("code"), overrides=overrides)
    except ValueError as error:
        raise SpecError(f"{where}: {error}") from None


def data_link_from_dict(where: str, value: dict) -> DataLink:
    _check_keys(where, value or {}, {"observes", "feeds", "error", "parameters", "transform"})
    value = value or {}
    error = value.get("error")
    try:
        return DataLink(observes=_text(value.get("observes")), feeds=_text(value.get("feeds")), error=None if error in (None, "none") else str(error),
                        parameters={k: parameter_from_dict(f"{where}.parameters.{k}", v) for k, v in (value.get("parameters") or {}).items()},
                        transform=value.get("transform"))
    except ValueError as error:
        raise SpecError(f"{where}: {error}") from None


def model_from_dict(value: dict) -> Model:
    _check_keys("model", value, {"model", "time", "sets", "calendar", "expectations", "parameters", "blocks", "data", "restrictions"})
    blocks = {}
    for key, block in (value.get("blocks") or {}).items():
        template = _TEMPLATE_KEY.match(str(key))
        name, over = (template.group(1), (template.group(2), template.group(3))) if template else (str(key), None)
        if name in blocks:
            raise SpecError(f"blocks: {name!r} appears twice")
        built = block_from_dict(f"blocks.{name}", block)
        blocks[name] = Block(**{f.name: getattr(built, f.name) for f in fields(Block) if f.name != "over"}, over=over)
    restrictions = {}
    for name, restriction in (value.get("restrictions") or {}).items():
        _check_keys(f"restrictions.{name}", restriction, {"quantity", "require"})
        restrictions[str(name)] = Restriction(str(restriction["quantity"]), str(restriction["require"]))
    return Model(name=str(value["model"]), time=value.get("time", "quarterly"),
                 sets={k: _set_elements(f"sets.{k}", v) for k, v in (value.get("sets") or {}).items()},
                 calendar={k: str(v) for k, v in (value.get("calendar") or {}).items()}, expectations=str(value.get("expectations", "rational")),
                 parameters={k: parameter_from_dict(f"parameters.{k}", v) for k, v in (value.get("parameters") or {}).items()},
                 blocks=blocks, data={str(k): data_link_from_dict(f"data.{k}", v) for k, v in (value.get("data") or {}).items()},
                 restrictions=restrictions)


def load_model(text: str) -> Model:
    return model_from_dict(yaml.safe_load(text))


# Writing ----------------------------------------------------------------------------------------------------------------------

def _thaw(value):
    return [_thaw(v) for v in value] if isinstance(value, tuple) else value


def parameter_to_dict(parameter: Parameter) -> dict:
    out: dict[str, Any] = {"fixed": _thaw(parameter.fixed)} if parameter.fixed is not None else {"prior": parameter.prior}
    if parameter.shape:
        out["shape"] = list(parameter.shape)
    if parameter.domain:
        out["domain"] = parameter.domain
    if parameter.separate:
        out["separate"] = {k: parameter_to_dict(v) for k, v in parameter.separate.items()}
    return out


def block_to_dict(block: Block) -> dict:
    out: dict[str, Any] = {}
    if block.inputs:
        out["inputs"] = list(block.inputs)
    if block.outputs:
        out["outputs"] = list(block.outputs)
    if block.parameters:
        out["parameters"] = {k: parameter_to_dict(v) for k, v in block.parameters.items()}
    if block.definitions:
        out["definitions"] = dict(block.definitions)
    if block.shocks:
        out["shocks"] = {(f"{k}[{', '.join(s.shape)}]" if s.shape else k): s.distribution for k, s in block.shocks.items()}
    if block.initial:
        out["initial"] = dict(block.initial)
    if block.equations:
        out["equations"] = list(block.equations)
    if block.code:
        out["code"] = block.code
    if block.overrides:
        out["overrides"] = {k: block_to_dict(v) for k, v in block.overrides.items()}
    return out


def data_link_to_dict(link: DataLink) -> dict:
    out = {k: getattr(link, k) for k in ("observes", "feeds", "error", "transform") if getattr(link, k) is not None}
    if link.parameters:
        out["parameters"] = {k: parameter_to_dict(v) for k, v in link.parameters.items()}
    return out


def model_to_dict(model: Model) -> dict:
    out: dict[str, Any] = {"model": model.name, "time": model.time}
    if model.sets:
        out["sets"] = {k: list(v) for k, v in model.sets.items()}
    if model.calendar:
        out["calendar"] = dict(model.calendar)
    out["expectations"] = model.expectations
    if model.parameters:
        out["parameters"] = {k: parameter_to_dict(v) for k, v in model.parameters.items()}
    out["blocks"] = {(f"{k}[{b.over[0]} in {b.over[1]}]" if b.over else k): block_to_dict(b) for k, b in model.blocks.items()}
    if model.data:
        out["data"] = {k: data_link_to_dict(v) for k, v in model.data.items()}
    if model.restrictions:
        out["restrictions"] = {k: {"quantity": r.quantity, "require": r.require} for k, r in model.restrictions.items()}
    return out


def dump_model(model: Model) -> str:
    return yaml.safe_dump(model_to_dict(model), sort_keys=False, width=200, allow_unicode=True)
