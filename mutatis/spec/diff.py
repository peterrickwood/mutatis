"""The difference between two versions of a model, as a list of edits.

`diff(a, b)` returns edits such that applying them to `a` gives `b`. For a single edit, the difference is usually that same
edit; where it is not (naming a number, for example, shows up as an added parameter and a replaced equation), it is a list of
edits with the same effect.
"""

from dataclasses import replace

from mutatis.spec import edits as e
from mutatis.spec.model import Block, Model, Parameter


def diff(a: Model, b: Model) -> list[e.Edit]:
    if a.name != b.name or a.time != b.time or a.calendar != b.calendar:
        raise ValueError("diff compares versions of one model: name, time and calendar must match")
    out: list[e.Edit] = []
    if a.expectations != b.expectations:
        out.append(e.SetExpectations(b.expectations))
    out += _sets(a.sets, b.sets)
    out += _parameters(None, a.parameters, b.parameters)
    for name in a.blocks.keys() - b.blocks.keys():
        out.append(e.RemoveBlock(name))
    for name, block in b.blocks.items():
        if name not in a.blocks:
            out.append(e.AddBlock(name, block))
        elif a.blocks[name] != block:
            out += _block(name, a.blocks[name], block)
    for series in a.data.keys() - b.data.keys():
        out.append(e.RemoveData(series))
    for series, link in b.data.items():
        if series not in a.data:
            out.append(e.AddData(series, link))
        elif a.data[series] != link:
            old = a.data[series]
            if replace(old, parameters={}) == replace(link, parameters={}):
                out += _parameters(series, old.parameters, link.parameters)
            else:
                out += [e.RemoveData(series), e.AddData(series, link)]
    for name in a.restrictions.keys() - b.restrictions.keys():
        out.append(e.RemoveRestriction(name))
    for name, restriction in b.restrictions.items():
        if a.restrictions.get(name) != restriction:
            if name in a.restrictions:
                out.append(e.RemoveRestriction(name))
            out.append(e.AddRestriction(name, restriction))
    return out


def _sets(a: dict, b: dict) -> list[e.Edit]:
    out: list[e.Edit] = []
    for name in a.keys() - b.keys():
        out.append(e.RemoveSet(name))
    for name, elements in b.items():
        if name not in a:
            out.append(e.AddSet(name, elements))
        elif a[name] != elements:
            removed = tuple(x for x in a[name] if x not in elements)
            kept = tuple(x for x in a[name] if x in elements)
            added = tuple(x for x in elements if x not in a[name])
            if kept + added == elements:
                out += ([e.ShrinkSet(name, removed)] if removed else []) + ([e.ExtendSet(name, added)] if added else [])
            else:  # the order changed: replace the whole set
                out += [e.RemoveSet(name), e.AddSet(name, elements)]
    return out


def _target(owner: str | None, name: str) -> str:
    return f"{owner}.{name}" if owner else name


def _parameters(owner: str | None, a: dict[str, Parameter], b: dict[str, Parameter]) -> list[e.Edit]:
    out: list[e.Edit] = []
    for name in a.keys() - b.keys():
        out.append(e.RemoveParameter(_target(owner, name)))
    for name, new in b.items():
        target = _target(owner, name)
        if name not in a:
            out.append(e.AddParameter(target, new))
            continue
        old = a[name]
        if old == new:
            continue
        if (old.shape, old.domain) != (new.shape, new.domain):
            out += [e.RemoveParameter(target), e.AddParameter(target, new)]
            continue
        if (old.fixed, old.prior) != (new.fixed, new.prior):
            out.append(e.SetPrior(target, fixed=new.fixed, prior=new.prior))
        for element in old.separate.keys() - new.separate.keys():
            out.append(e.Bind(target, element))
        for element, own in new.separate.items():
            if old.separate.get(element) != own:
                out.append(e.Untie(target, element, fixed=own.fixed, prior=own.prior))
    return out


def _block(name: str, old: Block, new: Block) -> list[e.Edit]:
    structure = ("inputs", "outputs", "definitions", "initial", "code", "over")
    same_structure = (all(getattr(old, f) == getattr(new, f) for f in structure)
                      and {k: s.shape for k, s in old.shocks.items()} == {k: s.shape for k, s in new.shocks.items()}
                      and len(old.equations) == len(new.equations))
    if old.over != new.over or (old.inputs, old.outputs) != (new.inputs, new.outputs):
        return [e.RemoveBlock(name), e.AddBlock(name, new)]
    if not same_structure:
        return [e.ReplaceBlock(name, replace(new, overrides={}, over=None))] + _overrides(name, replace(old, overrides={}), new)
    out = _parameters(name, old.parameters, new.parameters)
    for shock, value in new.shocks.items():
        if old.shocks[shock] != value:
            out.append(e.SetShock(f"{name}.{shock}", value.distribution))
    for index, (before, after) in enumerate(zip(old.equations, new.equations)):
        if before != after:
            out.append(e.ReplaceEquation(name, index, after))
    return out + _overrides(name, old, new)


def _overrides(name: str, old: Block, new: Block) -> list[e.Edit]:
    out: list[e.Edit] = []
    for element in old.overrides.keys() - new.overrides.keys():
        out.append(e.RestoreInstance(f"{name}[{element}]"))
    for element, block in new.overrides.items():
        if old.overrides.get(element) != block:
            out.append(e.ReplaceBlock(f"{name}[{element}]", block))
    return out
