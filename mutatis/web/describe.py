"""A specification as plain data for display: every block with its equations as mathematics, parameters, shocks and starting
values; every data series; the restrictions; and the graph of blocks with its layout. Only the specification is read."""

from mutatis.spec.expr import Name, ParseError, parse_expression, parse_requirement, walk
from mutatis.spec.latex import distribution_latex, symbol, text_to_latex, to_latex
from mutatis.spec.model import Block, Model, Parameter, port
from mutatis.spec.yaml_io import dump_model
from mutatis.web.graph import block_graph

_RELATIONS = {"<": "<", "<=": r"\le", ">": ">", ">=": r"\ge", "==": "=", "!=": r"\neq"}


def describe(model: Model) -> dict:
    variables = {port(r)[0] for b in model.blocks.values() for r in (*b.inputs, *b.outputs)} | set(model.calendar)
    variables |= {port(link.feeds)[0] for link in model.data.values() if link.feeds}
    return {"name": model.name, "time": model.time, "expectations": model.expectations,
            "sets": {name: [str(e) for e in elements] for name, elements in model.sets.items()},
            "calendar": [{"name": name, "text": text, "latex": _latex(lambda: f"{symbol(name)}_{{t}} = {text_to_latex(text)}")}
                         for name, text in model.calendar.items()],
            "parameters": [_parameter(name, name, p) for name, p in model.parameters.items()],
            "blocks": [_block(name, block, model) for name, block in model.blocks.items()],
            "data": [_series(series, link, variables) for series, link in model.data.items()],
            "restrictions": [_restriction(name, r.quantity, r.require) for name, r in model.restrictions.items()],
            "graph": block_graph(model), "yaml": dump_model(model)}


def _latex(make) -> str | None:
    """LaTeX, or None when the text does not parse (the page then shows the text as written)."""
    try:
        return make()
    except (ParseError, TypeError):
        return None


def _parameter(name: str, target: str, parameter: Parameter) -> dict:
    entry = {"name": name, "target": target, "kind": parameter.kind, "shape": list(parameter.shape), "domain": parameter.domain}
    if parameter.fixed is not None:
        entry["fixed"] = _thaw(parameter.fixed)
        if not isinstance(parameter.fixed, tuple):
            entry["latex"] = f"{symbol(name)} = {parameter.fixed}"
    elif parameter.kind == "no_prior":
        entry["prior"] = parameter.prior
        entry["latex"] = f"{symbol(name)}: \\text{{no prior}}"
    else:
        entry["prior"] = parameter.prior
        entry["latex"] = _latex(lambda: f"{symbol(name)} \\sim {distribution_latex(parameter.prior)}")
    if parameter.separate:
        entry["separate"] = [_parameter(f"{name}[{element}]", f"{target}[{element}]", own) for element, own in parameter.separate.items()]
    return entry


def _thaw(value):
    return [_thaw(v) for v in value] if isinstance(value, tuple) else value


def _block(name: str, block: Block, model: Model) -> dict:
    timed = {port(r)[0] for r in (*block.inputs, *block.outputs)} | set(block.shocks) | set(model.calendar)
    for key, text in block.definitions.items():   # a definition that involves a variable is itself one
        try:
            if any(isinstance(n, Name) and n.id in timed for n in walk(parse_expression(text))):
                timed.add(key)
        except ParseError:
            pass
    out = {"name": name, "inputs": list(block.inputs), "outputs": list(block.outputs), "code": block.code,
           "template": {"index": block.over[0], "set": block.over[1], "elements": [str(e) for e in model.sets.get(block.over[1], ())]} if block.over else None,
           "equations": [{"text": text, "latex": _latex(lambda: text_to_latex(text, timed))} for text in block.equations],
           "definitions": [{"name": key, "text": text, "latex": _latex(lambda: f"{to_latex(Name(key), timed)} = {text_to_latex(text, timed)}")}
                           for key, text in block.definitions.items()],
           "parameters": [_parameter(p, f"{name}.{p}", spec) for p, spec in block.parameters.items()],
           "shocks": [{"name": s, "shape": list(shock.shape), "distribution": shock.distribution,
                       "latex": _latex(lambda: f"{symbol(s)}_{{t}} \\sim {distribution_latex(shock.distribution)}")} for s, shock in block.shocks.items()],
           "initial": [{"variable": v, "text": text, "latex": _latex(lambda: f"{symbol(v)}_{{0}} \\sim {distribution_latex(text)}")}
                       for v, text in block.initial.items()],
           "steady_state": [{"variable": v, "text": text, "latex": _latex(lambda: f"\\bar{{{symbol(v)}}} = {text_to_latex(text)}")}
                            for v, text in block.steady_state.items()]}
    out["overrides"] = [{"element": element, **_block(f"{name}[{element}]", own, model)} for element, own in block.overrides.items()]
    return out


def _series(series: str, link, variables: set[str]) -> dict:
    left = to_latex(Name(series), {series})
    entry = {"series": series, "role": link.role, "observes": link.observes, "feeds": link.feeds, "error": link.error, "transform": link.transform,
             "parameters": [_parameter(p, f"{series}.{p}", spec) for p, spec in link.parameters.items()]}
    if link.observes is not None:
        entry["observes_latex"] = _latex(lambda: f"{left} = {text_to_latex(link.observes, variables)}")
        entry["error_latex"] = _latex(lambda: distribution_latex(link.error)) if link.error is not None else None
    if link.feeds is not None:
        entry["feeds_latex"] = _latex(lambda: f"{text_to_latex(link.feeds, variables)} = {left}")
    if link.transform is not None:
        entry["transform_latex"] = _latex(lambda: text_to_latex(link.transform, {"x"}))   # x: the series as published
    return entry


def _restriction(name: str, quantity: str, require: str) -> dict:
    def make():
        op, bound = parse_requirement(require)
        return f"{text_to_latex(quantity)} {_RELATIONS[op]} {to_latex(bound)}"
    return {"name": name, "quantity": quantity, "require": require, "latex": _latex(make)}

