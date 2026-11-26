"""Checks on a specification that need no numbers: does everything parse, does every name refer to something, does each
variable come from exactly one place. Counting equations against variables needs the shapes of vector variables, so it is
done when the model is compiled.

Each check returns findings rather than raising, so that a tool can show every problem at once.
"""

from dataclasses import dataclass

from mutatis.spec.expr import Call, Compare, Expr, Index, Name, Over, ParseError, Range, Shift, parse_expression
from mutatis.spec.model import Block, Model, port
from mutatis.spec.vocabulary import DISTRIBUTIONS, DOMAINS, EXPECTATION_SCHEMES, NO_PRIOR


@dataclass(frozen=True)
class Finding:
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def check(model: Model) -> list[Finding]:
    findings: list[Finding] = []
    elements = {str(x) for values in model.sets.values() for x in values}
    clash = elements & set(model.sets)
    if clash:
        findings.append(Finding("sets", f"names used both as a set and as an element: {sorted(clash)}"))
    findings += _expectations(model)
    for name, text in model.calendar.items():
        findings += _parses(f"calendar.{name}", text)
    findings += _parameters("parameters", model.parameters, model)
    produced, findings_produced = _produced(model)
    findings += findings_produced
    fed = {variable for series, link in model.data.items() if link.feeds for variable in _expand(link.feeds, model, {})}
    for name, block in model.blocks.items():
        findings += _block(name, block, model, produced, fed)
    findings += _data(model, produced, fed)
    for name, restriction in model.restrictions.items():
        findings += _parses(f"restrictions.{name}.quantity", restriction.quantity)
        findings += _parses(f"restrictions.{name}.require", "_ " + restriction.require)
    return findings


def _parses(where: str, text: str, shifts: bool = True) -> list[Finding]:
    try:
        parse_expression(text, shifts=shifts)
    except ParseError as error:
        return [Finding(where, str(error))]
    return []


def _expectations(model: Model) -> list[Finding]:
    try:
        expr = parse_expression(model.expectations, shifts=False)
    except ParseError as error:
        return [Finding("expectations", str(error))]
    name = expr.id if isinstance(expr, Name) else expr.func if isinstance(expr, Call) else None
    if name not in EXPECTATION_SCHEMES:
        return [Finding("expectations", f"unknown scheme {model.expectations!r}; known: {sorted(EXPECTATION_SCHEMES)}")]
    return []


def _distribution(where: str, text: str) -> list[Finding]:
    try:
        expr = parse_expression(text, shifts=False)
    except ParseError as error:
        return [Finding(where, str(error))]
    name = expr.id if isinstance(expr, Name) else expr.func if isinstance(expr, Call) else None
    if name not in DISTRIBUTIONS and name != NO_PRIOR:
        return [Finding(where, f"unknown distribution {text!r}; known: {sorted(DISTRIBUTIONS)}")]
    return []


def _parameters(where: str, parameters: dict, model: Model) -> list[Finding]:
    findings = []
    for name, parameter in parameters.items():
        if parameter.prior is not None:
            findings += _distribution(f"{where}.{name}.prior", parameter.prior)
        if parameter.domain is not None and parameter.domain not in DOMAINS:
            findings.append(Finding(f"{where}.{name}", f"unknown domain {parameter.domain!r}; known: {sorted(DOMAINS)}"))
        for set_name in parameter.shape:
            if set_name not in model.sets:
                findings.append(Finding(f"{where}.{name}", f"shape refers to unknown set {set_name!r}"))
    return findings


def _expand(reference: str, model: Model, substitute: dict[str, str]) -> set[tuple[str, tuple[str, ...]]]:
    """A port as the set of (variable, element names) it covers: 'y[series]' covers every element of series."""
    name, items = port(reference)
    combinations: list[tuple[str, ...]] = [()]
    for item in items:
        item = substitute.get(item, item)
        options = [str(x) for x in model.sets[item]] if item in model.sets else [item]
        combinations = [c + (o,) for c in combinations for o in options]
    return {(name, c) for c in combinations}


def instances(name: str, block: Block, model: Model):
    """The block's instances as (label, block, substitution). A plain block has one; a template has one per element of its set."""
    if block.over is None:
        yield name, block, {}
        return
    index, set_name = block.over
    for element in model.sets.get(set_name, ()):
        element = str(element)
        yield f"{name}[{element}]", block.overrides.get(element, block), {index: element}


def _produced(model: Model):
    produced: dict[tuple[str, tuple[str, ...]], str] = {}
    findings = []
    for name, block in model.blocks.items():
        if block.over is not None and block.over[1] not in model.sets:
            findings.append(Finding(f"blocks.{name}", f"template over unknown set {block.over[1]!r}"))
            continue
        for label, instance, substitute in instances(name, block, model):
            for output in instance.outputs:
                try:
                    covered = _expand(output, model, substitute)
                except ValueError as error:
                    findings.append(Finding(f"blocks.{label}.outputs", str(error)))
                    continue
                for variable in covered:
                    if variable in produced:
                        findings.append(Finding(f"blocks.{label}", f"{_show(variable)} is already an output of {produced[variable]}"))
                    produced[variable] = label
    return produced, findings


def _show(variable) -> str:
    name, items = variable
    return f"{name}[{', '.join(items)}]" if items else name


def _block(name: str, block: Block, model: Model, produced, fed) -> list[Finding]:
    where = f"blocks.{name}"
    findings = _parameters(f"{where}.parameters", block.parameters, model)
    for name, shock in block.shocks.items():
        findings += _distribution(f"{where}.shocks.{name}", shock.distribution)
    for variable, text in block.initial.items():
        findings += _distribution(f"{where}.initial.{variable}", text)
    if block.over is not None:
        set_elements = {str(x) for x in model.sets.get(block.over[1], ())}
        for element in block.overrides:
            if element not in set_elements:
                findings.append(Finding(f"{where}.overrides", f"{element!r} is not an element of {block.over[1]!r}"))
        for parameter_name, parameter in block.parameters.items():
            for element in parameter.separate:
                if element not in set_elements:
                    findings.append(Finding(f"{where}.parameters.{parameter_name}.separate", f"{element!r} is not an element of {block.over[1]!r}"))
    for instance_name, instance, substitute in instances(name, block, model):
        label = f"blocks.{instance_name}"
        for reference in instance.inputs:
            try:
                needed = _expand(reference, model, substitute)
            except ValueError as error:
                findings.append(Finding(f"{label}.inputs", str(error)))
                continue
            for variable in needed:
                if variable not in produced and variable not in fed and variable[0] not in model.calendar:
                    findings.append(Finding(f"{label}.inputs", f"{_show(variable)} is not an output of any block, fed by any data series, or a calendar variable"))
        findings += _names(label, instance, block, model)
    return findings


def _names(where: str, instance: Block, template: Block, model: Model) -> list[Finding]:
    """Every name in the equations and definitions refers to a port, parameter, shock, definition, set, element or calendar variable."""
    allowed = {port(r)[0] for r in (*instance.inputs, *instance.outputs)}
    allowed |= set(instance.parameters) | set(template.parameters) | set(instance.shocks) | set(instance.definitions) | set(model.parameters)
    allowed |= set(model.sets) | {str(x) for values in model.sets.values() for x in values} | set(model.calendar)
    if template.over is not None:
        allowed.add(template.over[0])
    findings = []
    texts = [(f"{where}.equations[{i}]", text, True) for i, text in enumerate(instance.equations)]
    texts += [(f"{where}.definitions.{name}", text, False) for name, text in instance.definitions.items()]
    for place, text, is_equation in texts:
        try:
            expr = parse_expression(text)
        except ParseError as error:
            findings.append(Finding(place, str(error)))
            continue
        if is_equation and not (isinstance(expr, Compare) and expr.op == "="):
            findings.append(Finding(place, "an equation needs exactly one '=' between its two sides"))
        for name in sorted(_free_names(expr) - allowed):
            findings.append(Finding(place, f"{name!r} is not an input, output, parameter, shock or definition of this block, nor a set, element or calendar variable"))
    return findings


def _free_names(expr: Expr, bound: frozenset = frozenset()) -> set[str]:
    match expr:
        case Name(id=name):
            return set() if name in bound else {name}
        case Over(index=index, body=body):
            return _free_names(body, bound | {index})
        case Range():
            return set()
        case Shift(base=base, lag=lag):
            return _free_names(base, bound) | _free_names(lag, bound)
        case Index(base=base, items=items):
            return _free_names(base, bound).union(*(_free_names(i, bound) for i in items))
        case Call(args=args, kwargs=kwargs):
            return set().union(*(_free_names(a, bound) for a in args), *(_free_names(v, bound) for _, v in kwargs))
    children = [getattr(expr, f) for f in ("operand", "left", "right") if hasattr(expr, f)]
    return set().union(*(_free_names(c, bound) for c in children))


def _data(model: Model, produced, fed) -> list[Finding]:
    findings = []
    variables = {name for name, _ in produced} | {name for name, _ in fed}
    for series, link in model.data.items():
        where = f"data.{series}"
        findings += _parameters(f"{where}.parameters", link.parameters, model)
        if link.error is not None:
            findings += _distribution(f"{where}.error", link.error)
        if link.feeds is not None:
            for variable in _expand(link.feeds, model, {}):
                if variable in produced:
                    findings.append(Finding(where, f"feeds {_show(variable)}, which is already an output of {produced[variable]}"))
        if link.observes is not None:
            try:
                expr = parse_expression(link.observes)
            except ParseError as error:
                findings.append(Finding(f"{where}.observes", str(error)))
                continue
            allowed = variables | set(link.parameters) | set(model.parameters) | set(model.sets) | {str(x) for v in model.sets.values() for x in v}
            for name in sorted(_free_names(expr) - allowed):
                findings.append(Finding(f"{where}.observes", f"{name!r} is not a model variable"))
    return findings
