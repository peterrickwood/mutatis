"""Data: dated series, and arranging them for one compiled model.

A `Dataset` is a set of named series on common quarterly dates, with NaN for missing values. `arrange` lines it up with a
compiled system: the observed series in the system's order, the outside variables (from data series or calendar definitions),
and, when possible, the starting state read straight from the first periods of data (`DataStart`).

Starting from the data works when every part of the state is either a variable observed exactly, an outside variable, or a lag
of one of these. Then the first `presample` periods give the state, and the model explains the periods after them. This is the
usual way VARs and equation-by-equation models are estimated ("conditional likelihood").
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mutatis.compile.symbols import OUTSIDE, VARIABLE, Element, Ref
from mutatis.compile.system import System
from mutatis.spec.expr import Binary, Call, Compare, Date, Expr, Index, Name, Num, Shift, Unary, parse_expression


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    dates: tuple[str, ...]
    series: dict[str, np.ndarray]

    def __post_init__(self):
        for name, values in self.series.items():
            if len(values) != len(self.dates):
                raise DataError(f"series {name!r} has {len(values)} values for {len(self.dates)} dates")

    @classmethod
    def read_csv(cls, path: str | Path) -> "Dataset":
        with open(path, newline="") as handle:
            rows = list(csv.reader(handle))
        header, body = rows[0], rows[1:]
        if header[0] != "date":
            raise DataError("the first column of a data file must be 'date'")
        values = np.array([[float(v) if v.strip() not in ("", "NA", "nan", "NaN") else np.nan for v in row[1:]] for row in body])
        return cls(tuple(row[0] for row in body), {name: values[:, i] for i, name in enumerate(header[1:])})

    @classmethod
    def read_excel(cls, path: "str | Path | io.BytesIO", sheet: str, date_column: str) -> "Dataset":
        """A sheet with one row per quarter: a date column (dates or text such as 1984Q1), then one column per series."""
        import openpyxl

        rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet].iter_rows(values_only=True))
        header = [str(h) for h in rows[0]]
        if date_column not in header:
            raise DataError(f"no column {date_column!r} in sheet {sheet!r}; columns: {header}")
        d = header.index(date_column)
        body = [r for r in rows[1:] if r[d] is not None]
        dates = tuple(_as_quarter(r[d]) for r in body)
        series = {name: np.array([_number(r[i], name) for r in body]) for i, name in enumerate(header) if i != d}
        return cls(dates, series)

    def between(self, first: str | None = None, last: str | None = None) -> "Dataset":
        """The periods from `first` to `last`, inclusive."""
        start = self.dates.index(first) if first else 0
        stop = self.dates.index(last) + 1 if last else len(self.dates)
        return Dataset(self.dates[start:stop], {k: v[start:stop] for k, v in self.series.items()})

    def write_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", *self.series])
            for i, date in enumerate(self.dates):
                writer.writerow([date, *(repr(float(v[i])) for v in self.series.values())])


def quarters(first: str, n: int) -> tuple[str, ...]:
    year, quarter = _quarter(first)
    out = []
    for _ in range(n):
        out.append(f"{year}Q{quarter}")
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    return tuple(out)


_MISSING = {"", "NA", "N/A", "NAN", "NULL"}


def _number(value, column: str) -> float:
    """A spreadsheet cell as a number; empty cells, NA, and Excel error values such as #N/A are missing."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if text.upper() in _MISSING or text.startswith("#"):
            return np.nan
        try:
            return float(text)
        except ValueError:
            raise DataError(f"column {column!r} holds text that is not a number: {value!r}") from None
    return float(value)


def _as_quarter(value) -> str:
    """A spreadsheet date (a date, or text such as 1984Q1) as the quarter it falls in."""
    if hasattr(value, "month"):
        return f"{value.year}Q{(value.month - 1) // 3 + 1}"
    _quarter(str(value))
    return str(value)


def _quarter(text: str) -> tuple[int, int]:
    try:
        year, quarter = text.split("Q")
        return int(year), int(quarter)
    except ValueError:
        raise DataError(f"dates must look like 1984Q1, not {text!r}") from None


@dataclass(frozen=True)
class DataStart:
    state: np.ndarray       # the state in the last presample period
    first: int              # index of the first period the model explains


@dataclass(frozen=True)
class Arranged:
    dates: tuple[str, ...]
    observed: np.ndarray    # (periods, observed series), NaN where missing
    outside: np.ndarray     # (periods, outside variables)
    auxiliary: dict[str, np.ndarray]
    start: DataStart | None


def transform_series(text: str | None, x: np.ndarray) -> np.ndarray:
    """Apply a transform written as an expression in the raw series `x`, such as "100 * log(x)" or "d(log(x))".

    Allowed: numbers, x, x(-k) (the series k periods earlier, NaN before the start), d(...) for the change from the previous
    period, + - * / ^, and log, exp, sqrt, abs.
    """
    if text is None:
        return x
    return np.asarray(_transform(parse_expression(text), np.asarray(x, dtype=float)), dtype=float) * np.ones(len(x))


def _lagged(values, k: int):
    out = np.full(len(values), np.nan)
    if k < len(values):
        out[k:] = values[:len(values) - k]
    return out


def _transform(expr: Expr, x: np.ndarray):
    match expr:
        case Num(value=value):
            return float(value)
        case Name(id="x"):
            return x
        case Shift(base=base, lag=lag):
            k = int(_transform(lag, x))
            if k > 0:
                raise DataError("a transform cannot look into the future")
            return _lagged(_transform(base, x), -k)
        case Unary(op=op, operand=operand):
            value = _transform(operand, x)
            return -value if op == "-" else value
        case Binary(op=op, left=left, right=right):
            a, b = _transform(left, x), _transform(right, x)
            return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}[op](a, b)
        case Call(func="d", args=(inner,)):
            value = _transform(inner, x)
            return value - _lagged(value, 1)
        case Call(func=func, args=(inner,)) if func in ("log", "exp", "sqrt", "abs"):
            return getattr(np, func)(_transform(inner, x))
    raise DataError("a transform may use only x, x(-k), numbers, d(...), + - * / ^, log, exp, sqrt and abs")


def arrange(system: System, dataset: Dataset, links: dict) -> Arranged:
    """Line the data up with the system. `links` is the specification's data links (series name -> DataLink), for transforms."""
    def series(name):
        if name not in dataset.series:
            raise DataError(f"the model needs the data series {name!r}, which the data does not have")
        transform = links[name].transform if name in links else None
        return transform_series(transform, np.asarray(dataset.series[name], dtype=float))

    periods = len(dataset.dates)
    observed = np.stack([series(o.series) for o in system.observations], axis=1) if system.observations else np.zeros((periods, 0))
    columns = []
    for info in system.outside.values():
        if info.calendar is not None:
            columns.append(calendar_values(info.calendar, dataset.dates)[:, None])
        else:
            labels = _labels(system, info.key, system.layout.outside)
            missing = [label for label in labels if label not in info.series]
            if missing:
                raise DataError(f"no data series feeds {missing}")
            columns.append(np.stack([series(info.series[label]) for label in labels], axis=1))
    outside = np.concatenate(columns, axis=1) if columns else np.zeros((periods, 0))
    auxiliary = {name: series(name) for name in system.auxiliary_series}
    return Arranged(dataset.dates, observed, outside, auxiliary, data_start(system, observed, outside))


def _labels(system: System, key: str, slots) -> list[str]:
    positions = slots[key].positions.reshape(-1)
    labels = system.outside_labels if slots is system.layout.outside else system.variable_labels
    return [labels[i] for i in positions]


def state_sources(system: System) -> list[tuple[str, int, int]] | None:
    """Where each part of the state can be read from the data: (kind, column, lag), with kind 'observed' or 'outside'.

    None when some part cannot be: a variable not observed exactly, or a copy of a shock.
    """
    exact_identity = {}
    for column, observation in enumerate(system.observations):
        label = _identity_label(observation.expr, system)
        if observation.error is None and label is not None:
            exact_identity[label] = column
    flat = {}
    for key, slot in system.layout.variables.items():
        for position, elements in zip(slot.positions.reshape(-1), _element_combinations(slot.axes, system.sets)):
            flat[int(position)] = (key, elements)
    sources = []
    for position in range(system.n_variables):
        source = _source(*flat[position], system, exact_identity, lag=0)
        if source is None:
            return None
        sources.append(source)
    return sources


def _source(key, elements, system, exact_identity, lag):
    if key in system.auxiliary:
        aux = system.auxiliary[key]
        if "@lag" in key:
            depth = int(key.rsplit("@lag", 1)[1])
            return _source(aux.base, elements, system, exact_identity, lag + depth)
        if aux.base_kind == OUTSIDE:
            outside_positions = system.layout.outside[aux.base].positions
            index = outside_positions[tuple(_position(system, a, e) for a, e in zip(system.layout.outside[aux.base].axes, elements))]
            return ("outside", int(index), lag)
        return None  # a copy of a shock cannot be read from data
    label = f"{key}[{', '.join(map(str, elements))}]" if elements else key
    return ("observed", exact_identity[label], lag) if label in exact_identity else None


def _position(system, axis, element):
    return system.layout.element_position(axis, element)


def _element_combinations(axes, sets):
    combos = [()]
    for axis in axes:
        combos = [c + (e,) for c in combos for e in sets[axis]]
    return combos


def _identity_label(expr: Expr, system: System) -> str | None:
    """If an observation is just one element of one variable, that element's label."""
    if isinstance(expr, Ref) and expr.kind == VARIABLE and not system.layout.variables[expr.key].axes:
        return expr.key
    if isinstance(expr, Index) and isinstance(expr.base, Ref) and expr.base.kind == VARIABLE and all(isinstance(i, Element) for i in expr.items):
        axes = system.layout.variables[expr.base.key].axes
        if len(axes) == len(expr.items):
            return f"{expr.base.key}[{', '.join(str(i.value) for i in expr.items)}]"
    return None


def data_start(system: System, observed: np.ndarray, outside: np.ndarray) -> DataStart | None:
    sources = state_sources(system)
    if sources is None:
        return None
    p = system.presample
    if observed.shape[0] <= p:
        raise DataError(f"the data has {observed.shape[0]} periods; at least {p + 1} are needed")
    state = np.array([(observed if kind == "observed" else outside)[p - 1 - lag, column] for kind, column, lag in sources])
    if np.any(np.isnan(state)):
        return None
    return DataStart(state, p)


def calendar_values(text: str, dates: tuple[str, ...]) -> np.ndarray:
    """Evaluate a calendar definition such as 'date < 2008Q2' on each date: 1 where a comparison holds, 0 where it does not."""
    expr = parse_expression(text)
    positions = np.array([year + (quarter - 1) / 4 for year, quarter in map(_quarter, dates)])
    return np.asarray(_calendar(expr, positions), dtype=float) * np.ones(len(dates))


def _calendar(expr: Expr, positions):
    match expr:
        case Name(id="date"):
            return positions
        case Date(year=year, quarter=quarter):
            return year + (quarter - 1) / 4
        case Num(value=value):
            return float(value)
        case Unary(op=op, operand=operand):
            return -_calendar(operand, positions) if op == "-" else _calendar(operand, positions)
        case Compare(op=op, left=left, right=right):
            a, b = _calendar(left, positions), _calendar(right, positions)
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op].astype(float)
        case Binary(op=op, left=left, right=right) if op in "+-*":
            a, b = _calendar(left, positions), _calendar(right, positions)
            return a + b if op == "+" else a - b if op == "-" else a * b
    raise DataError(f"a calendar definition may use only 'date', dates such as 2008Q2, numbers, comparisons and + - *")
