"""The benchmark report: one self-contained HTML file, charts drawn as inline SVG, no external resources.

Colours follow the dataviz reference palette: categorical slots 1 to 3 (blue, orange, aqua) for the three main methods, checked
with the palette validator in light and dark mode (all pairs pass; two light-mode hues are below 3:1 contrast, so every chart
has direct labels or distinct marker shapes and every number is also in a table). Pass and fail use the status colours, always
with an icon and a word.
"""

import html
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.samplers.methods import MAIN, METHODS
from benchmarks.samplers.metrics import EVIDENCE_LIMIT, MEAN_ERROR_LIMIT, PEAK_WEIGHT_LIMIT, SD_RATIO_RANGE

SERIES = {"default": 1, "nuts": 2, "smc": 3}
SHAPES = {"default": "circle", "nuts": "square", "smc": "triangle"}
EVIDENCE_OF = {"default": "stepping_stones", "nuts": "bridge", "smc": "smc"}   # the estimate charted for each main method
EVIDENCE_LABEL = {"stepping_stones": "stepping stones", "bridge": "bridge sampling", "smc": "SMC's own"}

CSS = """
:root { color-scheme: light;
  --page: #f9f9f9; --surface-1: #ffffff; --text-primary: #191919; --text-secondary: #666666; --grid: #e7e7e7; --axis: #c2c2c2;
  --border: rgba(11,11,11,0.10); --series-1: #257de0; --series-2: #f47230; --series-3: #00b894;
  --good: #0aac23; --warning: #fbb315; --critical: #dc3c3d; --good-text: #006c10; --critical-text: #b3261e; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --page: #0b0b0b; --surface-1: #191919; --text-primary: #efefef; --text-secondary: #9c9c9c; --grid: #303030; --axis: #474747;
  --border: rgba(255,255,255,0.10); --series-1: #338ced; --series-2: #e76624; --series-3: #00a888; --good-text: #53c556;
  --critical-text: #eb6263; } }
:root[data-theme="dark"] { color-scheme: dark;
  --page: #0b0b0b; --surface-1: #191919; --text-primary: #efefef; --text-secondary: #9c9c9c; --grid: #303030; --axis: #474747;
  --border: rgba(255,255,255,0.10); --series-1: #338ced; --series-2: #e76624; --series-3: #00a888; --good-text: #53c556;
  --critical-text: #eb6263; }
body { background: var(--page); color: var(--text-primary); font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 24px 16px; line-height: 1.45; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 20px; margin: 36px 0 8px; } h3 { font-size: 16px; margin: 24px 0 6px; }
p, li { max-width: 900px; } .muted { color: var(--text-secondary); }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 12px 0; overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: right; vertical-align: top; white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; } td.l, th.l { text-align: left; white-space: normal; }
.pass { color: var(--good-text); font-weight: 600; } .fail { color: var(--critical-text); font-weight: 600; } .part { font-weight: 600; }
.pill { display: inline-block; border-radius: 4px; padding: 0 6px; font-weight: 600; }
.pill.good { background: var(--good); color: #ffffff; } .pill.bad { background: var(--critical); color: #ffffff; }
.pill.mid { background: var(--warning); color: #0b0b0b; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin: 4px 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
svg text { fill: var(--text-secondary); font-size: 11px; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
svg .label { fill: var(--text-primary); font-size: 11px; }
svg .grid { stroke: var(--grid); stroke-width: 1; } svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .limit { stroke: var(--critical); stroke-width: 1; opacity: 0.7; }
.s1 { fill: var(--series-1); stroke: var(--series-1); } .s2 { fill: var(--series-2); stroke: var(--series-2); }
.s3 { fill: var(--series-3); stroke: var(--series-3); }
svg .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg .mark { stroke: var(--surface-1); stroke-width: 2; }
details summary { cursor: pointer; font-weight: 600; margin: 6px 0; }
code { font-size: 12px; }
"""


def _e(text) -> str:
    return html.escape(str(text))


def _num(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    a = abs(x)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def _marker(shape: str, cls: str, x: float, y: float, title: str) -> str:
    t = f"<title>{_e(title)}</title>"
    if shape == "square":
        return f'<rect class="{cls} mark" x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1">{t}</rect>'
    if shape == "triangle":
        return f'<path class="{cls} mark" d="M{x:.1f},{y - 5.5:.1f} L{x + 5.5:.1f},{y + 4.5:.1f} L{x - 5.5:.1f},{y + 4.5:.1f} Z">{t}</path>'
    return f'<circle class="{cls} mark" cx="{x:.1f}" cy="{y:.1f}" r="4.5">{t}</circle>'


def _legend(methods) -> str:
    items = []
    for m in methods:
        shape = SHAPES[m]
        icon = {"circle": '<circle cx="7" cy="7" r="4.5"', "square": '<rect x="2.5" y="2.5" width="9" height="9" rx="1"',
                "triangle": '<path d="M7,1.5 L12.5,11.5 L1.5,11.5 Z"'}[shape]
        items.append(f'<span><svg width="14" height="14" aria-hidden="true">{icon} class="s{SERIES[m]}"/></svg>{_e(METHODS[m].label)}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def _log_ticks(low: float, high: float) -> list[float]:
    ticks = [10.0 ** k for k in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)]
    return [t for t in ticks if low * 0.999 <= t <= high * 1.001]


def _tick_text(v: float) -> str:
    if v >= 1e4 or v < 1e-2:
        return f"1e{int(round(math.log10(v)))}" if abs(math.log10(v) - round(math.log10(v))) < 1e-9 else f"{v:.0e}"
    return f"{v:g}"


def line_chart(series: dict, xs: list, x_label: str, y_label: str, floor: float, limit: float | None = None, limit_label: str = "",
               width: int = 560, height: int = 300) -> str:
    """Lines over a categorical-but-ordered x (dimensions), log y. series: method -> {x: [values per seed]}; the line joins the
    medians and every seed is a marker. Values below `floor` are drawn at the floor."""
    left, right, top, bottom = 62, 150, 14, 44
    pw, ph = width - left - right, height - top - bottom
    values = [v for pts in series.values() for vs in pts.values() for v in vs if v is not None]
    if not values:
        return '<p class="muted">No results to chart.</p>'
    low = max(floor, min(max(v, floor) for v in values))
    high = max(max(values), limit or 0, low * 10)
    low_e, high_e = math.floor(math.log10(low)), math.ceil(math.log10(high))
    low, high = 10.0 ** low_e, 10.0 ** high_e
    sx = lambda i: left + (i + 0.5) * pw / len(xs)
    sy = lambda v: top + ph * (1 - (math.log10(max(v, floor)) - low_e) / (high_e - low_e))
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" role="img" aria-label="{_e(y_label)} against {_e(x_label)}">']
    for t in _log_ticks(low, high):
        out.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text x="{left - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{_tick_text(t)}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}"/>')
    for i, x in enumerate(xs):
        out.append(f'<text x="{sx(i):.1f}" y="{top + ph + 16}" text-anchor="middle">{_e(x)}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">{_e(x_label)}</text>')
    out.append(f'<text transform="translate(14,{top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">{_e(y_label)}</text>')
    if limit is not None:
        out.append(f'<line class="limit" x1="{left}" x2="{left + pw}" y1="{sy(limit):.1f}" y2="{sy(limit):.1f}"/>')
        out.append(f'<text class="label" x="{left + pw + 6}" y="{sy(limit) + 4:.1f}">{_e(limit_label)}</text>')
    ends = []
    for m, pts in series.items():
        cls = f"s{SERIES[m]}"
        medians = [(i, float(np.median([v for v in pts[x] if v is not None]))) for i, x in enumerate(xs) if pts.get(x) and any(v is not None for v in pts[x])]
        if len(medians) > 1:
            d = " ".join(f"{'M' if j == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for j, (i, v) in enumerate(medians))
            out.append(f'<path class="{cls} line" d="{d}"/>')
        for i, x in enumerate(xs):
            for v in pts.get(x, []):
                if v is not None:
                    out.append(_marker(SHAPES[m], cls, sx(i), sy(v), f"{METHODS[m].label}, {x_label} {x}: {_num(v, 4)}"))
        if medians:
            ends.append((sy(medians[-1][1]), m))
    ends.sort()
    last = -1e9
    for y, m in ends:   # end labels, nudged only enough not to overlap
        y = max(y, last + 13)
        last = y
        out.append(f'<text class="label" x="{left + pw + 6}" y="{y + 4:.1f}">{_e(m)}</text>')
    out.append("</svg>")
    return "".join(out)


def dot_plot(rows: list[str], series: dict, x_label: str, floor: float, limit: float | None = None, limit_label: str = "",
             width: int = 760) -> str:
    """One row per target, one marker per seed and method, log x. series: method -> {row: [values]}."""
    left, right, top, bottom, band = 230, 120, 10, 40, 30
    height = top + bottom + band * len(rows)
    pw = width - left - right
    values = [v for pts in series.values() for vs in pts.values() for v in vs if v is not None]
    if not values:
        return '<p class="muted">No results to chart.</p>'
    low_e = math.floor(math.log10(max(floor, min(max(v, floor) for v in values))))
    high_e = math.ceil(math.log10(max(max(values), limit or 0, 10.0 ** (low_e + 1))))
    sx = lambda v: left + pw * (math.log10(max(v, floor)) - low_e) / (high_e - low_e)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" role="img" aria-label="{_e(x_label)} by target">']
    for t in _log_ticks(10.0 ** low_e, 10.0 ** high_e):
        out.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{top + band * len(rows)}"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{top + band * len(rows) + 14}" text-anchor="middle">{_tick_text(t)}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">{_e(x_label)}</text>')
    if limit is not None:
        out.append(f'<line class="limit" x1="{sx(limit):.1f}" x2="{sx(limit):.1f}" y1="{top}" y2="{top + band * len(rows)}"/>')
        out.append(f'<text class="label" x="{sx(limit) + 4:.1f}" y="{top + 10}">{_e(limit_label)}</text>')
    offsets = {m: (k - (len(series) - 1) / 2) * 8 for k, m in enumerate(series)}
    for r, row in enumerate(rows):
        y0 = top + band * r + band / 2
        out.append(f'<text class="label" x="{left - 8}" y="{y0 + 4:.1f}" text-anchor="end">{_e(row)}</text>')
        if r:
            out.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{top + band * r}" y2="{top + band * r}"/>')
        for m, pts in series.items():
            for v in pts.get(row, []):
                if v is not None:
                    out.append(_marker(SHAPES[m], f"s{SERIES[m]}", sx(v), y0 + offsets[m], f"{METHODS[m].label}, {row}: {_num(v, 4)}"))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------------------


def _load(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _evidence_ok(r: dict, name: str):
    e = r.get("evidence", {}).get(name)
    return None if e is None else bool(e.get("ok"))


def _abs_evidence_error(r: dict, name: str):
    e = r.get("evidence", {}).get(name) or {}
    return None if e.get("error") is None else abs(e["error"])


def _summary(results: list[dict], methods: list[str]) -> str:
    rows = []
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        ok = [r for r in rs if r.get("status") == "ok"]
        accurate = sum(r["posterior_ok"] for r in ok)
        wrong = sum(r.get("confidently_wrong", False) for r in ok)
        failed = len(rs) - len(ok)
        names = sorted({n for r in ok for n in r.get("evidence", {})})
        ev = []
        for n in names:
            have = [r for r in ok if n in r.get("evidence", {})]
            ev.append(f"{EVIDENCE_LABEL.get(n, n)} {sum(_evidence_ok(r, n) for r in have)}/{len(have)}")
        seconds = [r["seconds"] for r in ok if "seconds" in r]
        grads = [r["diagnostics"].get("gradient_evaluations") for r in ok if r.get("diagnostics", {}).get("gradient_evaluations")]
        flagged = sum(1 for r in ok if r.get("checks_passed") is False)
        flagged_right = sum(1 for r in ok if r.get("checks_passed") is False and not r["posterior_ok"])
        share = accurate / len(ok) if ok else 0
        pill = "good" if share >= 0.95 and not wrong else "mid" if share >= 0.8 else "bad"
        icon = {"good": "✓", "mid": "!", "bad": "✗"}[pill]
        rows.append(
            f'<tr><td class="l"><b>{_e(m)}</b><br><span class="muted">{_e(METHODS[m].label)}</span></td>'
            f'<td><span class="pill {pill}">{icon} {accurate}/{len(ok)}</span></td>'
            f'<td>{wrong}</td><td>{flagged} ({flagged_right} rightly)</td><td>{failed}</td>'
            f'<td class="l">{"<br>".join(ev) or "–"}</td>'
            f'<td>{_num(float(np.median(seconds)), 1) if seconds else "–"}</td>'
            f'<td>{_num(int(np.median(grads))) if grads else "–"}</td></tr>')
    return ('<table><tr><th class="l">Method</th><th>Posterior accurate</th><th>Confidently wrong</th><th>Flagged by own checks</th>'
            '<th>Crashed or timed out</th><th class="l">Evidence within 0.5</th><th>Median seconds</th><th>Median gradient evaluations</th></tr>'
            + "".join(rows) + "</table>")


def _matrix(results: list[dict], targets: list[str], methods: list[str], info: dict) -> str:
    head = "".join(f"<th>{_e(m)}</th>" for m in methods)
    rows, group = [], None
    for t in targets:
        if info[t]["group"] != group:
            group = info[t]["group"]
            rows.append(f'<tr><td class="l" colspan="{len(methods) + 2}"><b>{_e(group)}</b></td></tr>')
        cells = []
        for m in methods:
            rs = [r for r in results if r["target"] == t and r["method"] == m]
            if not rs:
                cells.append('<td class="muted">–</td>')
                continue
            good = sum(r.get("posterior_ok", False) for r in rs)
            wrong = sum(r.get("confidently_wrong", False) for r in rs)
            crashed = sum(r.get("status") != "ok" for r in rs)
            pill = "good" if good == len(rs) else "bad" if good == 0 else "mid"
            icon = {"good": "✓", "mid": "!", "bad": "✗"}[pill]
            note = []
            if wrong:
                note.append(f"{wrong} confidently wrong")
            if crashed:
                note.append(f"{crashed} crashed")
            tip = "; ".join(f"seed {r['seed']}: " + ("accurate" if r.get("posterior_ok") else
                            (r.get("error", "")[:120] if r.get("status") != "ok" else ", ".join(r.get("accuracy", {}).get("failed", []))))
                            for r in rs)
            cells.append(f'<td title="{_e(tip)}"><span class="pill {pill}">{icon} {good}/{len(rs)}</span>'
                         + (f'<br><span class="muted">{_e(", ".join(note))}</span>' if note else "") + "</td>")
        rows.append(f'<tr><td class="l">{_e(t)}</td><td>{info[t]["size"]}</td>' + "".join(cells) + "</tr>")
    return f'<table><tr><th class="l">Target</th><th>Unknowns</th>{head}</tr>' + "".join(rows) + "</table>"


def _detail(results: list[dict], target: str, info: dict) -> str:
    i = info[target]
    rows = []
    for r in sorted((r for r in results if r["target"] == target), key=lambda r: (list(METHODS).index(r["method"]), r["seed"])):
        if r.get("status") != "ok":
            rows.append(f'<tr><td class="l">{_e(r["method"])}</td><td>{r["seed"]}</td><td class="l fail" colspan="12">'
                        f'✗ {_e(r.get("status"))}: {_e(r.get("error", "")[:300])}</td></tr>')
            continue
        a, d = r["accuracy"], r.get("diagnostics", {})
        ev = []
        for n, e in r.get("evidence", {}).items():
            if e.get("log_value") is None:
                ev.append(f"{EVIDENCE_LABEL.get(n, n)}: {_e(e.get('error_message', 'none')[:80])}")
            else:
                se = f" ± {_num(e['standard_error'], 2)}" if e.get("standard_error") is not None else ""
                cls = "pass" if e["ok"] else "fail"
                ev.append(f'{EVIDENCE_LABEL.get(n, n)}: {_num(e["log_value"], 2)}{se} <span class="{cls}">({e["error"]:+.2f})</span>')
        peaks = f"{', '.join(_num(s, 3) for s in a['peak_shares'])}" if "peak_shares" in a else "–"
        findings = "; ".join(d.get("findings") or []) if d.get("findings") else ("none" if d.get("converged") else "–")
        status = '<span class="pass">✓ accurate</span>' if r["posterior_ok"] else (
            f'<span class="fail">✗ {"confidently wrong" if r.get("confidently_wrong") else "inaccurate, flagged"}</span>')
        rows.append(
            f'<tr><td class="l">{_e(r["method"])}</td><td>{r["seed"]}</td><td class="l">{status}</td>'
            f'<td>{_num(a.get("max_abs_mean_error"))}<br><span class="muted">{_e(a.get("worst_quantity", ""))}</span></td>'
            f'<td>{_num(a.get("min_sd_ratio"))}–{_num(a.get("max_sd_ratio"))}</td><td>{_num(a.get("max_ks"))}</td><td>{peaks}</td>'
            f'<td class="l">{"<br>".join(ev) or "–"}</td>'
            f'<td>{_num(d.get("max_rhat"))}</td><td>{_num(d.get("min_ess"), 0)}</td><td>{_num(d.get("divergences"))}</td>'
            f'<td>{_num(d.get("rungs"))}{"" if d.get("round_trips") is None else " / " + _num(d.get("round_trips"))}</td>'
            f'<td>{_num(d.get("gradient_evaluations"))}</td><td>{_num(r.get("seconds"), 1)}</td>'
            f'<td class="l">{_e(findings)}</td></tr>')
    peaks_note = " Peak weights: the share of draws in each peak, true 0.3 and 0.7." if i.get("has_peaks") else ""
    return (f'<h3 id="{_e(target)}">{_e(target)}</h3><p>{_e(i["description"])}. {i["size"]} unknowns, '
            f'{i["n_quantities"]} checked quantities. Reference: {_e(i["reference_source"])}; true log evidence '
            f'{_num(i["true_log_evidence"], 3)}.{peaks_note}</p><div class="card"><table><tr><th class="l">Method</th><th>Seed</th>'
            '<th class="l">Result</th><th>Worst mean error (sds)</th><th>sd ratio range</th><th>Largest KS distance</th><th>Peak weights</th>'
            '<th class="l">Log evidence (error)</th><th>R-hat</th><th>ESS</th><th>Divergences</th><th>Rungs / round trips</th>'
            '<th>Gradient evaluations</th><th>Seconds</th><th class="l">Own checks\' findings</th></tr>' + "".join(rows) + "</table></div>")


def write(run_dir: Path) -> Path:
    results = _load(run_dir)
    info = {}
    for r in results:
        if "group" in r:
            info.setdefault(r["target"], r)
    from benchmarks.samplers.targets import TARGETS
    targets = [t for t in TARGETS if t in info]
    methods = [m for m in METHODS if any(r["method"] == m for r in results)]
    main = [m for m in MAIN if m in methods]
    ok = [r for r in results if r.get("status") == "ok"]
    info_path = run_dir / "run.json"
    run_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    suite = run_info.get("suite") or (results[0].get("suite") if results else None) or "unnamed"
    wall = run_info.get("total_seconds")
    seeds = defaultdict(set)
    for r in results:
        seeds[r["method"]].add(r["seed"])
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>Sampler benchmark: {_e(suite)} suite</title><style>{CSS}</style></head><body><main>",
             f"<h1>Sampler benchmark: {_e(suite)} suite</h1><p class='muted'>Run <code>{_e(run_dir.name)}</code>, report written "
             f"{time.strftime('%Y-%m-%d %H:%M')}.</p>",
             f"<div class='card'><p><b>Suite:</b> {_e(suite)}: {_e(run_info.get('description', ''))}.</p>"
             f"<p><b>Size:</b> {len(results)} runs recorded of {run_info.get('planned_runs', len(results))} planned ({len(ok)} completed): "
             f"{len(methods)} methods on {len(targets)} targets; seeds per method: "
             + ", ".join(f"{_e(m)} {len(seeds[m])}" for m in methods) + ".</p>"
             f"<p><b>Wall time:</b> {_num(wall / 60, 1) + ' minutes' if wall else 'not recorded'} for the whole suite"
             + (" (" + "; ".join(f"{s_['runs']} runs from {s_['start']} to {s_['end']}" for s_ in run_info.get("sessions", [])) + ")"
                if len(run_info.get("sessions", [])) > 1 else "")
             + f", including compilation and building the references, with {run_info.get('jobs', '?')} runs at a time, each pinned to "
             f"{run_info.get('threads_per_job', '?')} CPU(s), on a machine with {run_info.get('cpus_available', '?')} CPUs. Times are "
             "comparable within this report only.</p></div>"]

    parts.append(
        "<h2>How to read this</h2><p>Every target is a posterior whose answer is known exactly, or by numerical integration fine "
        "enough to be exact for these purposes (each reference is checked independently in <code>tests/benchmarks</code>). A run is "
        f"<b>accurate</b> when, over every checked quantity, the posterior mean is within {MEAN_ERROR_LIMIT:g} posterior standard "
        f"deviations of the truth and the posterior standard deviation is within a factor {SD_RATIO_RANGE[0]:g} to {SD_RATIO_RANGE[1]:g} "
        f"of the truth; for targets with two peaks, each peak's share of the draws must also be within {PEAK_WEIGHT_LIMIT:g} of its "
        f"true weight. The evidence (log marginal likelihood, log p(data | model)) is accurate when within {EVIDENCE_LIMIT:g} of the "
        "truth. These criteria apply to the worst quantity, so targets with many quantities are held to a stricter standard; "
        "sampling noise alone gives a worst mean error of about 3 / √ESS.</p><p>A run is <b>confidently wrong</b> when it is "
        "inaccurate but the method's own checks (R-hat, effective sample size, divergences, round trips) raised nothing. SMC has no "
        "checks of its own, so every inaccurate SMC run counts as confidently wrong. Hover over a cell or marker for details.</p>")

    parts.append("<h2>Summary</h2><div class='card'>" + _summary(results, methods) + "</div>")
    parts.append("<h2>Accuracy by target</h2><p>Seeds accurate out of seeds run. Hover over a cell for what failed.</p>"
                 "<div class='card'>" + _matrix(results, targets, methods, info) + "</div>")

    rows = targets
    err = {m: {t: [r["accuracy"].get("max_abs_mean_error") for r in ok if r["method"] == m and r["target"] == t] for t in rows} for m in main}
    parts.append("<h2>Worst error in the posterior mean, every target</h2><p>In posterior standard deviations; one marker per seed. "
                 "The red line is the criterion.</p><div class='card'>" + _legend(main) +
                 dot_plot(rows, err, "worst error in the posterior mean (posterior sds, log scale)", 1e-3, MEAN_ERROR_LIMIT, "criterion") + "</div>")
    spread = {m: {t: [max(abs(math.log(r["accuracy"]["min_sd_ratio"])), abs(math.log(r["accuracy"]["max_sd_ratio"])))
                      for r in ok if r["method"] == m and r["target"] == t and r["accuracy"].get("min_sd_ratio")] for t in rows} for m in main}
    parts.append("<h2>Worst error in the posterior spread, every target</h2><p>|log(sd ratio)| for the worst quantity: 0.1 is about "
                 "a 10% error. The red line is the criterion (a ratio of 0.9).</p><div class='card'>" + _legend(main) +
                 dot_plot(rows, spread, "worst |log(sd ratio)| (log scale)", 1e-3, -math.log(SD_RATIO_RANGE[0]), "criterion") + "</div>")
    ev = {m: {t: [_abs_evidence_error(r, EVIDENCE_OF[m]) for r in ok if r["method"] == m and r["target"] == t] for t in rows} for m in main}
    parts.append("<h2>Error in the evidence, every target</h2><p>|estimated − true log evidence|: stepping stones for the default, "
                 "bridge sampling for NUTS, SMC's own estimate for SMC. The red line is the criterion.</p><div class='card'>"
                 + _legend(main) + dot_plot(rows, ev, "|error in log evidence| (log scale)", 1e-3, EVIDENCE_LIMIT, "criterion") + "</div>")
    cost = {m: {t: [r.get("diagnostics", {}).get("gradient_evaluations") for r in ok if r["method"] == m and r["target"] == t] for t in rows} for m in main}
    parts.append("<h2>Cost, every target</h2><p>Gradient evaluations per run (one evaluation of the log density and its gradient "
                 "at one point), which does not depend on what else the machine was doing.</p><div class='card'>" + _legend(main)
                 + dot_plot(rows, cost, "gradient evaluations (log scale)", 1.0) + "</div>")
    secs = {m: {t: [r.get("seconds") for r in ok if r["method"] == m and r["target"] == t] for t in rows} for m in main}
    parts.append("<h2>Wall time, every target</h2><p>Seconds per run, including compilation (the first seed of each method on "
                 "each target compiles; later ones reuse the compiled code from disk).</p><div class='card'>" + _legend(main)
                 + dot_plot(rows, secs, "seconds (log scale)", 0.1) + "</div>")

    dims = [t for t in targets if t.startswith("normal_means_d")]
    if len(dims) > 1:
        xs = [str(info[t]["size"]) for t in dims]
        def by_dim(f):
            return {m: {str(info[t]["size"]): [f(r) for r in ok if r["method"] == m and r["target"] == t] for t in dims} for m in main}
        parts.append("<h2>Dimension: independent normal means</h2><p>The same problem in 2 to 200 dimensions. Lines join the "
                     "medians over seeds; each marker is one seed.</p>")
        parts.append("<div class='card'><h3>Worst error in the posterior mean</h3>" + _legend(main) + line_chart(
            by_dim(lambda r: r["accuracy"].get("max_abs_mean_error")), xs, "unknowns", "posterior sds (log scale)", 1e-3, MEAN_ERROR_LIMIT, "criterion") + "</div>")
        parts.append("<div class='card'><h3>Error in the evidence</h3>" + _legend(main) + line_chart(
            {m: by_dim(lambda r, m=m: _abs_evidence_error(r, EVIDENCE_OF[m]))[m] for m in main}, xs, "unknowns",
            "|error in log evidence| (log scale)", 1e-3, EVIDENCE_LIMIT, "criterion") + "</div>")
        parts.append("<div class='card'><h3>Gradient evaluations</h3>" + _legend(main) + line_chart(
            by_dim(lambda r: r.get("diagnostics", {}).get("gradient_evaluations")), xs, "unknowns", "gradient evaluations (log scale)", 1.0) + "</div>")

    wrong = [r for r in ok if r.get("confidently_wrong")]
    parts.append("<h2>Confidently wrong runs</h2>")
    if wrong:
        body = "".join(f'<tr><td class="l"><a href="#{_e(r["target"])}">{_e(r["target"])}</a></td><td class="l">{_e(r["method"])}</td>'
                       f'<td>{r["seed"]}</td><td class="l">{_e("; ".join(r["accuracy"].get("failed", [])))}</td>'
                       f'<td>{_num(r.get("diagnostics", {}).get("max_rhat"))}</td><td>{_num(r.get("diagnostics", {}).get("min_ess"), 0)}</td></tr>'
                       for r in sorted(wrong, key=lambda r: (r["method"], r["target"], r["seed"])))
        parts.append(f"<p>{len(wrong)} runs were inaccurate while their own checks raised nothing.</p><div class='card'><table>"
                     "<tr><th class='l'>Target</th><th class='l'>Method</th><th>Seed</th><th class='l'>What was wrong</th><th>R-hat</th><th>ESS</th></tr>"
                     + body + "</table></div>")
    else:
        parts.append("<p>None.</p>")
    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        parts.append("<h2>Runs that crashed or timed out</h2><div class='card'><table><tr><th class='l'>Target</th><th class='l'>Method</th>"
                     "<th>Seed</th><th class='l'>Error</th></tr>" + "".join(
                         f'<tr><td class="l">{_e(r["target"])}</td><td class="l">{_e(r["method"])}</td><td>{r["seed"]}</td>'
                         f'<td class="l">{_e(str(r.get("error", ""))[:400])}</td></tr>' for r in failed) + "</table></div>")

    parts.append("<h2>Every target in detail</h2><p>One row per run. Mean errors are in posterior standard deviations; the "
                 "largest Kolmogorov–Smirnov distance is the biggest gap between the draws' and the true cumulative distribution "
                 "over the quantities where it is known. Rungs / round trips: the ladder's size and how many times a state "
                 "travelled from the reference end to the posterior end and back.</p>")
    group = None
    for t in targets:
        if info[t]["group"] != group:
            group = info[t]["group"]
            parts.append(f"<h2 style='font-size:17px'>{_e(group)}</h2>")
        parts.append(_detail(results, t, info))
    parts.append("<h2>Methods</h2><ul>" + "".join(f"<li><b>{_e(m)}</b>: {_e(METHODS[m].label)}</li>" for m in methods) + "</ul>")
    parts.append("</main></body></html>")
    path = run_dir / "report.html"
    path.write_text("".join(parts))
    return path
