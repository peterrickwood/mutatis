/* The mutatis inspector, in the browser. It draws what the server's JSON API returns (see mutatis/web/server.py) and computes
   nothing about the model itself: the graph's layout, the equations as LaTeX, the densities and the fitted values all come from
   Python. Nothing here knows what kind of model it is looking at. */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const enc = encodeURIComponent;
let uniqueId = 0;

// ---------------------------------------------------------------------------------------------------------------- helpers

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  setProps(node, props);
  append(node, children);
  return node;
}

function sv(tag, props = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  setProps(node, props);
  append(node, children);
  return node;
}

function setProps(node, props) {
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "text") node.textContent = v;
    else if (k === "class") node.setAttribute("class", v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
}

function append(node, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false || c === "") continue;
    node.append(c instanceof Node ? c : String(c));
  }
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  return String(Number(v.toPrecision(digits)));
}

function niceTicks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (!(hi > lo)) return [lo];
  const rough = (hi - lo) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const ratio = rough / magnitude;
  const step = (ratio >= 7.5 ? 10 : ratio >= 3.5 ? 5 : ratio >= 1.5 ? 2 : 1) * magnitude;
  const ticks = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step) ticks.push(Number(t.toFixed(12)));
  return ticks;
}

function shorten(text, max) {
  text = String(text ?? "");
  return text.length > max ? text.slice(0, Math.max(1, max - 1)) + "…" : text;
}

function finite(values) {
  return values.filter(v => typeof v === "number" && Number.isFinite(v));
}

/** Mathematics from LaTeX made in Python; the specification's own text when KaTeX is not available or cannot draw it. */
function math(latex, text, display = false) {
  const span = el("span", {class: display ? "math display" : "math"});
  if (latex && typeof window !== "undefined" && window.katex) {
    try {
      window.katex.render(latex, span, {displayMode: display, throwOnError: true, strict: "ignore"});
      return span;
    } catch (error) { /* shown as text below */ }
  }
  span.classList.add("plain");
  span.textContent = text || latex || "";
  return span;
}

const answers = new Map();
function api(path) {
  if (!answers.has(path)) {
    const promise = fetch(path).then(async response => {
      let body = {};
      try { body = await response.json(); } catch (error) { /* not JSON */ }
      if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
      return body;
    });
    promise.catch(() => answers.delete(path));   // a failed answer may be asked for again
    answers.set(path, promise);
  }
  return answers.get(path);
}

/** A box that shows `waiting` until the promise settles, then what `build` makes of the answer, or the error. */
function later(promise, build, waiting = "Loading…") {
  const box = el("div", {}, el("p", {class: "loading", text: waiting}));
  promise.then(data => box.replaceChildren(build(data))).catch(error => {
    console.error(error);
    box.replaceChildren(el("p", {class: "error", text: String(error && error.message || error)}));
  });
  return box;
}

function editText(edit) {
  const [op, args] = Object.entries(edit)[0] || ["?", {}];
  return `${op}: ${JSON.stringify(args)}`;
}

function editList(edits) {
  if (!edits || !edits.length) return el("span", {class: "muted", text: "none"});
  return el("ul", {class: "edit-list"}, edits.map(e => {
    const text = editText(e);
    return el("li", {}, el("code", {text: shorten(text, 160), title: text.length > 160 ? text : null}));
  }));
}

function facts(pairs) {
  return el("dl", {class: "facts"}, pairs.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => [el("dt", {text: k}), el("dd", {}, v)]));
}

function legend(items) {
  return el("div", {class: "legend"}, items.map(([kind, cls, label]) => el("span", {}, el("span", {class: `key ${kind} ${cls}`}), label)));
}

// ---------------------------------------------------------------------------------------------------------------- charts

/** A tooltip inside a chart's box. */
function tooltip(box) {
  const tip = el("div", {class: "tooltip", hidden: true});
  box.append(tip);
  return {
    show(left, top, rows) {
      tip.replaceChildren(...rows.map(([cls, value, label]) => el("div", {class: "row"}, cls ? el("span", {class: `key ${cls}`}) : null, el("b", {text: value}), label ? el("span", {class: "muted", text: label}) : null)));
      tip.hidden = false;
      const width = box.clientWidth || 300;
      tip.style.left = `${Math.min(Math.max(0, left + 12), Math.max(0, width - tip.offsetWidth - 4))}px`;
      tip.style.top = `${top}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function pathThrough(points) {
  let d = "", pen = false;
  for (const p of points) {
    if (!p) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`;
    pen = true;
  }
  return d;
}

function bandPath(xs, lower, upper, x, y) {
  let d = "";
  let run = [];
  const flush = () => {
    if (run.length) {
      const top = run.map(i => `${x(xs[i]).toFixed(2)},${y(upper[i]).toFixed(2)}`);
      const bottom = run.slice().reverse().map(i => `${x(xs[i]).toFixed(2)},${y(lower[i]).toFixed(2)}`);
      d += `M${top.join("L")}L${bottom.join("L")}Z`;
    }
    run = [];
  };
  xs.forEach((_, i) => (Number.isFinite(lower[i]) && Number.isFinite(upper[i]) ? run.push(i) : flush()));
  flush();
  return d;
}

/**
 * A chart of lines and bands over a shared x axis, with a crosshair that snaps to the nearest x and lists every layer there.
 * layers: {kind: "line", values, cls: "line-1", name, key} or {kind: "band", lower, upper, cls: "band-1", name, key}
 */
function xyChart({xs, xLabel = v => fmt(v), layers, width = 640, height = 230, title, caption, zero = false}) {
  const box = el("div", {class: "chart"});
  if (title) box.append(el("div", {class: "title", text: title}));
  const pad = {l: 46, r: 12, t: 10, b: 26};
  const values = finite(layers.flatMap(l => (l.kind === "band" ? [...l.lower, ...l.upper] : l.values)));
  let lo = Math.min(...values), hi = Math.max(...values);
  if (zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (!values.length) { lo = 0; hi = 1; }
  if (hi === lo) { lo -= 1; hi += 1; }
  const margin = (hi - lo) * 0.05;
  lo -= margin; hi += margin;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const x = v => pad.l + (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const y = v => height - pad.b - (v - lo) / (hi - lo) * (height - pad.t - pad.b);
  const svg = sv("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title || "chart"});
  for (const t of niceTicks(lo, hi, 5)) {
    svg.append(sv("line", {x1: pad.l, x2: width - pad.r, y1: y(t), y2: y(t), class: t === 0 && zero ? "baseline" : "gridline"}));
    svg.append(sv("text", {x: pad.l - 6, y: y(t) + 3.5, "text-anchor": "end", text: fmt(t, 3)}));
  }
  const every = Math.max(1, Math.ceil(xs.length / 7));
  xs.forEach((v, i) => {
    if (i % every === 0) svg.append(sv("text", {x: x(v), y: height - 8, "text-anchor": "middle", text: xLabel(v, i)}));
  });
  svg.append(sv("line", {x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b, class: "baseline"}));
  for (const layer of layers.filter(l => l.kind === "band")) svg.append(sv("path", {d: bandPath(xs, layer.lower, layer.upper, x, y), class: layer.cls}));
  for (const layer of layers.filter(l => l.kind !== "band")) {   // every line above every band, so no wash tints a line
    svg.append(sv("path", {d: pathThrough(xs.map((v, i) => (Number.isFinite(layer.values[i]) ? [x(v), y(layer.values[i])] : null))), class: layer.cls}));
  }
  const cross = sv("line", {y1: pad.t, y2: height - pad.b, class: "crosshair", visibility: "hidden"});
  svg.append(cross);
  const hit = sv("rect", {x: pad.l, y: pad.t, width: width - pad.l - pad.r, height: height - pad.t - pad.b, fill: "transparent"});
  svg.append(hit);
  box.append(svg);
  if (caption) box.append(el("div", {class: "caption", text: caption}));
  const tip = tooltip(box);
  hit.addEventListener("pointermove", event => {
    const r = svg.getBoundingClientRect();
    const px = (event.clientX - r.left) / (r.width || 1) * width;
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(x(v) - px) < Math.abs(x(xs[best]) - px)) best = i; });
    cross.setAttribute("x1", x(xs[best]));
    cross.setAttribute("x2", x(xs[best]));
    cross.setAttribute("visibility", "visible");
    const rows = [[null, xLabel(xs[best], best), ""]];
    for (const layer of layers) {
      if (!layer.name) continue;
      rows.push(layer.kind === "band" ? [layer.key, `${fmt(layer.lower[best])} to ${fmt(layer.upper[best])}`, layer.name] : [layer.key, fmt(layer.values[best]), layer.name]);
    }
    tip.show(x(xs[best]) / width * (r.width || width), 24, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
  return box;
}

/** One element of a parameter: the posterior as a histogram of the stored draws, the prior as its density (or a histogram of draws from it). */
function densityChart(e, {compact = false} = {}) {
  const width = compact ? 150 : 260, height = compact ? 34 : 118;
  const pad = compact ? {l: 1, r: 1, t: 2, b: 2} : {l: 8, r: 8, t: 6, b: 20};
  const [lo, hi] = e.window;
  const x = v => pad.l + (v - lo) / (hi - lo) * (width - pad.l - pad.r);
  const posterior = e.posterior || [];
  const prior = e.prior ? e.prior.density : [];
  const top = Math.max(...finite([...posterior, ...prior]), 1e-12) * 1.05;
  const y = d => height - pad.b - Math.min(d, top) / top * (height - pad.t - pad.b);
  const svg = sv("svg", {viewBox: `0 0 ${width} ${height}`, class: compact ? "spark" : null, role: "img",
                         "aria-label": `${e.label}: posterior ${e.prior ? "and prior" : ""}`});
  const step = (edges, values) => edges.slice(0, -1).flatMap((b, i) => [[x(b), y(values[i] || 0)], [x(edges[i + 1]), y(values[i] || 0)]]);
  if (posterior.length) {
    const outline = step(e.bins, posterior);
    svg.append(sv("path", {d: `${pathThrough([[x(e.bins[0]), y(0)], ...outline, [x(e.bins[e.bins.length - 1]), y(0)]])}Z`, class: "band-1"}));
    svg.append(sv("path", {d: pathThrough(outline), class: "line-1"}));
  }
  if (e.prior && e.prior.x) svg.append(sv("path", {d: pathThrough(e.prior.x.map((v, i) => [x(v), y(e.prior.density[i])])), class: "line-2"}));
  else if (e.prior) svg.append(sv("path", {d: pathThrough(step(e.prior.bins, e.prior.density)), class: "line-2"}));
  svg.append(sv("line", {x1: pad.l, x2: width - pad.r, y1: y(0), y2: y(0), class: "baseline"}));
  if (compact) return svg;
  for (const t of niceTicks(lo, hi, 4)) svg.append(sv("text", {x: x(t), y: height - 6, "text-anchor": "middle", text: fmt(t, 3)}));
  const box = el("div", {class: "chart"}, el("div", {class: "title", text: e.label}), svg);
  const parts = [];
  if (e.mean !== undefined) parts.push(`mean ${fmt(e.mean)} · sd ${fmt(e.sd)} · 90%: ${fmt(e.q05)} to ${fmt(e.q95)}`);
  if (e.prior_wider) parts.push("the prior is much wider than the axis");
  if (e.outside > 0.001) parts.push(`${fmt(100 * e.outside, 2)}% of draws beyond the axis`);
  if (parts.length) box.append(el("div", {class: "caption", text: parts.join(" · ")}));
  const cross = sv("line", {y1: pad.t, y2: height - pad.b, class: "crosshair", visibility: "hidden"});
  const hit = sv("rect", {x: pad.l, y: 0, width: width - pad.l - pad.r, height, fill: "transparent"});
  svg.append(cross, hit);
  const tip = tooltip(box);
  const at = (edges, values, v) => { for (let i = 0; i < edges.length - 1; i++) if (v >= edges[i] && v <= edges[i + 1]) return values[i]; return null; };
  hit.addEventListener("pointermove", event => {
    const r = svg.getBoundingClientRect();
    const px = (event.clientX - r.left) / (r.width || 1) * width;
    const v = lo + (px - pad.l) / (width - pad.l - pad.r) * (hi - lo);
    cross.setAttribute("x1", px);
    cross.setAttribute("x2", px);
    cross.setAttribute("visibility", "visible");
    const rows = [[null, fmt(v), "value"]];
    if (posterior.length) rows.push(["k1", fmt(at(e.bins, posterior, v)), "posterior density"]);
    if (e.prior && e.prior.x) {
      let best = 0;
      e.prior.x.forEach((g, i) => { if (Math.abs(g - v) < Math.abs(e.prior.x[best] - v)) best = i; });
      rows.push(["k2", fmt(e.prior.density[best]), "prior density"]);
    } else if (e.prior) rows.push(["k2", fmt(at(e.prior.bins, e.prior.density, v)), "prior density (from draws)"]);
    tip.show(px / width * (r.width || width), 20, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
  return box;
}

const DENSITY_LEGEND = () => legend([["box", "k1", "posterior: the stored draws"], ["", "k2", "prior"]]);

/** One parameter: its prior, then a panel per element, or a table when there are many elements. */
function parameterBlock(entry) {
  const out = el("div", {class: "param"});
  const head = el("div", {class: "param-head"}, el("span", {class: "key-name", text: entry.key}));
  if (entry.axes && entry.axes.length) head.append(el("span", {class: "chip", text: `over ${entry.axes.join(" × ")}`}));
  if (entry.kind === "fixed") {
    head.append(el("span", {class: "muted", text: "fixed at"}), el("code", {text: shorten(JSON.stringify(entry.fixed), 400)}));
    out.append(head);
    return out;
  }
  if (entry.prior) head.append(el("span", {class: "muted", text: "prior"}), math(entry.prior_latex, entry.prior));
  out.append(head);
  if (entry.kind === "no_prior") { out.append(el("p", {class: "note", text: entry.note})); return out; }
  if (entry.prior_note) out.append(el("p", {class: "note", text: entry.prior_note}));
  const elements = entry.elements || [];
  if (elements.length <= 16) {
    const matrix = entry.shape && entry.shape.length === 2 && elements.length > 1;
    const grid = el("div", {class: matrix ? "panels matrix" : "panels", style: matrix ? `--columns: ${entry.shape[1]}` : null});
    elements.forEach(e => grid.append(densityChart(e)));
    out.append(grid);
  } else {
    out.append(elementTable(elements));
  }
  return out;
}

function elementTable(elements) {
  const filter = el("input", {type: "search", placeholder: "filter elements", "aria-label": "filter elements"});
  const rows = elements.map(e => el("tr", {"data-label": e.label},
    el("td", {}, el("code", {text: e.label})), el("td", {class: "num", text: fmt(e.mean)}), el("td", {class: "num", text: fmt(e.sd)}),
    el("td", {class: "num", text: fmt(e.q05)}), el("td", {class: "num", text: fmt(e.q50)}), el("td", {class: "num", text: fmt(e.q95)}),
    el("td", {}, densityChart(e, {compact: true}), el("div", {class: "small muted", text: `${fmt(e.window[0], 3)} to ${fmt(e.window[1], 3)}`}))));
  filter.addEventListener("input", () => {
    const wanted = filter.value.trim().toLowerCase();
    rows.forEach(r => { r.hidden = wanted && !r.dataset.label.toLowerCase().includes(wanted); });
  });
  const table = el("table", {class: "list"},
    el("thead", {}, el("tr", {}, ["element", "mean", "sd", "5%", "50%", "95%", "posterior and prior"].map((h, i) => el("th", {class: i && i < 6 ? "num" : null, text: h})))),
    el("tbody", {}, rows));
  return el("div", {}, el("div", {class: "toolbar"}, filter, el("span", {class: "small muted", text: `${elements.length} elements`})), el("div", {class: "scroll-x"}, table));
}

function dateLabel(dates) {
  return (v, i) => dates[i] ?? "";
}

function seriesChart(dates, s) {
  const xs = dates.map((_, i) => i);
  const layers = [];
  if (s.median) {
    layers.push({kind: "band", lower: s.lower, upper: s.upper, cls: "band-2", name: "band", key: "box k2"});
  }
  layers.push({kind: "line", values: s.data, cls: "line-1", name: "data", key: "k1"});
  if (s.median) layers.push({kind: "line", values: s.median, cls: "line-2", name: "fitted (median)", key: "k2"});
  const notes = [];
  if (s.transform) notes.push(`transformed: ${s.transform}`);
  if (s.exact) notes.push("observed exactly");
  if (s.variable) notes.push(`feeds ${s.variable}`);
  return xyChart({xs, xLabel: dateLabel(dates), layers, title: s.series, caption: notes.join(" · ") || null});
}

function quantityChart(first, second = null) {
  const xs = first.axis;
  const layers = [];
  const add = (q, n, label) => {
    if (q.kind === "ranges") {
      layers.push({kind: "band", lower: q.lower, upper: q.upper, cls: `band-${n}`, name: `${label}set of posterior means`, key: `box k${n}`});
      layers.push({kind: "line", values: q.lower, cls: `line-${n}`}, {kind: "line", values: q.upper, cls: `line-${n}`});
      if (!second) {
        layers.push({kind: "line", values: q.robust_lower, cls: "line-2", name: "68% robust interval, lower", key: "k2"});
        layers.push({kind: "line", values: q.robust_upper, cls: "line-2", name: "68% robust interval, upper", key: "k2"});
      }
    } else {
      layers.push({kind: "band", lower: q.lower, upper: q.upper, cls: `band-${n}`, name: `${label}68% interval`, key: `box k${n}`});
      layers.push({kind: "line", values: q.mean, cls: `line-${n}`, name: `${label}mean`, key: `k${n}`});
    }
  };
  add(first, 1, second ? "first: " : "");
  if (second) add(second, 2, "second: ");
  const box = xyChart({xs, xLabel: v => String(v), layers, zero: true, title: null, caption: `${first.axis_label} · ${first.draws ?? "?"} draws`});
  box.prepend(el("div", {class: "title"}, math(first.latex, first.quantity)));
  return box;
}

const QUANTITY_LEGEND = () => legend([["", "k1", "mean (or, for ranges, the bounds of the set of posterior means)"], ["box", "k1", "68% interval (or the set of means)"],
                                      ["", "k2", "68% robust interval (ranges only)"]]);

// ---------------------------------------------------------------------------------------------------------------- the graph

function graphDrawing(graph, {onSelect = null, changed = new Set()} = {}) {
  const arrow = `arrow-${++uniqueId}`;
  const svg = sv("svg", {viewBox: `0 0 ${graph.width} ${graph.height}`, width: graph.width, height: graph.height, role: "group",
                         "aria-label": "the model's blocks and data series"});
  svg.append(sv("defs", {}, sv("marker", {id: arrow, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"},
                                     sv("path", {d: "M0,0 L10,5 L0,10 z", class: "arrow"}))));
  for (const e of graph.edges) {
    const label = e.variables.join(", ");
    const start = e.path[0][0];
    const d = `M${start[0]},${start[1]}` + e.path.map(([, b, c, end]) => ` C${b[0]},${b[1]} ${c[0]},${c[1]} ${end[0]},${end[1]}`).join("");
    svg.append(sv("g", {class: "edge"},
      sv("title", {text: `${e.source.split(":")[1]} → ${e.target.split(":")[1]}: ${label}${e.loop ? " (closes a loop)" : ""}`}),
      sv("path", {d, "marker-end": `url(#${arrow})`}),
      sv("text", {x: e.label_at[0], y: e.label_at[1] - 4, "text-anchor": "middle", text: shorten(label, 26)})));
  }
  const groups = new Map();
  for (const n of graph.nodes) {
    const isBlock = n.kind === "block";
    const detail = isBlock ? [n.template ? `for each ${n.template}` : "", n.code ? `code: ${n.code}` : `→ ${n.outputs.join(", ")}`].filter(Boolean).join(" · ")
                           : n.role === "observes" ? `observes ${n.variable}` : n.role === "feeds" ? `feeds ${n.variable}` : "used by restrictions";
    const cls = `node ${isBlock ? "block" : `series ${n.role}`}${changed.has(n.id) ? " changed" : ""}`;
    const g = sv("g", {class: cls, transform: `translate(${n.x},${n.y})`, tabindex: onSelect ? 0 : null, role: onSelect ? "button" : null,
                       "aria-label": `${isBlock ? "block" : "data series"} ${n.name}: ${detail}`},
      sv("title", {text: `${n.name}\n${detail}${changed.has(n.id) ? "\n(changed)" : ""}`}),
      sv("rect", {width: n.width, height: n.height, rx: isBlock ? 6 : n.height / 2}),
      sv("text", {class: "name", x: n.width / 2, y: isBlock ? 23 : 17, "text-anchor": "middle", text: shorten(n.name, Math.floor((n.width - 14) / 7.6))}),
      sv("text", {class: "detail", x: n.width / 2, y: isBlock ? 42 : 31, "text-anchor": "middle", text: shorten(detail, Math.floor((n.width - 12) / 6.1))}),
      changed.has(n.id) ? sv("text", {class: "tag", x: 4, y: -6, text: "changed"}) : null);
    if (onSelect) {
      g.addEventListener("click", () => onSelect(n.id));
      g.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(n.id); } });
    }
    groups.set(n.id, g);
    svg.append(g);
  }
  return {
    svg,
    select(id) { groups.forEach((g, key) => g.classList.toggle("selected", key === id)); },
  };
}

// ---------------------------------------------------------------------------------------------------------------- pages

function runsPage(main) {
  main.append(later(api("/api/runs"), data => {
    const page = el("div");
    page.append(el("h1", {text: "Runs"}),
      el("p", {class: "note", text: "Each row is one estimated version of a model. Open one to see its blocks, equations, priors against posteriors and fit; tick two to compare them."}));
    if (!data.runs.length) {
      page.append(el("p", {text: `This store holds no runs yet. Store one with: python -m mutatis run MODEL.yaml --data DATA --store ${data.store}`}));
      return page;
    }
    const chosen = [];
    const button = el("button", {class: "primary", disabled: true, text: "Compare the two ticked runs",
                                 onclick: () => { location.hash = `#/compare/${enc(chosen[0])}/${enc(chosen[1])}`; }});
    const rows = data.runs.map(r => {
      if (r.error) return el("tr", {}, el("td"), el("td", {colspan: 9, class: "error", text: `${r.key}: ${r.error}`}));
      const tick = el("input", {type: "checkbox", "aria-label": `choose ${r.key} for comparison`});
      tick.addEventListener("change", () => {
        if (tick.checked) chosen.push(r.key); else chosen.splice(chosen.indexOf(r.key), 1);
        button.disabled = chosen.length !== 2;
      });
      return el("tr", {},
        el("td", {}, tick),
        el("td", {}, el("a", {href: `#/run/${enc(r.key)}`, text: r.model}), el("div", {class: "small muted", text: r.key})),
        el("td", {class: "small"}, r.edits && r.edits.length ? editList(r.edits) : el("span", {class: "muted", text: "as specified"})),
        el("td", {text: r.method}), el("td", {text: r.route}),
        el("td", {text: r.data_dates ? r.data_dates.join(" to ") : "–"}),
        el("td", {class: "num", text: r.draws ?? "–"}),
        el("td", {class: "num", text: fmt(r.log_marginal_likelihood)}),
        el("td", {}, r.converged === false ? el("span", {class: "error", text: "not converged"}) : null,
                     r.has_data ? null : el("span", {class: "chip", text: "data not stored"})),
        el("td", {class: "small muted", text: r.stored ? new Date(r.stored * 1000).toLocaleString() : ""}));
    });
    const heads = ["", "model", "edits", "method", "route", "data", "draws", "log marginal likelihood", "", "stored"];
    page.append(el("div", {class: "toolbar"}, button, el("span", {class: "small muted", text: "Tick two runs, then compare."})),
      el("div", {class: "scroll-x"}, el("table", {class: "list"}, el("thead", {}, el("tr", {}, heads.map((h, i) => el("th", {class: i === 6 || i === 7 ? "num" : null, text: h})))),
                                       el("tbody", {}, rows))));
    return page;
  }));
}

function runHeader(m) {
  const run = m.run || {};
  const d = m.diagnostics || {};
  const settings = Object.entries(run.settings || {}).filter(([k]) => !["route", "start", "method", "report"].includes(k)).map(([k, v]) => `${k} ${JSON.stringify(v)}`).join(", ");
  const head = el("div", {class: "card"}, el("h1", {text: m.name}),
    facts([["run", el("code", {text: m.key})], ["method", `${run.method} (route: ${run.route}; start: ${(run.settings || {}).start ?? "–"})`], ["settings", settings],
           ["data", run.data_dates ? run.data_dates.join(" to ") : null], ["seed", run.seed], ["edits from the base model", editList(run.edits)],
           ["log marginal likelihood", d.log_marginal_likelihood === null || d.log_marginal_likelihood === undefined ? "not defined (an improper prior), or not computed" : fmt(d.log_marginal_likelihood)],
           ["draws", d.draws], ["share of draws where the restrictions can hold", d.plausibility === undefined ? null : fmt(d.plausibility)],
           ["peak search", d.peak_search ? JSON.stringify(d.peak_search) : null]]));
  if (d.converged === false) head.append(el("div", {class: "warning", text: `Not converged: the results are suspect. ${d.status || ""}`}));
  return head;
}

function runPage(main, key, tab, selected) {
  const base = `/api/runs/${enc(key)}`;
  const model = api(`${base}/model`);
  main.append(el("p", {class: "small"}, el("a", {href: "#/", text: "← all runs"})), later(model, runHeader));
  const tabs = [["model", "Model"], ["parameters", "Priors and posteriors"], ["fit", "Data and fit"], ["quantities", "Reported quantities"]];
  main.append(el("nav", {class: "tabs"}, tabs.map(([id, label]) => el("a", {href: `#/run/${enc(key)}/${id}`, class: id === tab ? "on" : null, text: label}))));
  if (tab === "parameters") main.append(later(api(`${base}/parameters`), parametersTab, "Computing priors and posteriors from the stored draws…"));
  else if (tab === "fit") main.append(later(api(`${base}/fit`), fitTab, "Computing the model's predictions at posterior draws…"));
  else if (tab === "quantities") main.append(later(api(`${base}/quantities`), quantitiesTab));
  else main.append(later(model, m => modelTab(m, key, selected)));
}

function modelTab(m, key, selected) {
  const nodes = new Map(m.graph.nodes.map(n => [n.id, n]));
  const panel = el("div", {class: "panel card"});
  const parameters = () => api(`/api/runs/${enc(key)}/parameters`);
  const fit = () => api(`/api/runs/${enc(key)}/fit`);
  let current = nodes.has(selected) ? selected : null;
  const drawing = graphDrawing(m.graph, {onSelect: id => {
    current = current === id ? null : id;
    history.replaceState(null, "", `#/run/${enc(key)}/model${current ? `/${enc(current)}` : ""}`);
    show();
  }});
  function show() {
    drawing.select(current);
    panel.replaceChildren(current ? nodePanel(m, nodes.get(current), parameters, fit) : overviewPanel(m, parameters));
  }
  show();
  const layout = el("div", {class: "model-layout"},
    el("div", {}, el("div", {class: "graph-box"}, drawing.svg),
       el("p", {class: "note", text: "Square boxes are blocks; rounded boxes are data series, outlined in blue when the model must explain them (the second line says what each one observes or feeds). Each arrow is labelled with the variables it carries; arcs below the boxes close a loop. Click a box for its equations, parameters and data."})),
    panel);
  const below = el("div");
  below.append(el("h2", {text: "The specification as stored"}), el("details", {}, el("summary", {text: "YAML"}), el("pre", {text: m.yaml})));
  return el("div", {}, layout, below);
}

function ownedPanels(parameters, owner) {
  return later(parameters(), p => {
    const mine = p.parameters.filter(entry => entry.owner === owner);
    if (!mine.length) return el("span");
    return el("div", {}, p.note ? el("p", {class: "note", text: p.note}) : null, DENSITY_LEGEND(), mine.map(parameterBlock));
  }, "Computing priors and posteriors…");
}

function parameterLines(list) {
  return el("div", {class: "equations"}, list.map(p => el("div", {}, p.latex ? math(p.latex, `${p.name}: ${p.prior ?? JSON.stringify(p.fixed)}`, true)
                                                                         : el("code", {text: `${p.name}: ${p.prior ?? JSON.stringify(p.fixed)}`}),
                                                          p.separate ? parameterLines(p.separate) : null)));
}

function overviewPanel(m, parameters) {
  const out = el("div", {}, el("h2", {text: "The model as a whole"}),
    el("p", {class: "note", text: "Click a block or a data series in the graph to see it here."}),
    facts([["time", m.time], ["expectations", m.expectations], ["blocks", String(m.blocks.length)], ["data series", String(m.data.length)]]));
  if (Object.keys(m.sets).length) out.append(el("h3", {text: "Sets"}), facts(Object.entries(m.sets).map(([k, v]) => [k, v.join(", ")])));
  if (m.calendar.length) out.append(el("h3", {text: "Calendar variables"}), el("div", {class: "equations"}, m.calendar.map(c => math(c.latex, `${c.name} = ${c.text}`, true))));
  if (m.parameters.length) out.append(el("h3", {text: "Parameters shared by every block"}), parameterLines(m.parameters), ownedPanels(parameters, null));
  if (m.restrictions.length) {
    out.append(el("h3", {text: "Restrictions"}), el("div", {class: "equations"},
      m.restrictions.map(r => el("div", {}, el("span", {class: "small muted", text: r.name}), math(r.latex, `${r.quantity} ${r.require}`, true)))));
  }
  return out;
}

function blockPanel(b) {
  const out = el("div");
  if (b.code) out.append(el("p", {text: `Written as code, registered as ${b.code}: its insides cannot be shown.`}));
  if (b.equations.length) out.append(el("h3", {text: "Equations"}), el("div", {class: "equations"}, b.equations.map(e => math(e.latex, e.text, true))));
  if (b.definitions.length) out.append(el("h3", {text: "Definitions"}), el("div", {class: "equations"}, b.definitions.map(e => math(e.latex, `${e.name} = ${e.text}`, true))));
  if (b.shocks.length) out.append(el("h3", {text: "Shocks"}), el("div", {class: "equations"}, b.shocks.map(s => math(s.latex, `${s.name}: ${s.distribution}`, true))));
  if (b.initial.length) out.append(el("h3", {text: "Starting values"}), el("div", {class: "equations"}, b.initial.map(s => math(s.latex, `${s.variable}: ${s.text}`, true))));
  if (b.steady_state.length) out.append(el("h3", {text: "Steady state: where the search starts"}), el("div", {class: "equations"}, b.steady_state.map(s => math(s.latex, `${s.variable} = ${s.text}`, true))));
  if (b.parameters.length) out.append(el("h3", {text: "Parameters"}), parameterLines(b.parameters));
  return out;
}

function nodePanel(m, node, parameters, fit) {
  const out = el("div");
  if (node.kind === "block") {
    const b = m.blocks.find(x => x.name === node.name);
    out.append(el("h2", {}, b.name, b.template ? el("span", {class: "chip", text: `one for each ${b.template.index} in ${b.template.set}: ${b.template.elements.join(", ")}`}) : null));
    out.append(facts([["takes in", b.inputs.join(", ") || "nothing"], ["gives out", b.outputs.join(", ")]]));
    out.append(blockPanel(b));
    for (const o of b.overrides) out.append(el("h3", {text: `Instance ${o.element}: replaced by a block of its own`}), blockPanel(o));
    if (b.parameters.length) out.append(ownedPanels(parameters, b.name));
    return out;
  }
  const s = m.data.find(x => x.series === node.name);
  out.append(el("h2", {}, s.series, el("span", {class: "chip", text: s.role === "observes" ? "observed" : s.role === "feeds" ? "feeds the model" : "used by restrictions"})));
  if (s.observes !== null) {
    out.append(el("h3", {text: "What it observes"}), math(s.observes_latex, `${s.series} = ${s.observes}`, true),
      el("p", {}, s.error ? ["measurement error ", math(s.error_latex, s.error)] : "observed exactly: no measurement error"));
  }
  if (s.feeds !== null) out.append(el("h3", {text: "What it feeds"}), math(s.feeds_latex, `${s.feeds} = ${s.series}`, true));
  if (s.transform) out.append(el("p", {}, "used as ", math(s.transform_latex, s.transform), el("span", {class: "muted", text: ", where x is the series as published"})));
  if (s.parameters.length) out.append(el("h3", {text: "Parameters"}), parameterLines(s.parameters), ownedPanels(parameters, s.series));
  out.append(el("h3", {text: "Data and fit"}), later(fit(), f => {
    if (f.note) return el("p", {class: "note", text: f.note});
    const found = [...f.observed, ...f.fed, ...f.auxiliary].find(x => x.series === s.series);
    if (!found) return el("p", {class: "note", text: "This series is not part of the data the run used."});
    return el("div", {}, found.median ? fitLegend() : null, seriesChart(f.dates, found), found.median ? el("p", {class: "note", text: fitNote(f)}) : null);
  }, "Computing the model's predictions at posterior draws…"));
  return out;
}

function fitLegend() {
  return legend([["", "k1", "data"], ["", "k2", "fitted: median prediction"], ["box", "k2", "90% band across posterior draws"]]);
}

function fitNote(f) {
  const left = f.draws_tried - f.draws;
  return `Fitted is ${f.basis}, at ${f.draws} posterior draws; the band covers ${Math.round(100 * (f.band[1] - f.band[0]))}% of them.` +
         (left ? ` ${left} draws gave no usable solution and are left out.` : "");
}

function parametersTab(p) {
  const out = el("div");
  if (p.note) out.append(el("p", {class: "warning", text: p.note}));
  out.append(el("p", {class: "note", text: "For each parameter: its prior, and its posterior as a histogram of the stored draws, on one axis. Matrices with many elements are shown as a table."}),
             DENSITY_LEGEND());
  const owners = [];
  for (const entry of p.parameters) {
    const owner = entry.owner ?? "";
    if (!owners.includes(owner)) owners.push(owner);
  }
  for (const owner of owners) {
    out.append(el("h2", {class: "owner", text: owner ? owner : "Shared by every block"}));
    p.parameters.filter(e => (e.owner ?? "") === owner).forEach(e => out.append(parameterBlock(e)));
  }
  return out;
}

function fitTab(f) {
  if (f.note) return el("p", {class: "warning", text: f.note});
  const out = el("div", {}, el("p", {class: "note", text: fitNote(f)}), fitLegend(),
    el("div", {class: "wide-charts"}, f.observed.map(s => seriesChart(f.dates, s))));
  if (f.fed.length) out.append(el("h2", {text: "Series that feed the model"}), el("p", {class: "note", text: "Taken as given: the model does not explain them."}),
                               el("div", {class: "wide-charts"}, f.fed.map(s => seriesChart(f.dates, s))));
  if (f.auxiliary.length) out.append(el("h2", {text: "Series used only by restrictions"}), el("div", {class: "wide-charts"}, f.auxiliary.map(s => seriesChart(f.dates, s))));
  return out;
}

function quantitiesTab(q) {
  if (!q.quantities.length) return el("p", {class: "note", text: "This run reports no quantities. List some when running (--report) to have them summarised here."});
  return el("div", {}, el("p", {class: "note", text: "Each reported quantity, element by element: the posterior mean and 68% interval; or, where the data cannot decide and no prior was imposed, the set of posterior means and the 68% robust interval."}),
            QUANTITY_LEGEND(), el("div", {class: "wide-charts"}, q.quantities.map(entry => quantityChart(entry))));
}

function intervalChart(a, b) {
  const width = 220, height = 30;
  const values = finite([a && a.q05, a && a.q95, b && b.q05, b && b.q95, a && a.mean, b && b.mean]);
  const svg = sv("svg", {viewBox: `0 0 ${width} ${height}`, class: "spark", role: "img", "aria-label": "90% intervals and means, first and second run"});
  if (!values.length) return svg;
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi === lo) { lo -= 1; hi += 1; }
  const x = v => 6 + (v - lo) / (hi - lo) * (width - 12);
  [[a, 1, 10], [b, 2, 21]].forEach(([e, n, y]) => {
    if (!e || e.mean === undefined) return;
    svg.append(sv("line", {x1: x(e.q05), x2: x(e.q95), y1: y, y2: y, class: `line-${n}`}), sv("circle", {cx: x(e.mean), cy: y, r: 4, class: `dot-${n}`}));
  });
  return svg;
}

function cell(entry) {
  if (!entry) return "absent";
  if (entry.no_prior) return "no prior: ranges";
  if (entry.fixed !== undefined) return `fixed ${shorten(JSON.stringify(entry.fixed), 40)}`;
  return `${fmt(entry.mean)} (${fmt(entry.sd)})`;
}

function comparePage(main, first, second) {
  main.append(el("p", {class: "small"}, el("a", {href: "#/", text: "← all runs"})));
  main.append(later(api(`/api/compare/${enc(first)}/${enc(second)}`), c => {
    const out = el("div", {}, el("h1", {text: "Two runs side by side"}));
    const card = (r, n, label) => el("div", {class: "card"}, el("h3", {}, el("span", {class: `key k${n}`}), ` ${label}: `, el("a", {href: `#/run/${enc(r.key)}`, text: r.model})),
      facts([["run", el("code", {text: r.key})], ["method", `${r.method} (route: ${r.route})`], ["data", r.data_dates ? r.data_dates.join(" to ") : null], ["edits from the base model", editList(r.edits)]]));
    out.append(el("div", {class: "versus"}, card(c.first, 1, "first"), card(c.second, 2, "second")));
    out.append(el("h2", {text: "What changed in the specification"}));
    if (c.note) out.append(el("p", {class: "warning", text: c.note}));
    else if (!c.edits.length) out.append(el("p", {text: "The specifications are the same."}));
    else out.append(el("p", {text: "Edits that turn the first specification into the second:"}), editList(c.edits));
    if (!c.same_data) out.append(el("p", {class: "warning", text: "The two runs used different data."}));
    const changedIds = new Set([...c.changed.blocks.map(b => `block:${b}`), ...c.changed.data.map(s => `data:${s}`)]);
    const other = [c.changed.expectations ? "the expectations setting" : null, c.changed.sets.length ? `sets: ${c.changed.sets.join(", ")}` : null,
                   c.changed.model_parameters.length ? `shared parameters: ${c.changed.model_parameters.join(", ")}` : null,
                   c.changed.restrictions.length ? `restrictions: ${c.changed.restrictions.join(", ")}` : null].filter(Boolean);
    if (other.length) out.append(el("p", {}, "Also changed: ", other.join("; "), "."));
    const drawing = graphDrawing(c.graph, {changed: changedIds, onSelect: id => { location.hash = `#/run/${enc(c.second.key)}/model/${enc(id)}`; }});
    out.append(el("div", {class: "graph-box"}, drawing.svg),
      el("p", {class: "note", text: changedIds.size ? "The second version's graph; parts that differ from the first are outlined in orange. Click a box to open it in the second run." : "The second version's graph. No block or data series differs."}));
    const [la, lb] = c.log_marginal_likelihood;
    out.append(el("h2", {text: "Which version the data favour"}));
    if (la === null || lb === null) out.append(el("p", {text: "The log marginal likelihood is not defined for at least one run (an improper prior), or was not computed."}));
    else out.append(el("p", {text: `Log marginal likelihood: ${fmt(la)} (first) against ${fmt(lb)} (second). The data favour the ${lb > la ? "second" : "first"} by ${fmt(Math.abs(lb - la))}. This depends on how wide the priors are.`}));
    out.append(el("h2", {text: "Parameters"}), legend([["", "k1", "first: 90% interval and mean"], ["", "k2", "second"]]));
    const filter = el("input", {type: "search", placeholder: "filter parameters", "aria-label": "filter parameters"});
    const rows = c.parameters.map(p => {
      const change = p.first && p.second && p.first.mean !== undefined && p.second.mean !== undefined ? p.second.mean - p.first.mean : null;
      return el("tr", {"data-label": p.label}, el("td", {}, el("code", {text: p.label})), el("td", {class: "num", text: cell(p.first)}), el("td", {class: "num", text: cell(p.second)}),
                el("td", {class: "num", text: change === null ? "" : (change > 0 ? "+" : "") + fmt(change)}), el("td", {}, intervalChart(p.first, p.second)));
    });
    filter.addEventListener("input", () => {
      const wanted = filter.value.trim().toLowerCase();
      rows.forEach(r => { r.hidden = wanted && !r.dataset.label.toLowerCase().includes(wanted); });
    });
    out.append(el("div", {class: "toolbar"}, filter), el("div", {class: "scroll-x"}, el("table", {class: "list"},
      el("thead", {}, el("tr", {}, ["parameter", "first: mean (sd)", "second: mean (sd)", "change in mean", "intervals"].map((h, i) => el("th", {class: i && i < 4 ? "num" : null, text: h})))),
      el("tbody", {}, rows))));
    out.append(el("h2", {text: "Reported quantities in both"}));
    if (!c.quantities.length) out.append(el("p", {class: "note", text: "The two runs report no quantity in common."}));
    else out.append(legend([["", "k1", "first"], ["", "k2", "second"]]), el("div", {class: "wide-charts"}, c.quantities.map(q => quantityChart(q.first, q.second))));
    return out;
  }));
}

function render() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const main = document.getElementById("main");
  main.replaceChildren();
  if (parts[0] === "run" && parts[1]) runPage(main, parts[1], parts[2] || "model", parts[3]);
  else if (parts[0] === "compare" && parts[1] && parts[2]) comparePage(main, parts[1], parts[2]);
  else runsPage(main);
}

if (typeof document !== "undefined" && document.getElementById("main")) {
  window.addEventListener("hashchange", render);
  api("/api/runs").then(data => { document.getElementById("store").textContent = data.store; }).catch(() => {});
  render();
}
if (typeof module !== "undefined") module.exports = {fmt, niceTicks, shorten, pathThrough, bandPath};
