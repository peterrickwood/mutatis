"""A specification as a graph of blocks, laid out for drawing.

Nodes are the blocks (a template block is one node standing for all its instances) and the data series. Edges:

- from a block that outputs a variable to each block that takes it as an input;
- from a series that feeds an outside variable to each block that takes that variable as an input;
- from a block to each series that observes one of its outputs.

Several variables between the same two nodes make one edge.

The layout is layered, left to right (after Sugiyama, Tagawa and Toda). Series that feed the model are in the first column and
observed series in the last. Each block is one column to the right of the furthest block it takes an input from, after setting
aside edges that close a loop (households and firms that feed each other, say); those are drawn as arcs below the boxes. An edge
that spans several columns passes through a slot of its own in each column between, so no edge crosses a box. Within a column,
boxes and slots are ordered to keep edges short, ties kept in specification order, so the same specification always gives the
same picture. Each edge is a chain of cubic curves; positions are in the units of an SVG drawing.
"""

from mutatis.spec.checks import _expand, _produced, _show, instances
from mutatis.spec.expr import Index, Name, parse_expression, walk
from mutatis.spec.model import Model

BLOCK_SIZE = (176.0, 58.0)
SERIES_SIZE = (150.0, 40.0)
COLUMN_GAP, ROW_GAP, MARGIN = 96.0, 34.0, 24.0
SLOT_HEIGHT, SLOT_GAP = 10.0, 10.0      # a passing slot, and its distance from its neighbours
LOOP_CLEARANCE = 22.0
SWEEPS = 4


def block_graph(model: Model) -> dict:
    nodes, edges = _nodes_and_edges(model)
    return _layout(nodes, edges)


def _nodes_and_edges(model: Model) -> tuple[list[dict], dict[tuple[str, str], list[str]]]:
    label_to_block = {label: name for name, block in model.blocks.items() for label, _, _ in instances(name, block, model)}
    produced = {variable: label_to_block[label] for variable, label in _produced(model)[0].items()}
    fed = {variable: series for series, link in model.data.items() if link.feeds for variable in _expand(link.feeds, model, {})}
    nodes = []
    for name, block in model.blocks.items():
        labels = [label for label, _, _ in instances(name, block, model)]
        nodes.append({"id": f"block:{name}", "kind": "block", "name": name, "outputs": list(block.outputs), "inputs": list(block.inputs),
                      "template": f"{block.over[0]} in {block.over[1]}" if block.over else None, "instances": labels if block.over else [],
                      "overrides": sorted(block.overrides), "code": block.code})
    for series, link in model.data.items():
        nodes.append({"id": f"data:{series}", "kind": "series", "name": series, "role": link.role,
                      "variable": link.observes if link.observes is not None else link.feeds})

    edges: dict[tuple[str, str], list[str]] = {}

    def connect(source: str, target: str, label: str):
        if source != target:
            carried = edges.setdefault((source, target), [])
            if label not in carried:
                carried.append(label)

    for name, block in model.blocks.items():
        for _, instance, substitute in instances(name, block, model):
            for reference in instance.inputs:
                for variable in sorted(_expand(reference, model, substitute)):
                    if variable in produced:
                        connect(f"block:{produced[variable]}", f"block:{name}", variable[0])
                    elif variable in fed:
                        connect(f"data:{fed[variable]}", f"block:{name}", _show(variable))
    elements = {str(e) for values in model.sets.values() for e in values}
    for series, link in model.data.items():
        if link.observes is None:
            continue
        for variable in _observed_variables(parse_expression(link.observes), produced, elements):
            connect(f"block:{produced[variable]}", f"data:{series}", _show(variable))
    return nodes, edges


def _observed_variables(expr, produced: dict, elements: set[str]) -> list[tuple]:
    """The produced variables an observation expression refers to: one element where it names one (y[gdp]), else every element."""
    found, inside = [], set()
    for node in walk(expr):
        if isinstance(node, Index) and isinstance(node.base, Name) and all(isinstance(i, Name) and i.id in elements for i in node.items):
            variable = (node.base.id, tuple(i.id for i in node.items))
            if variable in produced:
                found.append(variable)
                inside.add(id(node.base))
        elif isinstance(node, Name) and id(node) not in inside:
            found += sorted(v for v in produced if v[0] == node.id)
    return list(dict.fromkeys(found))


# Layout -----------------------------------------------------------------------------------------------------------------------

def _layout(nodes: list[dict], edges: dict[tuple[str, str], list[str]]) -> dict:
    blocks = [n["id"] for n in nodes if n["kind"] == "block"]
    feeding = [n["id"] for n in nodes if n["kind"] == "series" and n["role"] == "feeds"]
    others = [n["id"] for n in nodes if n["kind"] == "series" and n["role"] != "feeds"]
    between = [(a, b) for a, b in edges if a in blocks and b in blocks]
    order = _loop_free_order(blocks, between)
    rank = {node: i for i, node in enumerate(order)}
    loops = {(a, b) for a, b in between if rank[a] > rank[b]}
    layer = {}
    for node in order:
        before = [layer[a] for a, b in between if b == node and (a, b) not in loops]
        layer[node] = 1 + max(before) if before else 0
    first = 1 if feeding else 0
    column = {**{n: 0 for n in feeding}, **{n: first + layer[n] for n in blocks}}
    last = max(column.values(), default=-1) + 1
    column.update({n: last for n in others})
    columns = [[n["id"] for n in nodes if column[n["id"]] == c] for c in range(last + 1)]
    columns = [c for c in columns if c]
    column = {n: i for i, members in enumerate(columns) for n in members}
    loops |= {pair for pair in edges if column[pair[0]] >= column[pair[1]]}

    # An edge that spans several columns passes through a slot of its own in each column between, so it never crosses a box.
    chains = {}
    for a, b in edges:
        if (a, b) not in loops:
            chains[(a, b)] = [f"pass:{a}>{b}:{c}" for c in range(column[a] + 1, column[b])]
            for c, slot in zip(range(column[a] + 1, column[b]), chains[(a, b)]):
                columns[c].append(slot)
                column[slot] = c
    hops = [hop for (a, b), chain in chains.items() for hop in zip([a, *chain], [*chain, b])]
    columns = _order_columns(columns, hops + sorted(loops))

    size = {n["id"]: BLOCK_SIZE if n["kind"] == "block" else SERIES_SIZE for n in nodes}
    width_of = [max(size[n][0] for n in members if n in size) for members in columns]
    extent = lambda c, item: size.get(item, (width_of[c], SLOT_HEIGHT))
    gap = lambda a, b: ROW_GAP if a in size and b in size else SLOT_GAP
    heights = [sum(extent(c, i)[1] for i in members) + sum(gap(a, b) for a, b in zip(members, members[1:])) for c, members in enumerate(columns)]
    placed = {}
    x = MARGIN
    for c, members in enumerate(columns):
        y = MARGIN + (max(heights) - heights[c]) / 2
        for i, item in enumerate(members):
            y += gap(members[i - 1], item) if i else 0.0
            w, h = extent(c, item)
            placed[item] = {"x": x + (width_of[c] - w) / 2, "y": y, "width": w, "height": h, "column": c}
            y += h
        x += width_of[c] + COLUMN_GAP

    ports = _ports(edges, chains, loops, placed)
    drawn = []
    for k, (a, b) in enumerate(sorted(edges, key=lambda pair: (pair not in loops, abs(column[pair[0]] - column[pair[1]])))):
        path = _loop(a, b, placed, ports, k) if (a, b) in loops else _route(a, b, chains[(a, b)], placed, ports)
        drawn.append({"source": a, "target": b, "variables": edges[(a, b)], "loop": (a, b) in loops, "path": [[list(p) for p in s] for s in path],
                      "label_at": list(_middle(path, (a, b) in loops))})
    drawn.sort(key=lambda e: list(edges).index((e["source"], e["target"])))
    points = [p for e in drawn for segment in e["path"] for p in segment] + [(p["x"] + p["width"], p["y"] + p["height"]) for p in placed.values()]
    return {"nodes": [{**n, **placed[n["id"]]} for n in nodes], "edges": drawn,
            "width": max(px for px, _ in points) + MARGIN, "height": max(py for _, py in points) + MARGIN}


def _ports(edges, chains, loops, placed) -> dict:
    """Where each edge meets each box: edges leave from the right side and arrive on the left, spread along it in the order of
    where they go; edges that close a loop meet the bottom, spread along it."""
    centre = lambda item: placed[item]["y"] + placed[item]["height"] / 2
    ports = {}
    for node, box in placed.items():
        if node.startswith("pass:"):
            continue
        leaving = sorted((pair for pair in chains if pair[0] == node), key=lambda pair: centre((chains[pair] or [pair[1]])[0]))
        arriving = sorted((pair for pair in chains if pair[1] == node), key=lambda pair: centre((chains[pair] or [pair[0]])[-1]))
        for side, group in (("out", leaving), ("in", arriving)):
            for i, pair in enumerate(group):
                ports[(side, pair)] = box["y"] + box["height"] * (i + 1) / (len(group) + 1)
        below = sorted((pair for pair in loops if node in pair), key=lambda pair: placed[pair[1] if pair[0] == node else pair[0]]["x"])
        for i, pair in enumerate(below):
            ports[("below", node, pair)] = box["x"] + box["width"] * (i + 1) / (len(below) + 1)
    return ports


def _route(a: str, b: str, chain: list[str], placed: dict, ports: dict) -> list:
    """Cubic curves from the source's right side, straight across each passing slot, to the target's left side."""
    points = [(placed[a]["x"] + placed[a]["width"], ports[("out", (a, b))])]
    for slot in chain:
        box = placed[slot]
        points += [(box["x"], box["y"] + box["height"] / 2), (box["x"] + box["width"], box["y"] + box["height"] / 2)]
    points.append((placed[b]["x"], ports[("in", (a, b))]))
    path = []
    for i, (p, q) in enumerate(zip(points, points[1:])):
        if i % 2:   # across a slot
            path.append((p, (p[0] + (q[0] - p[0]) / 3, p[1]), (p[0] + 2 * (q[0] - p[0]) / 3, q[1]), q))
        else:
            reach = max(30.0, (q[0] - p[0]) / 2)
            path.append((p, (p[0] + reach, p[1]), (q[0] - reach, q[1]), q))
    return path


def _loop(a: str, b: str, placed: dict, ports: dict, k: int) -> list:
    """An arc from the source's bottom to the target's bottom whose lowest point clears every box in the columns it spans."""
    start = (ports[("below", a, (a, b))], placed[a]["y"] + placed[a]["height"])
    end = (ports[("below", b, (a, b))], placed[b]["y"] + placed[b]["height"])
    first, last = sorted((placed[a]["column"], placed[b]["column"]))
    floor = max(box["y"] + box["height"] for box in placed.values() if first <= box["column"] <= last)
    lane = floor + LOOP_CLEARANCE * (1 + k)
    depth = (lane - (start[1] + end[1]) / 8) / 0.75   # the curve's middle point then lies on the lane
    return [(start, (start[0], depth), (end[0], depth), end)]


def _middle(path: list, loop: bool) -> tuple[float, float]:
    """Where an edge's label goes: the middle of its middle curve between columns (curves across a slot do not count)."""
    between = path if loop else path[::2]
    p = between[len(between) // 2]
    return tuple((p[0][i] + 3 * p[1][i] + 3 * p[2][i] + p[3][i]) / 8 for i in range(2))


def _loop_free_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """An order of the nodes in which few edges point backwards (Eades, Lin and Smyth's greedy rule): repeatedly take nodes with
    no edges out to the end, nodes with no edges in to the front, and otherwise the node with most more edges out than in. Ties go
    to the earlier node in the given order."""
    remaining = list(nodes)
    front, back = [], []

    def degree(node, direction):
        return sum(1 for a, b in edges if (a if direction == "out" else b) == node and a in remaining and b in remaining)

    while remaining:
        moved = True
        while moved:
            moved = False
            for node in remaining:
                if degree(node, "out") == 0:
                    back.insert(0, node)
                    remaining.remove(node)
                    moved = True
                    break
                if degree(node, "in") == 0:
                    front.append(node)
                    remaining.remove(node)
                    moved = True
                    break
        if remaining:
            chosen = max(remaining, key=lambda n: (degree(n, "out") - degree(n, "in"), -remaining.index(n)))
            front.append(chosen)
            remaining.remove(chosen)
    return front + back


def _order_columns(columns: list[list[str]], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Reorder each column so a node sits near the average height of its neighbours in the columns on one side, sweeping left to
    right then right to left a few times. A node with no neighbours on that side keeps its place."""
    columns = [list(c) for c in columns]
    column = {n: i for i, members in enumerate(columns) for n in members}
    neighbours = {n: set() for n in column}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)

    def height(node):
        members = columns[column[node]]
        return members.index(node) - (len(members) - 1) / 2

    for _ in range(SWEEPS):
        for direction in (1, -1):
            sequence = range(1, len(columns)) if direction == 1 else range(len(columns) - 2, -1, -1)
            for c in sequence:
                def key(node, c=c):
                    side = [height(m) for m in neighbours[node] if (column[m] - c) * direction < 0]
                    return sum(side) / len(side) if side else height(node)
                columns[c] = sorted(columns[c], key=key)   # sorted is stable: ties keep their order
    return columns
