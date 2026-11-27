"""The graph of blocks: which nodes and edges a specification gives, and the properties of its layout, for every toy and
demonstration."""

import numpy as np
import pytest

from mutatis.spec.yaml_io import load_model
from mutatis.web.graph import block_graph
from tests.spec.fixtures import MODEL_FILES, REPO, toy

ALL = MODEL_FILES + sorted((REPO / "demos").glob("*/model.yaml"))


def _edges(graph):
    return {(e["source"], e["target"]): e["variables"] for e in graph["edges"]}


def test_forward_looking_toy():
    graph = block_graph(toy("forward_looking"))
    assert [n["id"] for n in graph["nodes"]] == ["block:demand", "block:price", "data:x_obs", "data:p_obs"]
    assert _edges(graph) == {("block:demand", "block:price"): ["x"], ("block:demand", "data:x_obs"): ["x"], ("block:price", "data:p_obs"): ["p"]}
    assert not any(e["loop"] for e in graph["edges"])


def test_templates_are_one_node_and_series_attach_to_the_variables_they_observe_or_feed():
    graph = block_graph(toy("sectors"))
    capital = next(n for n in graph["nodes"] if n["id"] == "block:capital")
    assert capital["template"] == "j in sectors" and capital["instances"] == ["capital[n]", "capital[m]"]
    assert _edges(graph) == {("block:rate", "block:investment"): ["r"], ("block:investment", "block:capital"): ["invest"],
                             ("block:capital", "block:total"): ["k"], ("block:total", "data:total_capital"): ["K"], ("block:rate", "data:rate"): ["r"]}
    fed = block_graph(toy("error_correction"))
    assert _edges(fed) == {("data:income", "block:spending"): ["income"], ("block:spending", "data:spend"): ["spend"]}
    assert _edges(block_graph(toy("two_variable"))) == {("block:dynamics", "data:a"): ["y[a]"], ("block:dynamics", "data:b"): ["y[b]"]}


def test_a_demonstration_with_outside_variables_and_a_series_used_only_by_restrictions():
    graph = block_graph(load_model((REPO / "demos" / "read2022" / "model.yaml").read_text()))
    edges = _edges(graph)
    assert edges[("data:TOT", "block:economy")] == ["x[tot]"] and edges[("block:economy", "data:GDP")] == ["y[gdp]"]
    assert not any("data:PROXY" in pair for pair in edges)
    column = {n["id"]: n["column"] for n in graph["nodes"]}
    assert column["data:TOT"] == 0 and column["block:economy"] == 1 and column["data:GDP"] == column["data:PROXY"] == 2


def test_loops_are_set_aside_and_drawn_as_arcs():
    text = """
model: ring
blocks:
  a: {inputs: [z], outputs: [x], equations: ["x = 0.5 * z(-1)"]}
  b: {inputs: [x], outputs: [y], equations: ["y = x"]}
  c: {inputs: [y], outputs: [z], equations: ["z = y"]}
data:
  obs: {observes: x, error: "normal(0, 1)"}
"""
    graph = block_graph(load_model(text))
    loops = [e for e in graph["edges"] if e["loop"]]
    assert len(loops) == 1                                  # one edge is enough to open a three-block ring
    nodes = {n["id"]: n for n in graph["nodes"]}
    a, b = nodes[loops[0]["source"]], nodes[loops[0]["target"]]
    assert a["column"] > b["column"]
    (start, first, second, end), = loops[0]["path"]
    assert start[1] == a["y"] + a["height"] and end[1] == b["y"] + b["height"]   # from the bottom of one box to the bottom of the other
    middle = (start[1] + 3 * first[1] + 3 * second[1] + end[1]) / 8
    assert middle > max(n["y"] + n["height"] for n in nodes.values())           # below every box in the columns it spans


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_layout_properties(path):
    model = load_model(path.read_text())
    graph = block_graph(model)
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert len(nodes) == len(model.blocks) + len(model.data)
    boxes = list(nodes.values())
    for i, p in enumerate(boxes):                            # no two boxes overlap, and all lie inside the drawing
        assert 0 <= p["x"] and p["x"] + p["width"] <= graph["width"] and 0 <= p["y"] and p["y"] + p["height"] <= graph["height"]
        for q in boxes[i + 1:]:
            assert p["x"] + p["width"] <= q["x"] or q["x"] + q["width"] <= p["x"] or p["y"] + p["height"] <= q["y"] or q["y"] + q["height"] <= p["y"], (p["id"], q["id"])
    last = max(n["column"] for n in boxes)
    for n in boxes:
        if n["kind"] == "series":
            assert n["column"] == (0 if n["role"] == "feeds" else last)
    for e in graph["edges"]:
        a, b = nodes[e["source"]], nodes[e["target"]]
        start, end = e["path"][0][0], e["path"][-1][-1]
        assert all(segment[-1] == following[0] for segment, following in zip(e["path"], e["path"][1:]))   # one unbroken line
        assert all(0 <= x <= graph["width"] and 0 <= y <= graph["height"] for segment in e["path"] for x, y in segment)
        if e["loop"]:
            continue
        assert a["column"] < b["column"]                     # left to right, from the source's right side to the target's left side
        assert start[0] == a["x"] + a["width"] and a["y"] < start[1] < a["y"] + a["height"]
        assert end[0] == b["x"] and b["y"] < end[1] < b["y"] + b["height"]
        for segment in e["path"]:                            # and it crosses no box on the way
            for t in np.linspace(0, 1, 41):
                x, y = (sum(c * p[i] for c, p in zip(((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3), segment)) for i in (0, 1))
                inside = [n["id"] for n in boxes if n["x"] + 0.5 < x < n["x"] + n["width"] - 0.5 and n["y"] + 0.5 < y < n["y"] + n["height"] - 0.5]
                assert inside in ([], [a["id"]], [b["id"]]), (e["source"], e["target"], inside)
    blocks = [n for n in boxes if n["kind"] == "block"]
    for n in blocks:                                         # each block sits one column right of the furthest block it takes from
        before = [nodes[e["source"]]["column"] for e in graph["edges"] if e["target"] == n["id"] and not e["loop"] and e["source"].startswith("block:")]
        first = min(m["column"] for m in blocks)
        assert n["column"] == (max(before) + 1 if before else first)
    assert block_graph(load_model(path.read_text())) == graph   # the same specification always gives the same picture
