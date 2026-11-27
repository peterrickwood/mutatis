"""Stage 3 acceptance (slow tier): the read-only inspector displays every toy and both demonstrations, through the same code.

Run: .venv/bin/python -m pytest -m slow -s tests/test_stage3_inspector.py

Every model is estimated through the one generic `run` into one results store, with short runs where the default would take
long: Read (2022) under Restriction (1), conventionally and as ranges (1,000 exact draws each); Herbst and Schorfheide's model by
SMC with 500 particles; every toy in tests/models by SMC with 300 particles (except those in CANNOT_SAMPLE, stored from prior
draws, with the reason given there). Then, for every run, the inspector's answers must be
complete (every equation as mathematics, a posterior and, where proper, a prior for every estimated element, a fitted median and
band for every observed series in every period, the reported quantities), and the server must answer on every page.
"""

import json
import math
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml

from mutatis.measure.manifest import load_manifest
from mutatis.spec.edits import load_edit_file
from mutatis.spec.yaml_io import load_model
from mutatis.versions import run as runs
from mutatis.web import estimates
from mutatis.web.server import Inspector, make_server
from tests.spec.fixtures import toy
from tests.web.conftest import TOYS, store_draws, toy_data

pytestmark = pytest.mark.slow
DEMOS = Path(__file__).resolve().parents[1] / "demos"
# Toys no gradient sampler can estimate, stored from 1,000 prior draws instead so the inspector is still tested on them. The two
# shocks of two_shock_forward_looking are mixed by a rotation with an even prior, which gradient moves cannot change (its ranges,
# with no prior on the rotation, are estimated in tests/infer/test_restrictions.py).
CANNOT_SAMPLE = {"two_shock_forward_looking"}


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    store = tmp_path_factory.mktemp("stage3")
    keys, seconds = {}, {}

    def timed(name, make):
        start = time.monotonic()
        keys[name] = make().folder.name
        seconds[name] = time.monotonic() - start

    read = DEMOS / "read2022"
    model, data, report = load_model((read / "model.yaml").read_text()), load_manifest(read / "data.yaml"), tuple(yaml.safe_load((read / "report.yaml").read_text()))
    restriction = load_edit_file(read / "edits" / "restriction1.yaml")
    timed("read2022", lambda: runs.run(model, data, store=store, edits=restriction, report=report, n_draws=1000))
    timed("read2022 ranges", lambda: runs.run(model, data, store=store, edits=restriction + load_edit_file(read / "edits" / "ranges.yaml"), report=report, n_draws=1000))
    hs = DEMOS / "herbst_schorfheide2016"
    timed("herbst_schorfheide2016", lambda: runs.run(load_model((hs / "model.yaml").read_text()), load_manifest(hs / "data.yaml"), store=store, method="smc", n_particles=500))
    for name in TOYS:
        if name in CANNOT_SAMPLE:
            timed(name, lambda: store_draws(store / name, toy(name), toy_data(name, periods=80), n=1000))
            continue
        timed(name, lambda: runs.run(toy(name), toy_data(name, periods=80), store=store, method="smc", n_particles=300))
    print("\n   seconds to estimate and store: " + ", ".join(f"{k} {v:.0f}" for k, v in seconds.items()))
    return store, keys


def _complete(inspector: Inspector, key: str, report: int = 0):
    model = inspector.model(key)
    assert all(eq["latex"] for b in model["blocks"] for eq in b["equations"]), "an equation was not written as mathematics"
    assert all(s["observes_latex"] for s in model["data"] if s["observes"]), "an observation was not written as mathematics"
    assert len(model["graph"]["nodes"]) == len(model["blocks"]) + len(model["data"])
    system = inspector.loaded(key).problem.system
    for entry in inspector.parameters(key)["parameters"]:
        info = system.parameters[entry["key"]]
        if entry["kind"] == "fixed":
            assert "fixed" in entry
            continue
        if entry["kind"] == "no_prior":
            assert entry["note"] and not entry["elements"]
            continue
        assert len(entry["elements"]) == math.prod(info.shape), entry["key"]
        for element in entry["elements"]:
            assert len(element["posterior"]) == estimates.BINS and element["window"][0] < element["window"][1], element["label"]
            assert (element.get("prior") is not None) == info.prior.proper, element["label"]
    fit = inspector.fit(key)
    assert fit["draws"] >= 0.95 * fit["draws_tried"], f"only {fit['draws']} of {fit['draws_tried']} draws gave a usable model"
    assert [s["series"] for s in fit["observed"]] == [o.series for o in system.observations]
    for series in fit["observed"]:
        for k in ("median", "lower", "upper"):
            assert len(series[k]) == len(fit["dates"]) and all(v is not None and np.isfinite(v) for v in series[k]), (series["series"], k)
    quantities = inspector.quantities(key)["quantities"]
    assert len(quantities) == report and all(q["latex"] and q["axis_label"] == "horizon" for q in quantities)
    return model, fit


def test_every_toy_and_both_demonstrations_display(stored):
    store, keys = stored
    inspector = Inspector(store)
    assert {r["key"] for r in inspector.runs()["runs"]} == set(keys.values())
    for name, key in keys.items():
        start = time.monotonic()
        model, fit = _complete(inspector, key, report=4 if name.startswith("read2022") else 0)
        print(f"   {name}: {model['name']}, route {fit['route']}, {len(fit['observed'])} series fitted, payloads in {time.monotonic() - start:.1f} s")
    ranges = inspector.parameters(keys["read2022 ranges"])["parameters"]
    assert next(p for p in ranges if p["key"] == "economy.Q")["kind"] == "no_prior"
    assert inspector.quantities(keys["read2022 ranges"])["quantities"][0]["kind"] == "ranges"


def test_the_two_versions_of_read_compare(stored):
    store, keys = stored
    compared = Inspector(store).compare(keys["read2022"], keys["read2022 ranges"])
    assert compared["edits"] == [{"set_prior": {"target": "economy.Q", "prior": "none"}}] and compared["changed"]["blocks"] == ["economy"]
    assert compared["same_data"] and len(compared["quantities"]) == 4
    assert [q["second"]["kind"] for q in compared["quantities"]] == ["ranges"] * 4


def test_the_server_answers_on_every_page(stored):
    store, keys = stored
    server = make_server(store, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for path in ("/", "/static/app.js", "/static/style.css", "/api/runs"):
            assert urllib.request.urlopen(address + path, timeout=60).status == 200, path
        for key in keys.values():
            for what in ("model", "parameters", "fit", "quantities"):
                with urllib.request.urlopen(f"{address}/api/runs/{key}/{what}", timeout=600) as response:
                    assert response.status == 200
                    json.loads(response.read(), parse_constant=lambda c: pytest.fail(f"{c} in {key}/{what}"))
        with urllib.request.urlopen(f"{address}/api/compare/{keys['read2022']}/{keys['read2022 ranges']}", timeout=600) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
