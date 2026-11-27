"""The inspector's HTTP server: every page and answer over a store of toy runs, read-only, and nothing outside its own files."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from mutatis.web import estimates
from mutatis.web.server import Inspector, make_server, to_json
from tests.web.conftest import TOYS


@pytest.fixture(scope="module")
def address(toy_store):
    server = make_server(toy_store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url, method="GET"):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=300) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type"), error.read()


def strict(body: bytes):
    """JSON as a browser reads it: NaN and infinities are not allowed."""
    return json.loads(body, parse_constant=lambda name: pytest.fail(f"{name} in a JSON answer"))


def test_the_page_its_files_and_every_answer(address):
    for path, kind in (("/", "text/html"), ("/static/app.js", "text/javascript"), ("/static/style.css", "text/css")):
        status, content_type, _ = get(address + path)
        assert status == 200 and content_type.startswith(kind), path
    listing = strict(get(address + "/api/runs")[2])
    assert {r["key"] for r in listing["runs"]} == set(TOYS) and all(r["has_data"] for r in listing["runs"])
    for key in ("hidden_level", "error_correction"):
        for what in ("model", "parameters", "fit", "quantities"):
            status, content_type, body = get(f"{address}/api/runs/{key}/{what}")
            assert status == 200 and content_type == "application/json", (key, what)
            strict(body)
    compared = strict(get(f"{address}/api/compare/hidden_level/error_correction")[2])
    assert compared["first"]["key"] == "hidden_level" and compared["second"]["key"] == "error_correction"


def test_refusals(address):
    assert get(f"{address}/api/runs/no_such_run/fit")[0] == 404
    assert get(f"{address}/api/runs/hidden_level/no_such_page")[0] == 404
    for path in ("/static/../server.py", "/static/%2e%2e/server.py", "/static/missing.js", "/server.py", "/api/runs/..%2f..%2fetc/model"):
        assert get(address + path)[0] == 404, path
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        status, _, body = get(address + "/api/runs", method)
        assert status == 405 and "read-only" in strict(body)["error"]


def test_nan_and_numpy_values_become_plain_json():
    import numpy as np

    assert json.loads(to_json({"a": [1.0, float("nan"), np.float64(np.inf)], "b": np.arange(2), "c": np.bool_(True)})) == {"a": [1.0, None, None], "b": [0, 1], "c": True}


def test_each_answer_is_computed_once_however_many_ask_at_once(toy_store, monkeypatch):
    calls = []

    def slow(loaded):
        calls.append(1)
        time.sleep(0.2)
        return {"observed": []}

    monkeypatch.setattr(estimates, "fit_panels", slow)
    inspector = Inspector(toy_store)
    threads = [threading.Thread(target=inspector.fit, args=("hidden_level",)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and inspector.fit("hidden_level") == {"observed": []}
