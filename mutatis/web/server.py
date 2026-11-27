"""The read-only inspector: a small HTTP server over one results store, answering GET requests only.

    GET /                                the page (static/index.html), which draws everything below in the browser
    GET /static/<file>                   its script and style sheet
    GET /api/runs                        every run in the store
    GET /api/runs/<key>/model            the specification: blocks, equations as LaTeX, data series, the graph and its layout
    GET /api/runs/<key>/parameters       each parameter's prior against its posterior
    GET /api/runs/<key>/fit              each data series against the model's one-step-ahead prediction
    GET /api/runs/<key>/quantities       the reported quantities
    GET /api/compare/<key>/<key>         two runs side by side

A run's key is the name of its folder in the store. Answers are computed when first asked for and kept; the store is only read.
"""

import json
import math
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from mutatis.versions.run import load_result
from mutatis.web import estimates
from mutatis.web.compare import comparison
from mutatis.web.describe import describe

STATIC = Path(__file__).resolve().parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class NotFound(LookupError):
    pass


class Inspector:
    """The answers for one results store, computed on first request and kept."""

    def __init__(self, store: str | Path):
        self.store = Path(store).resolve()
        if not self.store.is_dir():
            raise FileNotFoundError(f"no results store at {self.store}")
        self._answers: dict[tuple, object] = {}
        self._locks: dict[tuple, threading.Lock] = {}
        self._guard = threading.Lock()

    def folders(self) -> dict[str, Path]:
        """Run folders by key: the store's subfolders that hold a run, or the store itself if it is one run's folder."""
        candidates = [self.store] if (self.store / "run.json").exists() else sorted(p for p in self.store.iterdir() if p.is_dir())
        return {p.name: p for p in candidates if (p / "run.json").exists() and (p / "summary.json").exists() and (p / "spec.yaml").exists()}

    def _answer(self, name: tuple, make):
        with self._guard:
            if name in self._answers:
                return self._answers[name]
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:   # one computation per answer, however many requests ask for it at once
            if name not in self._answers:
                self._answers[name] = make()
            return self._answers[name]

    def _folder(self, key: str) -> Path:
        folders = self.folders()
        if key not in folders:
            raise NotFound(f"no run {key!r} in {self.store}")
        return folders[key]

    def result(self, key: str):
        return self._answer(("result", key), lambda: load_result(self._folder(key)))

    def loaded(self, key: str):
        return self._answer(("loaded", key), lambda: estimates.load(self.result(key)))

    def runs(self) -> dict:
        listed = []
        for key, folder in self.folders().items():
            try:
                result = self.result(key)
            except Exception as error:   # a damaged folder is listed, not fatal
                listed.append({"key": key, "error": str(error)})
                continue
            summary, record = result.summary, result.run
            listed.append({"key": key, "model": record.get("model"), "base_model": record.get("base_model"), "edits": record.get("edits", []),
                           "method": record.get("method"), "route": record.get("route"), "seed": record.get("seed"), "data_dates": record.get("data_dates"),
                           "data": record.get("data"), "draws": summary.get("draws"), "log_marginal_likelihood": summary.get("log_marginal_likelihood"),
                           "converged": summary.get("converged"), "status": summary.get("status"), "has_data": (folder / "data.npz").exists(),
                           "quantities": list(summary.get("quantities", {})), "stored": (folder / "summary.json").stat().st_mtime})
        listed.sort(key=lambda r: (-r.get("stored", 0), r["key"]))
        return {"store": str(self.store), "runs": listed}

    def model(self, key: str) -> dict:
        def make():
            result = self.result(key)
            diagnostics = {k: v for k, v in result.summary.items() if k not in ("parameters", "quantities")}
            return {"key": key, "run": result.run, "diagnostics": diagnostics, **describe(result.model)}
        return self._answer(("model", key), make)

    def parameters(self, key: str) -> dict:
        return self._answer(("parameters", key), lambda: estimates.parameter_panels(self.loaded(key)))

    def fit(self, key: str) -> dict:
        return self._answer(("fit", key), lambda: estimates.fit_panels(self.loaded(key)))

    def quantities(self, key: str) -> dict:
        return self._answer(("quantities", key), lambda: {"quantities": estimates.quantity_panels(self.result(key))})

    def compare(self, first: str, second: str) -> dict:
        return self._answer(("compare", first, second), lambda: comparison(self.result(first), self.result(second)))

    def route(self, path: str):
        """The answer for an API path, such as /api/runs/KEY/fit."""
        parts = [unquote(p) for p in path.strip("/").split("/")]
        match parts:
            case ["api", "runs"]:
                return self.runs()
            case ["api", "runs", key, ("model" | "parameters" | "fit" | "quantities") as what]:
                return getattr(self, what)(key)
            case ["api", "compare", first, second]:
                return self.compare(first, second)
        raise NotFound(f"no such address: {path}")


def to_json(value) -> bytes:
    """JSON with NaN and infinities as null, and NumPy numbers and arrays as plain ones."""
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, (np.floating, float)):
            return float(v) if math.isfinite(v) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return json.dumps(clean(value), allow_nan=False).encode()


def make_server(store: str | Path, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    inspector = Inspector(store)

    class Handler(BaseHTTPRequestHandler):
        server_version = "mutatis-inspector"

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path.startswith("/api/"):
                    self._send(HTTPStatus.OK, to_json(inspector.route(path)), "application/json")
                else:
                    self._static(path)
            except NotFound as missing:
                self._send(HTTPStatus.NOT_FOUND, to_json({"error": str(missing)}), "application/json")
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, to_json({"error": f"{type(error).__name__}: {error}"}), "application/json")

        def _static(self, path: str):
            name = "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/") if path.startswith("/static/") else None
            target = (STATIC / name).resolve() if name else None
            if target is None or target.parent != STATIC or not target.is_file() or target.suffix not in TYPES:
                raise NotFound(f"no such page: {path}")
            self._send(HTTPStatus.OK, target.read_bytes(), TYPES[target.suffix])

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self):
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, to_json({"error": "the inspector is read-only: it answers GET requests only"}), "application/json")

        do_POST = do_PUT = do_PATCH = do_DELETE = _refuse

        def log_message(self, format, *args):   # quiet: errors are printed above
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.inspector = inspector
    return server


def serve(store: str | Path, port: int = 8765, host: str = "127.0.0.1") -> None:
    server = make_server(store, port, host)
    print(f"mutatis inspector: http://{host}:{server.server_address[1]}/  (store: {server.inspector.store}; Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
