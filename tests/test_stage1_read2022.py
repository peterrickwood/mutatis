"""Stage 1 acceptance (slow tier): Read (2022), Restrictions (1) to (5), conventional and range results, against his published
figure data. Needs network access the first time (the data are downloaded per demos/read2022/data.yaml).

Run: .venv/bin/python -m pytest -m slow -s tests/test_stage1_read2022.py

One kind of difference is expected and explained (docs/design/decisions.md, 28): at horizons where a response is itself
restricted, Read's published bounds are exactly zero, while the exact bounds computed here are sometimes slightly inside the
restricted side, and an independent optimiser confirms them. Only that kind of difference is accepted, and each one is printed.
"""

from pathlib import Path

import pytest
import yaml

from mutatis.measure.manifest import load_manifest
from mutatis.spec.edits import load_edit_file
from mutatis.spec.yaml_io import load_model
from mutatis.versions import run as runs

pytestmark = pytest.mark.slow
DEMO = Path(__file__).resolve().parents[1] / "demos" / "read2022"


@pytest.fixture(scope="module")
def inputs():
    return load_model((DEMO / "model.yaml").read_text()), load_manifest(DEMO / "data.yaml"), tuple(yaml.safe_load((DEMO / "report.yaml").read_text()))


def _explained(row) -> bool:
    """Read's bound is zero where a restriction pins it, and ours is a little inside the allowed side."""
    return abs(row["theirs"]) < 1e-6 and abs(row["ours"]) < 0.05 and row["statistic"] in ("set_mean_lower", "set_mean_upper", "robust68_lower", "robust68_upper")


@pytest.mark.parametrize("restriction", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", ["conventional", "ranges"])
def test_restriction(restriction, kind, inputs, tmp_path_factory):
    model, dataset, report = inputs
    edits = load_edit_file(DEMO / "edits" / f"restriction{restriction}.yaml")
    if kind == "ranges":
        edits += load_edit_file(DEMO / "edits" / "ranges.yaml")
    store = tmp_path_factory.getbasetemp() / "read2022"
    result = runs.run(model, dataset, store=store, edits=edits, report=report, n_draws=1000)
    rows = runs.compare_with_reference(result, DEMO / "reference" / f"restriction{restriction}.csv")
    assert rows, "nothing was compared"
    failures = [r for r in rows if not r["ok"] and not _explained(r)]
    explained = [r for r in rows if not r["ok"] and _explained(r)]
    print(f"\nRestriction ({restriction}), {kind}: {len(rows)} statistics, {len(rows) - len(failures) - len(explained)} within tolerance, "
          f"{len(explained)} explained differences, {len(failures)} failures; share of draws where the restrictions can hold "
          f"{result.summary.get('plausibility', 1.0):.3f}")
    for r in explained + failures:
        print(f"   {'explained' if r in explained else 'FAILURE  '} {r['quantity']}[{r['element']}] {r['statistic']}: ours {r['ours']:.5f}, Read {r['theirs']:.5f}, tolerance {r['tolerance']:.5f}")
    assert not failures
