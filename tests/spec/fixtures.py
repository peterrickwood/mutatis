"""Paths to the shared specification files used by tests."""

from pathlib import Path

from mutatis.spec.yaml_io import load_model

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "tests" / "models"
MODEL_FILES = sorted(MODELS.glob("*.yaml"))
SKETCHES = [REPO / "docs" / "design" / "specification-sketches.md"]


def toy(name: str):
    return load_model((MODELS / f"{name}.yaml").read_text())
