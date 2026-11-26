"""Reading and writing specifications, and the checks that need no numbers."""

import pytest

from mutatis.spec.checks import check
from mutatis.spec.model import Parameter
from mutatis.spec.numbers import find_numbers
from mutatis.spec.yaml_io import SpecError, dump_model, load_model
from tests.spec.fixtures import MODEL_FILES, toy


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: p.stem)
def test_round_trip_and_clean_checks(path):
    model = load_model(path.read_text())
    assert load_model(dump_model(model)) == model
    assert check(model) == []


def test_a_comma_in_an_unquoted_inline_value_is_caught():
    text = """
model: m
blocks:
  b:
    outputs: [x]
    parameters:
      c: {prior: normal(0, 10)}
    equations: ["x = c"]
"""
    with pytest.raises(SpecError, match="quote the value"):
        load_model(text)


def test_unknown_keys_and_bad_parameters_are_errors():
    with pytest.raises(SpecError, match="unknown key"):
        load_model("model: m\nblocks:\n  b: {outputs: [x], equation: ['x = 1']}\n")
    with pytest.raises(SpecError, match="exactly one of"):
        load_model("model: m\nblocks:\n  b:\n    outputs: [x]\n    parameters: {c: {fixed: 1, prior: flat}}\n    equations: ['x = c']\n")


def test_templates_sets_and_shock_shapes_are_read():
    model = toy("sectors")
    assert model.blocks["capital"].over == ("j", "sectors")
    assert model.blocks["investment"].shocks["u"].shape == ("sectors",)
    ranged = load_model("model: m\nsets: {lags: 1..4}\nblocks: {}\n")
    assert ranged.sets["lags"] == (1, 2, 3, 4)


def _messages(model):
    return [str(f) for f in check(model)]


def test_checks_find_unknown_names_duplicate_outputs_and_missing_inputs():
    model = toy("forward_looking")
    price = model.blocks["price"]
    broken = model.__class__(**{**model.__dict__, "blocks": {**model.blocks, "price": price.__class__(**{**price.__dict__, "equations": ("p = x + b * E(p(+1))",)})}})
    assert any("'b' is not an input" in m for m in _messages(broken))
    twice = model.__class__(**{**model.__dict__, "blocks": {**model.blocks, "copy": price}})
    assert any("p is already an output of price" in m for m in _messages(twice))
    orphan = model.__class__(**{**model.__dict__, "blocks": {"price": price}})
    assert any("x is not an output of any block" in m for m in _messages(orphan))


def test_checks_find_bad_distributions_and_domains():
    model = toy("hidden_level")
    bad = model.with_parameter("level.sigma_eta", Parameter(prior="lognormal(0, 1)"))
    assert any("unknown distribution" in m for m in _messages(bad))
    bad = model.with_parameter("level.sigma_eta", Parameter(prior="half_normal(scale=1)", domain="negative"))
    assert any("unknown domain" in m for m in _messages(bad))


def test_numbers_written_in_equations_are_found():
    sites = find_numbers(toy("sectors"))
    assert [(s.block, s.text) for s in sites] == [("capital", "1")]  # the 1 in (1 - delta); the shift in k[j](-1) is structure, not a size
    sites = find_numbers(toy("error_correction"))
    assert sites == []
