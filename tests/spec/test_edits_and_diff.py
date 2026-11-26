"""Edits change a model as stated, can be written out and read back, and diff recovers them."""

import random

import pytest

from mutatis.spec import edits as e
from mutatis.spec.checks import check
from mutatis.spec.diff import diff
from mutatis.spec.edits import EditError, apply_edits, edit_from_dict
from mutatis.spec.model import Block, DataLink, Parameter, Restriction
from mutatis.spec.numbers import find_numbers
from tests.spec.fixtures import toy

HOUSING_INVESTMENT = Block(inputs=("r",), outputs=("invest[h]",), parameters={"phi_h": Parameter(prior="normal(2, 0.5)")},
                           equations=("invest[h] = -phi_h * r",))

# (model name, a single edit). Each must apply, survive writing out and reading back, and be recovered exactly by diff.
SINGLE_EDITS = [
    ("forward_looking", e.SetPrior("price.a", fixed=0.5)),
    ("forward_looking", e.SetPrior("demand.rho", prior="uniform(0, 1)")),
    ("forward_looking", e.SetExpectations("naive")),
    ("forward_looking", e.ReplaceEquation("price", 0, "p = x + a * E(p(+1)) + 0")),
    ("forward_looking", e.SetShock("demand.e", "student_t(4, 0, sigma_x)")),
    ("forward_looking", e.AddParameter("price.b", Parameter(fixed=0.1))),
    ("forward_looking", e.RemoveParameter("price.a")),
    ("forward_looking", e.AddData("extra", DataLink(observes="x + p", error="normal(0, 1)"))),
    ("forward_looking", e.RemoveData("p_obs")),
    ("forward_looking", e.AddRestriction("price_rises", Restriction("response(p, e, horizons=0)", ">= 0"))),
    ("hidden_level", e.SetPrior("y.sigma_eps", fixed=0.2)),
    ("hidden_level", e.RemoveBlock("level")),
    ("two_variable", e.SetPrior("dynamics.Sigma", prior="jeffreys")),
    ("two_variable", e.ExtendSet("series", ("c",))),
    ("two_variable", e.ShrinkSet("series", ("b",))),
    ("two_variable", e.AddSet("lags", (1, 2))),
    ("sectors", e.ExtendSet("sectors", ("h",))),
    ("sectors", e.Untie("investment.phi", "m", prior="normal(2, 0.5)")),
    ("sectors", e.ReplaceBlock("investment[m]", Block(inputs=("r",), outputs=("invest[m]",), equations=("invest[m] = -2 * r",)))),
]


@pytest.mark.parametrize("name, edit", SINGLE_EDITS, ids=lambda v: getattr(v, "op", v))
def test_single_edit(name, edit):
    before = toy(name)
    after = edit.apply(before)
    assert after != before
    assert edit_from_dict(edit.to_dict()) == edit
    assert diff(before, after) == [edit]
    assert apply_edits(before, diff(before, after)) == after


def test_the_housing_style_expansion_is_a_short_list_of_edits():
    before = toy("sectors")
    expansion = [
        e.ExtendSet("sectors", ("h",)),
        e.ReplaceBlock("investment[h]", HOUSING_INVESTMENT),
        e.Untie("capital.delta", "h", prior="beta(mean=0.02, sd=0.01)"),
    ]
    after = apply_edits(before, expansion)
    assert check(after) == []
    assert apply_edits(before, diff(before, after)) == after
    collapse = apply_edits(after, [e.RestoreInstance("investment[h]"), e.Bind("capital.delta", "h"), e.ShrinkSet("sectors", ("h",))])
    assert collapse == before


def test_replacing_a_block_must_keep_its_ports():
    with pytest.raises(EditError, match="same outputs"):
        e.ReplaceBlock("investment[m]", Block(inputs=("r",), outputs=("invest[n]",), equations=("invest[n] = r",))).apply(toy("sectors"))
    with pytest.raises(EditError, match="same inputs"):
        e.ReplaceBlock("price", Block(outputs=("p",), equations=("p = 1",))).apply(toy("forward_looking"))


def test_name_number():
    before = toy("sectors")
    (site,) = find_numbers(before)
    after = e.NameNumber("capital", ((site.equation, site.start),), "full").apply(before)
    assert after.blocks["capital"].equations[0] == "k[j] = (full - delta) * k[j](-1) + invest[j]"
    assert after.blocks["capital"].parameters["full"] == Parameter(fixed=1)
    assert check(after) == []
    assert apply_edits(before, diff(before, after)) == after
    shift = before.blocks["capital"].equations[0].index("1)")
    with pytest.raises(EditError, match="no nameable number"):
        e.NameNumber("capital", ((0, shift),), "lag").apply(before)  # the 1 in k[j](-1) is a time shift, not a size


def test_edit_lists_read_from_yaml():
    edits = [edit_from_dict(d) for d in [
        {"set_prior": {"target": "economy.Q", "prior": "none"}},
        {"extend_set": {"set": "production_sectors", "add": ["h"]}},
        {"add_restriction": {"name": "rate_rises", "restriction": {"quantity": "response(y[cash_rate], eps[policy], horizons=0..3)", "require": ">= 0"}}},
    ]]
    assert edits == [e.SetPrior("economy.Q", prior="none"), e.ExtendSet("production_sectors", ("h",)),
                     e.AddRestriction("rate_rises", Restriction("response(y[cash_rate], eps[policy], horizons=0..3)", ">= 0"))]
    with pytest.raises(EditError, match="unknown edit"):
        edit_from_dict({"set_priors": {}})


@pytest.mark.parametrize("seed", range(20))
def test_diff_reproduces_any_sequence_of_edits(seed):
    rng = random.Random(seed)
    model = toy("sectors")
    options = [
        lambda m: e.SetPrior("rate.rho", fixed=rng.choice([0.3, 0.6])),
        lambda m: e.SetPrior("investment.phi", prior=rng.choice(["normal(1, 0.5)", "normal(0, 2)"])),
        lambda m: e.ExtendSet("sectors", (f"s{rng.randrange(10**6)}",)),
        lambda m: e.Untie("investment.phi", rng.choice(m.sets["sectors"]), fixed=rng.random()),
        lambda m: e.ReplaceEquation("rate", 0, rng.choice(["r = rho * r(-1) + e", "r = 0.9 * rho * r(-1) + e"])),
        lambda m: e.SetShock("rate.e", rng.choice(["normal(0, sigma_e)", "student_t(5, 0, sigma_e)"])),
        lambda m: e.AddData(f"d{rng.randrange(10**6)}", DataLink(observes="r", error="normal(0, 1)")),
    ]
    edited = model
    for _ in range(6):
        edit = rng.choice(options)(edited)
        try:
            edited = edit.apply(edited)
        except EditError:
            continue
    assert apply_edits(model, diff(model, edited)) == edited
