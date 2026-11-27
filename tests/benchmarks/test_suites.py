"""Which runs each benchmark suite plans."""

import pytest

from benchmarks.samplers.methods import MAIN
from benchmarks.samplers.run import HELD_OUT, LONG, SHORT, VARIANTS
from benchmarks.samplers.targets import GYM, TARGETS


def test_the_long_suite_runs_the_held_out_targets_with_the_main_methods_only():
    tasks = LONG.tasks()
    for name in GYM:
        runs = {(t["method"], t["seed"]) for t in tasks if t["target"] == name}
        assert runs == {(m, s) for m in MAIN for s in range(4)}, name
    ours = [n for n in TARGETS if n not in GYM]
    for name in ours:
        runs = {(t["method"], t["seed"]) for t in tasks if t["target"] == name}
        assert runs == {(m, s) for m in MAIN for s in range(8)} | {(m, s) for m in VARIANTS for s in range(4)}, name
    assert len(tasks) == len(GYM) * len(MAIN) * 4 + len(ours) * (len(MAIN) * 8 + len(VARIANTS) * 4)


def test_the_short_suite_has_both_kinds_of_two_peak_target_and_no_held_out_ones():
    targets = {t["target"] for t in SHORT.tasks()}
    assert {"two_peaks_d5", "two_peaks_d20", "two_equal_peaks_d5", "two_equal_peaks_d20"} <= targets
    assert not targets & set(GYM)


def test_the_small_held_out_suite_runs_the_default_on_every_gym_target_but_the_largest():
    tasks = HELD_OUT.tasks()
    expected = {(name, "default", s) for name in GYM if name != "gym_stochastic_volatility" for s in range(2)}
    assert {(t["target"], t["method"], t["seed"]) for t in tasks} == expected and len(tasks) == len(expected) == 22


def test_chosen_methods_replace_the_main_ones_on_the_main_targets_only():
    tasks = SHORT.tasks(("default", "ladder_fitted_reference"))
    targets = [t for t in TARGETS if t in SHORT.main_targets]
    assert {(t["target"], t["method"], t["seed"]) for t in tasks} == {(n, m, s) for n in targets for m in ("default", "ladder_fitted_reference") for s in range(2)}
    assert len(tasks) == len(targets) * 2 * 2
    with pytest.raises(ValueError, match="unknown methods"):
        SHORT.tasks(("default", "no_such_method"))
