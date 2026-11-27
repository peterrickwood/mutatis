"""The report is written from recorded results alone, whatever mix of outcomes they hold, and is well-formed HTML."""

import json
from html.parser import HTMLParser

from benchmarks.samplers import report


def _record(target, group, size, method, seed, ok, wrong=False, status="ok"):
    r = {"target": target, "group": group, "description": f"{target} description", "size": size, "n_quantities": size,
         "reference_source": "exact", "true_log_evidence": -3.0, "has_peaks": target.startswith("two_peaks"),
         "method": method, "seed": seed, "status": status, "jobs": 2}
    if status != "ok":
        return r | {"error": "Boom: it broke", "posterior_ok": False, "confidently_wrong": False}
    return r | {"seconds": 12.5, "posterior_ok": ok, "checks_passed": None if method == "smc" else not wrong and ok,
                "confidently_wrong": wrong,
                "diagnostics": {"max_rhat": 1.002, "min_ess": 812.0, "divergences": 0, "rungs": 12, "round_trips": 30,
                                "gradient_evaluations": 123456, "converged": ok, "findings": [] if ok else ["R-hat 1.2"]},
                "accuracy": {"max_abs_mean_error": 0.03 if ok else 0.4, "worst_quantity": "z[0]", "min_sd_ratio": 0.97, "max_sd_ratio": 1.02,
                             "max_ks": 0.01, "failed": [] if ok else ["mean error 0.400 sd in z[0]"], "posterior_ok": ok,
                             **({"peak_shares": [0.31, 0.69]} if target.startswith("two_peaks") else {})},
                "evidence": {"stepping_stones": {"log_value": -3.1, "standard_error": 0.05, "error": -0.1, "ok": True},
                             "bridge": {"error_message": "LinAlgError: singular", "ok": False}}
                if method == "default" else {"smc": {"log_value": -9.0, "standard_error": None, "error": -6.0, "ok": False}}
                if method == "smc" else {"bridge": {"log_value": -2.95, "standard_error": 0.02, "error": 0.05, "ok": True}}}


class _Checker(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "circle", "rect", "path", "line"}

    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        assert self.stack and self.stack[-1] == tag, (self.stack[-5:], tag)
        self.stack.pop()


def test_report_from_mixed_results(tmp_path):
    rows = []
    for target, group, size in [("normal_means_d2", "Independent normal means", 2), ("normal_means_d10", "Independent normal means", 10),
                                ("two_peaks_d5", "Two separated peaks", 5)]:
        for method in ["default", "nuts", "smc", "random_walk"]:
            for seed in range(2):
                rows.append(_record(target, group, size, method, seed, ok=(method != "smc" or seed == 0), wrong=(method == "smc" and seed == 1)))
    rows.append(_record("funnel_d10", "Funnel", 10, "nuts", 0, ok=False, status="failed"))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    path = report.write(tmp_path)
    text = path.read_text()
    checker = _Checker()
    checker.feed(text)
    assert checker.stack == []
    for heading in ["Summary", "Accuracy by target", "Confidently wrong runs", "Runs that crashed or timed out", "Every target in detail",
                    "Dimension: independent normal means"]:
        assert heading in text
    assert "Boom: it broke" in text and "LinAlgError" in text
