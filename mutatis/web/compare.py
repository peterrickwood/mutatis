"""Two stored runs side by side, as plain data for display: the edits that turn the first specification into the second
(`mutatis.spec.diff`), which parts of the model changed, every parameter's posterior summary in both, the reported quantities
that both runs have, and the log marginal likelihoods where defined."""

from mutatis.spec.diff import diff
from mutatis.versions.run import Result
from mutatis.web.describe import describe
from mutatis.web.estimates import quantity_panel


def comparison(first: Result, second: Result) -> dict:
    a, b = first.model, second.model
    try:
        edits, note = [e.to_dict() for e in diff(a, b)], None
    except ValueError as reason:
        edits, note = None, f"the specifications are not versions of one model: {reason}"
    changed = {"blocks": sorted(n for n in a.blocks.keys() | b.blocks.keys() if a.blocks.get(n) != b.blocks.get(n)),
               "data": sorted(s for s in a.data.keys() | b.data.keys() if a.data.get(s) != b.data.get(s)),
               "restrictions": sorted(r for r in a.restrictions.keys() | b.restrictions.keys() if a.restrictions.get(r) != b.restrictions.get(r)),
               "model_parameters": sorted(p for p in a.parameters.keys() | b.parameters.keys() if a.parameters.get(p) != b.parameters.get(p)),
               "expectations": a.expectations != b.expectations, "sets": sorted(s for s in a.sets.keys() | b.sets.keys() if a.sets.get(s) != b.sets.get(s))}
    pa, pb = first.summary["parameters"], second.summary["parameters"]
    parameters = [{"label": label, "first": pa.get(label), "second": pb.get(label)} for label in [*pa, *(n for n in pb if n not in pa)]]
    qa, qb = first.summary.get("quantities", {}), second.summary.get("quantities", {})
    quantities = [{"quantity": text, "first": quantity_panel(text, qa[text]), "second": quantity_panel(text, qb[text])} for text in qa if text in qb]
    return {"first": _run(first), "second": _run(second), "edits": edits, "note": note, "changed": changed, "same_data": first.run["data"] == second.run["data"],
            "parameters": parameters, "quantities": quantities, "graph": describe(b)["graph"],
            "log_marginal_likelihood": [first.summary.get("log_marginal_likelihood"), second.summary.get("log_marginal_likelihood")]}


def _run(result: Result) -> dict:
    return {"key": result.folder.name, "model": result.model.name, "method": result.run["method"], "route": result.run["route"],
            "edits": result.run.get("edits", []), "data_dates": result.run.get("data_dates")}
