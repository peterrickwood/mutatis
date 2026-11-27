"""A comparison of samplers on synthetic posterior problems whose answers are known.

    python -m benchmarks.samplers                                 the short suite: about an hour on 8 cores
    python -m benchmarks.samplers --suite long --jobs 180         the long suite: for hundreds of cores, within a day

Each run of one method on one target with one seed happens in its own process (so a run that hangs can be stopped), and its
result is one line of results.jsonl in the run's folder; a stopped benchmark resumes where it left off. The report,
report.html in the same folder, is written at the end, or on its own with `--report-only`.

Modules: `targets` (the problems and their known answers), `quadrature` (answers by numerical integration for the model
targets), `methods` (the samplers compared), `metrics` (accuracy against the known answers), `worker` (one run), `run` (the
suites, the schedule, and why each run is pinned to its own CPUs), `report` (the HTML report).
"""
