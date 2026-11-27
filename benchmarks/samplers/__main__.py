"""python -m benchmarks.samplers [--suite short|long|held_out] [--methods A,B] [--jobs N] [--threads T] [--name NAME] [--limit N] [--report-only]

The short suite (the default) takes about an hour on 8 cores with `--jobs 7`; the long suite is for a machine with hundreds
of cores (`--suite long --jobs 180` on 192). A stopped run resumes when given the same `--name`.
"""

import argparse
import os
import time
from pathlib import Path

from benchmarks.samplers import report, run


def main():
    parser = argparse.ArgumentParser(description="Compare samplers on synthetic problems with known answers.")
    parser.add_argument("--suite", choices=sorted(run.SUITES), default="short")
    parser.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1), help="runs at a time, each in its own process")
    parser.add_argument("--threads", type=int, default=1, help="CPUs given to each run")
    parser.add_argument("--name", default=None, help="the run's folder under benchmarks/samplers/results (default: suite and time)")
    parser.add_argument("--methods", default=None, help="comma-separated methods to run on the suite's main targets, in place of the main methods")
    parser.add_argument("--limit", type=int, default=None, help="only the suite's first N runs (to check that it works)")
    parser.add_argument("--report-only", action="store_true", help="write the report from the results already recorded")
    args = parser.parse_args()
    suite = run.SUITES[args.suite]
    run_dir = run.RESULTS / (args.name or f"{suite.name}-{time.strftime('%Y%m%d-%H%M')}")
    if not args.report_only:
        run.run(run_dir, suite, args.jobs, args.threads, args.limit, methods=tuple(args.methods.split(",")) if args.methods else None)
    print(f"report: {Path(report.write(run_dir)).resolve()}")


if __name__ == "__main__":
    main()
