"""The benchmark's suites and schedule: which runs, in how many parallel processes, resuming from results already recorded.

Suites:
- `short` (the default): a representative subset that covers every kind of target at a spread of sizes, the three main
  methods with two seeds, and the variants of the default on a handful of targets. Sized to finish in about an hour on an
  8-core machine.
- `long`: every target and size (including larger ones), all methods, more seeds. Meant for a machine with hundreds of cores,
  for example `python -m benchmarks.samplers --suite long --jobs 180` on 192, within a day.

Each run is a separate process pinned to its own CPUs (`--threads` of them, default 1), so that many processes do not fight
over the cores. The pinning uses the CPU affinity mask (`os.sched_setaffinity`), set by the worker before JAX starts: XLA
sizes its thread pool from that mask. The XLA flags often suggested for this (`--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1`) are accepted by JAX 0.12.1 but have no effect: a process given them still used 6.2 cores on a
large matrix product (CPU time over wall time), while the same process pinned to one core used 1.00 and to two cores 1.93.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from benchmarks.samplers.methods import MAIN, METHODS
from benchmarks.samplers.targets import TARGETS

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
MODEL_TARGETS = tuple(t for t in TARGETS if t.startswith(("hidden_level", "forward_looking")))
VARIANTS = tuple(m for m in METHODS if m not in MAIN)


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    main_targets: tuple
    main_seeds: int
    variant_targets: tuple
    variant_seeds: int
    timeout: float           # seconds before a run is stopped and recorded as failed

    def tasks(self) -> list[dict]:
        out = []
        for target in TARGETS:   # in the registry's order, so similar targets run together
            if target in self.main_targets:
                out += [{"target": target, "method": m, "seed": s} for m in MAIN for s in range(self.main_seeds)]
            if target in self.variant_targets:
                out += [{"target": target, "method": m, "seed": s} for m in VARIANTS for s in range(self.variant_seeds)]
        return out


SHORT = Suite(
    "short", "every kind of target at a spread of sizes, leaving out the most expensive; the three main methods with 2 seeds; "
    "the four variants of the default with 1 seed on 4 targets",
    main_targets=("normal_means_d2", "normal_means_d10", "normal_means_d50", "correlated_normal_d10_c100", "correlated_normal_d10_c10000",
                  "scales_d20", "two_peaks_d1", "two_peaks_d5", "two_peaks_d20", "banana_d2_strong", "banana_d10_mild", "funnel_d10",
                  "student_t_d10", "conjugate_regression_n2_p1", "conjugate_regression_n2_p4", "hidden_level_shocks_T50", "forward_looking"),
    main_seeds=2,
    variant_targets=("normal_means_d10", "two_peaks_d5", "funnel_d10", "conjugate_regression_n2_p1"),
    variant_seeds=1,
    timeout=1800.0)

LONG = Suite(
    "long", "every target and size, including larger ones; the three main methods with 8 seeds; the four variants of the "
    "default with 4 seeds on every target",
    main_targets=tuple(TARGETS), main_seeds=8, variant_targets=tuple(TARGETS), variant_seeds=4, timeout=6 * 3600.0)

SUITES = {s.name: s for s in (SHORT, LONG)}


def cpu_groups(jobs: int, threads: int) -> list[str]:
    """One group of `threads` CPUs per parallel job, from the CPUs this process may use. Groups overlap only when jobs x threads
    exceeds the CPUs available."""
    available = sorted(os.sched_getaffinity(0))
    return [",".join(str(available[(i * threads + k) % len(available)]) for k in range(threads)) for i in range(jobs)]


def _call(args: list[str], timeout: float, cpus: str | None) -> tuple[dict | None, str]:
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO))
    if cpus:
        env["MUTATIS_BENCHMARK_CPUS"] = cpus
    try:
        done = subprocess.run([sys.executable, "-m", "benchmarks.samplers.worker", *args], cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:.0f} s"
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):]), ""
    return None, (done.stderr or done.stdout)[-3000:]


def prepare_references(targets, jobs: int, threads: int, log):
    """Build the cached references of the model targets before runs start, in parallel with each other."""
    todo = [t for t in targets if t in MODEL_TARGETS]
    groups = cpu_groups(max(1, min(jobs, len(todo))), threads)
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {pool.submit(_call, ["--target", t, "--reference-only"], 7200, groups[i % len(groups)]): t for i, t in enumerate(todo)}
        for future in as_completed(futures):
            result, problem = future.result()
            log(f"reference {futures[future]}: " + (f"{result['seconds']:.0f} s" if result else f"FAILED {problem}"))


def _log(text: str):
    print(text, flush=True)   # flushed, so a log file shows progress while the suite runs


def run(run_dir: Path, suite: Suite, jobs: int, threads: int = 1, limit: int | None = None, log=_log) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path, info_path = run_dir / "results.jsonl", run_dir / "run.json"
    task_list = suite.tasks()[:limit] if limit else suite.tasks()
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["target"], r["method"], r["seed"]))
    todo = [t for t in task_list if (t["target"], t["method"], t["seed"]) not in done]
    info = json.loads(info_path.read_text()) if info_path.exists() else {"suite": suite.name, "description": suite.description, "sessions": []}
    info |= {"planned_runs": len(task_list), "jobs": jobs, "threads_per_job": threads, "cpus_available": len(os.sched_getaffinity(0))}
    session = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": len(todo)}
    log(f"suite {suite.name}: {len(task_list)} runs, {len(done)} already recorded, {len(todo)} to do, {jobs} at a time, {threads} CPU(s) each")
    started = time.perf_counter()
    prepare_references(sorted({t["target"] for t in todo}), jobs, threads, log)
    lock = threading.Lock()
    free = queue.Queue()
    for group in cpu_groups(jobs, threads):
        free.put(group)

    def one(task):
        cpus = free.get()
        try:
            result, problem = _call(["--target", task["target"], "--method", task["method"], "--seed", str(task["seed"])], suite.timeout, cpus)
        finally:
            free.put(cpus)
        if result is None:
            result = task | {"status": "failed", "error": problem, "posterior_ok": False, "confidently_wrong": False}
        result |= {"jobs": jobs, "suite": suite.name}
        with lock:
            with path.open("a") as f:
                f.write(json.dumps(result) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(one, t) for t in todo]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            status = r["status"] if r["status"] != "ok" else ("accurate" if r["posterior_ok"] else "INACCURATE")
            log(f"[{i}/{len(todo)}, {time.perf_counter() - started:.0f} s] {r['target']} {r['method']} seed {r['seed']}: {status}"
                + (f", {r['seconds']:.1f} s" if "seconds" in r else "") + (f" ({str(r.get('error', ''))[:200]})" if r["status"] != "ok" else ""))
    session |= {"end": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": time.perf_counter() - started}
    info["sessions"].append(session)
    info["total_seconds"] = sum(s["seconds"] for s in info["sessions"])
    info_path.write_text(json.dumps(info, indent=1))
    return path
