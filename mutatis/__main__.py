"""The one generic command: python -m mutatis {simulate, run, compare}.

    python -m mutatis simulate MODEL.yaml --params PARAMS.yaml --periods 200 --out data.csv
    python -m mutatis run MODEL.yaml --data data.csv [--edits EDITS.yaml] [--store results] [--seed 0] [--particles 2000]
    python -m mutatis compare results/KEY_A results/KEY_B

Every model goes through the same commands; nothing here knows what kind of model it is.
"""

import argparse
import sys
from pathlib import Path

import yaml


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mutatis", description="Estimate a model, change it, compare the versions.")
    commands = parser.add_subparsers(dest="command", required=True)
    sim = commands.add_parser("simulate", help="simulate data from a model at given parameter values")
    sim.add_argument("model")
    sim.add_argument("--params", required=True, help="YAML mapping from parameter targets to values")
    sim.add_argument("--periods", type=int, default=200)
    sim.add_argument("--first-date", default="2000Q1")
    sim.add_argument("--outside", help="CSV of outside variables (date column, then one column per data series that feeds one)")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--out", required=True)
    est = commands.add_parser("run", help="estimate a model on data and store the result")
    est.add_argument("model")
    est.add_argument("--data", required=True, help="a CSV file, or a YAML data manifest")
    est.add_argument("--edits", action="append", default=[], help="YAML list of edits to apply first; may be given more than once")
    est.add_argument("--report", help="YAML list of quantities to summarise")
    est.add_argument("--store", default="results")
    est.add_argument("--seed", type=int, default=0)
    est.add_argument("--particles", type=int, help="SMC particles")
    est.add_argument("--draws", type=int, help="draws from the exact formula")
    est.add_argument("--route", default="auto", choices=["auto", "read_off", "kalman", "sample"])
    est.add_argument("--method", default="auto", choices=["auto", "smc", "nuts", "exact_regression"])
    cmp_ = commands.add_parser("compare", help="compare two stored results")
    cmp_.add_argument("first")
    cmp_.add_argument("second")
    args = parser.parse_args(argv)

    from mutatis.spec.yaml_io import load_model
    from mutatis.versions import run as runs

    if args.command == "compare":
        print(runs.compare(runs.load_result(args.first), runs.load_result(args.second)))
        return 0
    model = load_model(Path(args.model).read_text())
    if args.command == "simulate":
        import jax
        import jax.numpy as jnp
        import numpy as np

        from mutatis.compile.system import compile_model
        from mutatis.measure.data import Dataset
        from mutatis.quantities.simulate import simulate_data
        from mutatis.versions.pipeline import solver

        system = compile_model(model)
        params = {k: jnp.asarray(v, dtype=float) for k, v in yaml.safe_load(Path(args.params).read_text()).items()}
        missing = [k for k, p in system.parameters.items() if p.kind != "fixed" and k not in params]
        if missing:
            print(f"give values for: {', '.join(missing)}", file=sys.stderr)
            return 2
        solved = solver(system)({**system.fixed_parameters(), **params})
        outside = None
        if args.outside:
            given = Dataset.read_csv(args.outside)
            outside = np.stack([given.series[system.outside[key].series[label]] for key in system.outside for label in
                                [system.outside_labels[i] for i in system.layout.outside[key].positions.reshape(-1)]], axis=1)[:args.periods]
        start = None if solved.initial_gaussian() is not None else np.zeros(system.n_variables)
        simulate_data(system, solved, jax.random.key(args.seed), args.periods, args.first_date, outside, start).write_csv(args.out)
        print(f"wrote {args.periods} periods to {args.out}")
        return 0
    from mutatis.measure.data import Dataset
    from mutatis.measure.manifest import load_manifest
    from mutatis.spec.edits import load_edit_file

    edits = [e for path in args.edits for e in load_edit_file(path)]
    report = tuple(yaml.safe_load(Path(args.report).read_text())) if args.report else ()
    dataset = load_manifest(args.data) if args.data.endswith((".yaml", ".yml")) else Dataset.read_csv(args.data)
    options = {k: v for k, v in {"n_particles": args.particles, "n_draws": args.draws}.items() if v is not None}
    result = runs.run(model, dataset, store=args.store, edits=edits, seed=args.seed, route=args.route, method=args.method, report=report, **options)
    print(f"stored in {result.folder}  ({result.run['method']}, {result.run['route']})")
    shown = [(n, e) for n, e in result.summary["parameters"].items() if "mean" in e]
    for name, entry in shown[:30]:
        print(f"  {name:<28} {entry['mean']:.4f}  ({entry['q05']:.4f} to {entry['q95']:.4f})")
    if len(shown) > 30:
        print(f"  ... and {len(shown) - 30} more parameters in {result.folder / 'summary.json'}")
    for quantity, stats in result.summary.get("quantities", {}).items():
        if "mean" in stats:
            print(f"  {quantity}: mean {', '.join(f'{v:.3f}' for v in stats['mean'][:6])} ...")
        else:
            print(f"  {quantity}: set of means [{', '.join(f'{a:.3f}' for a in stats['set_mean_lower'][:4])} ...] to [{', '.join(f'{b:.3f}' for b in stats['set_mean_upper'][:4])} ...]")
    if "plausibility" in result.summary:
        print(f"  share of draws where the restrictions can hold: {result.summary['plausibility']:.3f}")
    if result.summary["log_marginal_likelihood"] is not None:
        print(f"  log marginal likelihood: {result.summary['log_marginal_likelihood']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
