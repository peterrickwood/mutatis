"""How SMC's answers degrade with the number of unknowns, on Gaussian targets whose posterior and marginal likelihood are known.

The target is `tests.infer.targets.NormalMean` with d unknowns (prior N(0, 10^2), 50 observations each). For each d and seed it
prints the error in the log marginal likelihood, the worst error in a posterior mean (in posterior standard deviations), the
ratio of the estimated to the true posterior standard deviation, the number of temperature steps and the mean acceptance rate.

Run: .venv/bin/python experiments/smc_dimension_study.py 1,5,20,50,100,200 3 [setting=value ...]
where the settings are smc() keyword arguments, e.g. n_mcmc_steps=20.

With the defaults on 2026-11-26 (decision 38): errors within 0.4 up to d = 50, then +27 to +29 at d = 100 and +4 to +21 at
d = 200, while posterior means stayed within 0.08 posterior sds.
"""

import sys
import time

import jax
import numpy as np

from mutatis.infer.smc import smc
from tests.infer.targets import NormalMean


def main():
    dims = [int(d) for d in sys.argv[1].split(",")]
    seeds = int(sys.argv[2])
    settings = {}
    for setting in sys.argv[3:]:
        key, value = setting.split("=")
        settings[key] = float(value) if "." in value else int(value) if value.isdigit() else value
    print("settings:", settings or "defaults")
    for d in dims:
        target = NormalMean(np.random.default_rng(d).normal(size=(50, d)) + np.linspace(-2, 2, d))
        post_sd = np.sqrt(target.post_var)
        rows = []
        for seed in range(seeds):
            start = time.perf_counter()
            result = smc(target, jax.random.key(seed), **settings)
            rows.append((result.log_marginal_likelihood - target.log_evidence, np.max(np.abs(result.mean() - target.post_mean)) / post_sd,
                         np.mean(result.sd()) / post_sd, len(result.diagnostics["temperatures"]) - 1, np.mean(result.diagnostics["acceptance"]),
                         time.perf_counter() - start))
        a = np.array(rows)
        print(f"d={d:4d}: log-evidence error {' '.join(f'{x:+7.2f}' for x in a[:, 0])}   worst mean error (posterior sds) "
              f"{' '.join(f'{x:5.2f}' for x in a[:, 1])}   sd ratio {' '.join(f'{x:5.3f}' for x in a[:, 2])}   steps "
              f"{' '.join(f'{int(x):3d}' for x in a[:, 3])}   acceptance {a[:, 4].mean():.2f}   {a[:, 5].mean():5.1f} s", flush=True)


if __name__ == "__main__":
    main()
