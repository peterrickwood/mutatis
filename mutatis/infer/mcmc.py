"""General MCMC, built from interchangeable parts: a step rule, a tempering ladder, independent replicas, starts, and a
stopping rule. By default each part is chosen from the problem itself and the hardware (decision 42), with always a ladder
(decision 43).

- **Step rule**: how one chain moves. `nuts` (the No-U-Turn Sampler: gradient-based, chooses its own trajectory length),
  `hmc` (gradient-based, random trajectory length up to `hmc_max_steps`), or `random_walk` (no gradients; `random_walk_steps`
  Metropolis steps per scan). Each rung tunes its own step size (dual averaging, as in Stan) and its own mass matrix, which
  stretches the steps to the posterior's shape: a full matrix (scales and correlations) up to `dense_limit` unknowns, one
  scale per unknown above that.
- **Starts**: `search` (the default) climbs from prior draws to the nearest peak, many at once: L-BFGS, then Newton steps
  until the end point is certified a peak (the density falls in every direction there: minus the Hessian is positive
  definite; and the predicted further rise, half the Newton decrement, is below 1e-8). End points that cannot be certified
  are not peaks and are left out. Certified peaks are told apart by the posterior density between them: two are separate
  when it dips by more than 2 log units on the straight line from one to the other. The search goes on until it has made
  enough starts to find, with 95% probability, any peak reached from at least 1 in 10,000 prior draws (ln 20 / 10^-4,
  about 30,000 starts) and the last round found no new peak, or until it has used 10% of `max_seconds`. Every result
  states the limit reached: with N starts, a peak reached from fewer than 1 in N / ln 20 prior draws could have been missed.
  The certificate needs the full Hessian, so there is no search above 200 unknowns. Chains then start from Pathfinder draws
  (Zhang, Carpenter, Gelman and Vehtari, JMLR 2022) spread over the peaks found. `pathfinder`: Pathfinder runs from prior
  draws, no search. `prior`: prior draws.
- **Ladder** (parallel tempering): chains at "inverse temperatures" beta from 0 to 1. The chain at beta targets
  reference^(1 - beta) x posterior^beta, so beta = 1 is the posterior and beta = 0 the reference: the prior, or a normal
  distribution fitted by Pathfinder ("fitted", needed when the prior is improper). The beta = 0 rung draws exactly from the
  reference. After every scan, neighbouring rungs propose to exchange their states, alternating even and odd pairs (the
  non-reversible scheme of Syed, Bouchard-Cote, Deligiannidis and Doucet, JRSS B 2022). During warm-up the betas are respaced
  so that every exchange is rejected about equally often, and with automatic rungs their number is set from the estimated
  "communication barrier" (the sum of rejection rates along the ladder; about twice it is enough).
  `rungs="auto"` (the default): always a ladder with an automatic number of rungs. `rungs="decide"`: no ladder unless the
  data call for one: the search found two or more peaks, or during warm-up the chains settle in separate peaks, or still
  disagree after 1,000 scans of warm-up; it was the default in the try-out of decision 42, and was confidently wrong on
  curved peaks, where it ran plain chains. A whole number fixes the rungs; 1 is ordinary MCMC.
- **Replicas and processes**: independent copies of the whole ladder (or independent chains). `replicas="auto"` chooses from
  the hardware: on a CPU, one replica per usable CPU (at least 4, at most 64), each group of replicas run in its own process
  pinned to its own CPU (`mutatis.numerics.parallel`; in one process, chains do not spread over the cores). With one usable
  CPU, 4 replicas in this process. On a GPU or TPU, 64 replicas in this process (a rule of thumb, not measured). The search's
  climbs and the Pathfinder runs are spread over the same processes. Every program is compiled once, here, and sent to the
  processes compiled.
- **Stopping**: never a fixed number of draws. Warm-up runs in windows (Stan's: a first window, slow windows doubling in
  length, a last window) and ends after a slow window in which the chains agree (R-hat below 1.05) and the ladder's rung
  count has settled. Sampling then runs in blocks, each at least half as long as all before it, until every unknown has
  rank-normalised split R-hat below 1.01, a bulk effective sample size of at least (1 / precision)^2 (so the Monte Carlo
  error of each posterior mean is at most `precision` posterior standard deviations), a tail effective sample size of at
  least 400 (Vehtari and others, 2021), and, with a ladder, at least 2 round trips per replica from the reference end to
  the posterior end and back. If `max_seconds` runs out first, the draws so far are returned with `converged` False, a
  `status` saying what was not met, and a `NotConvergedWarning`. At most 10,000 draws per chain are kept: past that the
  kept draws are thinned to every second one (and every second scan from then on), so a long run's memory stays bounded;
  the tests are then judged on the thinned draws, which only makes them stricter. Slow warm-up windows stop doubling at
  3,200 scans.

Decisions that change the sampler are all taken during warm-up; draws are kept only after the last change, since adapting
while keeping draws would break the guarantee that the draws come from the posterior.

NUTS as usually run is the setting `nuts(problem, key)`: one rung, chains from prior draws, 4 replicas.

The ladder's record (each rung's log-likelihood ratios) is kept so the evidence can be computed afterwards by stepping stones
(`mutatis.infer.evidence`); computing it is a separate, optional step.
"""

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import blackjax
import blackjax.vi.pathfinder
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.optimizers.lbfgs import minimize_lbfgs

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.convergence import bulk_ess, split_rhat, tail_ess
from mutatis.infer.posterior import Posterior
from mutatis.numerics.parallel import Local, Workers, compile_program, usable_cpus

STEPS = ("nuts", "hmc", "random_walk")
STARTS = ("search", "pathfinder", "prior")
TARGET_ACCEPTANCE = {"nuts": 0.8, "hmc": 0.8, "random_walk": 0.234}
RHAT_LIMIT = 1.01             # every unknown, over the kept draws
WARMUP_RHAT_LIMIT = 1.05      # over one warm-up window: the chains have found the same distribution
TAIL_ESS = 400                # Vehtari and others (2021)
ROUND_TRIPS_PER_REPLICA = 2   # with a ladder: states have travelled from the reference end to the posterior end and back
VALLEY = 2.0                  # log-density dip on the line between two points that puts them in different peaks
SEARCH_BATCH = 8              # climbs per compiled call (the time limit is checked between calls)
DETECTION = 1e-4              # the search aims to find, with 95% probability, any peak reached from 1 in 10,000 prior draws
DETECTION_CONFIDENCE = 0.95
SEARCH_SHARE = 0.1            # at most this share of max_seconds goes to the search
LBFGS_ITERATIONS, LBFGS_LINE_SEARCH = 200, 50
NEWTON_STEPS = 20             # Newton steps after L-BFGS, to certify a peak
PEAK_TOLERANCE = 1e-8         # certified when half the Newton decrement (the predicted further rise in log density) is below this
SAME_PEAK = 0.01              # certified points closer than this (squared distance in the peak's standard deviations) coincide
ESCALATE_AFTER = 1000         # warm-up scans in slow windows after which chains that still disagree get a ladder
DENSE_LIMIT = 200             # full mass matrix up to this many unknowns
SEARCH_LIMIT = 200            # no peak search above this many unknowns (the certificate needs the full Hessian)
CHUNK = 25                    # scans per compiled call; the time limit is checked between calls
FIRST_WINDOW, FIRST_SLOW_WINDOW, LAST_WINDOW = 75, 25, 50
FIRST_BLOCK, MIN_BLOCK = 100, 100
MAX_KEPT = 10_000             # draws kept per chain; past this the kept draws are thinned (every second one), so memory is bounded
MAX_WINDOW = 3200             # longest slow warm-up window, in scans
CPU_REPLICAS, ACCELERATOR_REPLICAS, MAX_PROCESSES = 4, 64, 64
PATHFINDER_RUNS = 4           # Pathfinder runs for the replicas' starts (more when the search found more peaks)
PATHFINDER_DRAWS = 200
NEVER, BOTTOM, TOP = -1, 0, 1


class MCMCError(RuntimeError):
    pass


class NotConvergedWarning(UserWarning):
    """The sampler stopped at its time limit before its convergence tests were met: its results are suspect."""


@dataclass(frozen=True)
class Reference:
    """The beta = 0 end of the ladder: a normalised density that can be drawn from exactly."""

    name: str
    log_density: Callable
    draw: Callable


@dataclass(frozen=True)
class Ladder:
    """What the evidence step needs: each rung's beta and, for every recorded scan, each rung's log ratio
    ell = log(prior x likelihood) - log(reference), with the reference normalised."""

    betas: np.ndarray   # (rungs,)
    ell: np.ndarray     # (replicas, draws, rungs)
    reference: str


def mcmc(problem, key, step: str = "nuts", rungs="auto", reference: str = "auto", replicas="auto", starts: str = "search",
         precision: float = 0.025, max_seconds: float = 3600.0, initial_rungs: int = 8, max_rungs: int = 64,
         hmc_max_steps: int = 10, random_walk_steps: int = 10, target_acceptance: float | None = None,
         dense_limit: int = DENSE_LIMIT, processes="auto") -> Posterior:
    """Sample a posterior problem (contract 2: `log_prior`, `log_likelihood`, `draw_prior`, `size`, `proper`) until the
    convergence tests are met or `max_seconds` (the whole run) is used up."""
    clock = time.monotonic()
    if step not in STEPS:
        raise MCMCError(f"unknown step rule {step!r}: use one of {', '.join(STEPS)}")
    if starts not in STARTS:
        raise MCMCError(f"unknown starts {starts!r}: use one of {', '.join(STARTS)}")
    if not getattr(problem, "gradient_ready", True):
        raise MCMCError("this problem has unknowns that cannot be moved (a rotation, or no prior); use the exact formula")
    if rungs not in ("decide", "auto") and (not isinstance(rungs, int) or rungs < 1):
        raise MCMCError("rungs is 'decide', 'auto' or a whole number of at least 1")
    if replicas != "auto" and (not isinstance(replicas, int) or replicas < 2):
        raise MCMCError("replicas is 'auto' or a whole number of at least 2 (agreement between replicas is a convergence test)")
    if processes != "auto" and (not isinstance(processes, int) or processes < 1):
        raise MCMCError("processes is 'auto' or a whole number of at least 1")
    if not precision > 0 or not max_seconds > 0:
        raise MCMCError("precision and max_seconds must be positive")
    proper = getattr(problem, "proper", True)
    if starts == "prior" and not proper:
        raise MCMCError("cannot start from prior draws: the prior is improper; use starts='search' or 'pathfinder'")
    reference = ("prior" if proper else "fitted") if reference == "auto" else reference
    if reference not in ("prior", "fitted"):
        raise MCMCError(f"unknown reference {reference!r}: use auto, prior or fitted")
    if reference == "prior" and not proper and rungs != 1:
        raise MCMCError("the ladder cannot end at an improper prior; use reference='fitted'")

    hw = hardware()
    R, n_processes = _plan(hw, replicas, processes)
    executor = Workers(usable_cpus()[:n_processes]) if n_processes > 1 else Local()   # worker processes start while this one compiles
    try:
        return _mcmc(problem, key, step, rungs, reference, starts, precision, max_seconds, initial_rungs, max_rungs, hmc_max_steps,
                     random_walk_steps, target_acceptance, dense_limit, clock, hw, R, _Runner(executor))
    finally:
        executor.close()


def _plan(hw: dict, replicas, processes) -> tuple[int, int]:
    """How many replicas, and over how many processes (each with an equal share)."""
    if hw["accelerator"]:
        return (ACCELERATOR_REPLICAS if replicas == "auto" else replicas), 1
    cpus = min(hw["usable_cpus"], MAX_PROCESSES) if processes == "auto" else processes
    R = max(CPU_REPLICAS, cpus) if replicas == "auto" else replicas
    return R, max(k for k in range(1, min(cpus, R) + 1) if R % k == 0)


def _mcmc(problem, key, step, rungs, reference, starts, precision, max_seconds, initial_rungs, max_rungs, hmc_max_steps,
          random_walk_steps, target_acceptance, dense_limit, clock, hw, R, runner) -> Posterior:
    log_target = lambda z: problem.log_prior(z) + problem.log_likelihood(z)
    logp_batch = jax.jit(jax.vmap(log_target))
    rule = _UntilConverged(precision, max_seconds, clock)
    climber = _Climber(problem, log_target, logp_batch, runner)
    key_search, key_starts, key_run, key_escalated = jax.random.split(key, 4)

    search, timings = None, {}
    tic = time.monotonic()
    if starts == "search":
        search = climber.search(key_search, min(rule.wall_deadline(), time.time() + SEARCH_SHARE * max_seconds))
        initial = _spread(search, R) if search.n_peaks else _initial_points(problem, key_starts, log_target, R)
    else:
        initial = _initial_points(problem, key_starts, log_target, R)
    timings["search"], tic = time.monotonic() - tic, time.monotonic()
    runs = min(R, max(PATHFINDER_RUNS, search.n_peaks if search else 0))
    start, pooled = _pathfinder(runner, key_starts, log_target, initial, runs, rule.wall_deadline()) if starts != "prior" else (jnp.asarray(initial), None)
    timings["pathfinder"] = time.monotonic() - tic

    if rungs == "decide":
        use_ladder = search is not None and search.n_peaks >= 2
        reason = f"the search found {search.n_peaks} separate peaks" if use_ladder else None
    else:
        use_ladder, reason = rungs != 1, ("always (the rungs setting)" if rungs != 1 else None)

    def build(ladder: bool) -> _Sampler:
        nonlocal pooled
        ref, ell_fn = None, None
        if ladder and reference == "prior":
            ref, ell_fn = Reference("prior", problem.log_prior, problem.draw_prior), problem.log_likelihood
        elif ladder:
            if pooled is None:
                _, pooled = _pathfinder(runner, key_starts, log_target, initial, min(R, PATHFINDER_RUNS), rule.wall_deadline())
            if pooled is None:
                raise MCMCError("Pathfinder failed from every start, so no reference can be fitted")
            ref = _fitted_reference(pooled)
            ell_fn = lambda z: log_target(z) - ref.log_density(z)
        accept = TARGET_ACCEPTANCE[step] if target_acceptance is None else target_acceptance
        return _Sampler(problem.size, step, ref, ell_fn, log_target, accept, hmc_max_steps, random_walk_steps,
                        dense=problem.size <= dense_limit, names=_unknown_names(problem), logp_batch=logp_batch, runner=runner,
                        climber=climber)

    scales = jnp.asarray(np.maximum(np.var(pooled, axis=0), 1e-12) if pooled is not None else np.ones(problem.size))
    automatic = rungs in ("decide", "auto")
    n_rungs = (initial_rungs if automatic else rungs) if use_ladder else 1
    result = build(use_ladder).run(key_run, start, scales, n_rungs, automatic, max_rungs, rule,
                                   may_escalate=rungs == "decide" and not use_ladder)
    earlier = None
    if isinstance(result, _Escalation):
        earlier, reason = result, result.reason
        scales = jnp.maximum(result.scales, 1e-12)
        result = build(True).run(key_escalated, result.positions, scales, initial_rungs, True, max_rungs, rule)
    d = result.diagnostics
    if earlier is not None:
        d["warmup"] = tuple(earlier.history) + tuple(d["warmup"])
        d["timings"]["warm-up"] += earlier.cost["seconds"]
        d["gradient_evaluations"] += earlier.cost["gradients"]
        d["density_evaluations"] += earlier.cost["densities"]
    found = search is not None and search.skipped is None
    d |= {"hardware": hw, "processes": len(runner), "starts": starts,
          "starts_used": search.n_starts if search else (runs if starts == "pathfinder" else 0), "peaks_found": search.n_peaks if found else None,
          "search": search.summary() if search else None, "peak_search": _search_statement(search, starts, getattr(problem, "proper", True)),
          "ladder_used": d["rungs"] > 1, "ladder_reason": reason, "escalated": earlier is not None, "seconds": time.monotonic() - clock,
          "timings": timings | d["timings"]}
    if not d["converged"]:
        warnings.warn(f"NOT CONVERGED — RESULTS ARE SUSPECT: {d['status']}", NotConvergedWarning, stacklevel=3)
    return result


def nuts(problem, key, n_chains: int = 4, precision: float = 0.025, max_seconds: float = 3600.0, processes="auto") -> Posterior:
    """NUTS as usually run: independent chains started from prior draws, no ladder. A named setting of `mcmc`."""
    return mcmc(problem, key, step="nuts", rungs=1, replicas=n_chains, starts="prior", precision=precision, max_seconds=max_seconds,
                processes=processes)


def hardware() -> dict:
    """What the sampler runs on: the platform JAX uses, its device count, and the CPUs this process may use."""
    devices = jax.devices()
    platform = devices[0].platform
    return {"platform": platform, "devices": len(devices), "usable_cpus": len(usable_cpus()), "accelerator": platform != "cpu"}


def _unknown_names(problem) -> tuple[str, ...]:
    layout = getattr(problem, "layout", None)
    if layout is None or not hasattr(layout, "unknowns"):
        return tuple(f"z[{i}]" for i in range(problem.size))
    names = []
    for u in layout.unknowns:
        names += [u.name] if u.size == 1 else [f"{u.name}[{i}]" for i in range(u.size)]
    return tuple(names) if len(names) == problem.size else tuple(f"z[{i}]" for i in range(problem.size))


class _Runner:
    """Compiled programs run through an executor: worker processes, or this process (`mutatis.numerics.parallel`). Each
    program is compiled once, in this process."""

    def __init__(self, executor):
        self.executor = executor

    def __len__(self) -> int:
        return len(self.executor)

    def ensure(self, name, fn, *example_args):
        """Compile `fn` for arguments shaped like the examples and hand it to the executor, unless done already."""
        if not self.executor.loaded(name):
            self.executor.load(name, compile_program(fn, *example_args, use_cache=not self.executor.sends_programs))

    def map(self, name, inputs: list, deadline: float) -> list:
        """One call of a program without state per input (a list of arrays), spread over the processes in turn. Returns
        each call's outputs, or None for calls not reached before `deadline` (wall-clock time)."""
        W = len(self.executor)
        groups = [list(range(k, len(inputs), W)) for k in range(W)]
        results = self.executor.iterate(name, [[] for _ in range(W)], [[inputs[i] for i in g] for g in groups], deadline)
        out = [None] * len(inputs)
        for g, (_, outputs) in zip(groups, results):
            for i, o in zip(g, outputs):
                out[i] = o
        return out


# ---------------------------------------------------------------------------------------------------------------- starts

def _initial_points(problem, key, log_target, n: int) -> np.ndarray:
    """n points to start from: prior draws with a finite posterior density, or standard normal draws if the prior is improper."""
    if getattr(problem, "proper", True):
        return np.asarray(_finite_prior_draws(problem, key, log_target, n))
    return np.asarray(jax.random.normal(key, (n, problem.size)))


def _finite_prior_draws(problem, key, log_target, n: int, tries: int = 50):
    """n prior draws at which the posterior density is finite (the model may have no solution at some prior draws)."""
    candidates = jax.vmap(problem.draw_prior)(jax.random.split(key, n * tries))
    finite = np.isfinite(np.asarray(jax.vmap(log_target)(candidates)))
    if finite.sum() < n:
        raise MCMCError(f"only {finite.sum()} of {n * tries} prior draws give a finite posterior density; need {n}")
    return candidates[np.flatnonzero(finite)[:n]]


def _pathfinder_one(log_target):
    """One Pathfinder run from x0 (a key given as its raw data): its normal approximation's draws and their log densities."""
    def one(raw_key, x0):
        k_fit, k_draw = jax.random.split(jax.random.wrap_key_data(raw_key))
        state, _ = blackjax.vi.pathfinder.approximate(k_fit, log_target, x0, num_samples=PATHFINDER_DRAWS)
        draws, _ = blackjax.vi.pathfinder.sample(k_draw, state, PATHFINDER_DRAWS)
        return draws, jax.vmap(log_target)(draws)
    return one


def _pathfinder(runner, key, log_target, initial, runs: int, deadline: float):
    """Pathfinder runs from the first `runs` initial points, spread over the processes. Replica r starts from a draw of run
    r mod runs (successive replicas on the same run take successive draws), or from its initial point if that run failed.
    Also returns all the runs' draws pooled, for initial scales and a fitted reference."""
    initial = np.asarray(initial)
    raw = np.asarray(jax.random.key_data(jax.random.split(key, runs)))
    runner.ensure("pathfinder", _pathfinder_one(log_target), raw[0], initial[0])
    found, pooled = [], []
    for out in runner.map("pathfinder", [[raw[i], initial[i]] for i in range(runs)], deadline):
        draws, values = out if out is not None else (np.zeros((0, initial.shape[1])), np.zeros(0))
        ok = np.all(np.isfinite(draws), axis=1) & np.isfinite(values)
        found.append(draws[ok])
        if ok.any():
            pooled.append(draws[ok])
    starts = []
    for r in range(len(initial)):
        draws = found[r % runs]
        starts.append(draws[(r // runs) % len(draws)] if len(draws) else initial[r])
    return jnp.asarray(np.stack(starts)), (np.concatenate(pooled) if pooled else None)


def _fitted_reference(draws, variance_inflation: float = 4.0) -> Reference:
    """A normal distribution with the mean and (inflated) covariance of Pathfinder's draws. Inflated because Pathfinder,
    like other optimisation-based approximations, tends to be too narrow, and a reference narrower than the posterior makes
    other peaks harder to reach."""
    mean = np.mean(draws, axis=0)
    cov = variance_inflation * (np.atleast_2d(np.cov(draws, rowvar=False)) + 1e-9 * np.eye(draws.shape[1]))
    chol = jnp.asarray(np.linalg.cholesky(cov))
    mean = jnp.asarray(mean)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
    d = draws.shape[1]

    def log_density(z):
        u = jax.scipy.linalg.solve_triangular(chol, z - mean, lower=True)
        return -0.5 * jnp.sum(u**2) - 0.5 * log_det - 0.5 * d * jnp.log(2 * jnp.pi)

    return Reference("fitted", log_density, lambda k: mean + chol @ jax.random.normal(k, (d,)))


# ------------------------------------------------------------------------------------------------------------ peak search

def _climb_one(log_target):
    """Climbs from x0 to the nearest peak: L-BFGS, then Newton steps (each the full step, halved until the density rises by at
    least a quarter of the rise the curvature predicts) until the end point is certified a peak or a step fails. Certified:
    minus the Hessian is positive definite (the density falls in every direction) and half the Newton decrement (the further
    rise predicted) is below `PEAK_TOLERANCE`. Returns the end point, its log density, whether it is certified, and the peak's
    standard deviations there (square roots of the diagonal of minus the inverse Hessian)."""
    grad, hess = jax.grad(log_target), jax.hessian(log_target)
    fractions = 0.5 ** jnp.arange(12)

    def newton(x):
        g, H = grad(x), hess(x)
        L = jnp.linalg.cholesky(-0.5 * (H + H.T))           # not finite unless minus the Hessian is positive definite
        direction = jax.scipy.linalg.cho_solve((L, True), g)
        return direction, g @ direction, jnp.all(jnp.isfinite(L)), L

    def climb(x0):
        x = minimize_lbfgs(lambda x: -log_target(x), x0, maxiter=LBFGS_ITERATIONS, maxls=LBFGS_LINE_SEARCH)[0].params

        def unfinished(s):
            x, fx, direction, decrement, concave, L, k, stuck = s
            return concave & (decrement / 2 >= PEAK_TOLERANCE) & (k < NEWTON_STEPS) & ~stuck

        def step(s):
            x, fx, direction, decrement, concave, L, k, stuck = s
            trial = x + fractions[:, None] * direction
            f_trial = jax.vmap(log_target)(trial)
            good = jnp.isfinite(f_trial) & (f_trial >= fx + 0.25 * fractions * decrement)
            i = jnp.argmax(good)
            x, fx = jnp.where(good[i], trial[i], x), jnp.where(good[i], f_trial[i], fx)
            return (x, fx) + newton(x) + (k + 1, ~good[i])

        state = (x, log_target(x)) + newton(x) + (0, False)
        x, fx, _, decrement, concave, L, _, _ = jax.lax.while_loop(unfinished, step, state)
        certified = concave & (decrement / 2 < PEAK_TOLERANCE) & jnp.isfinite(fx)
        sd = jnp.sqrt(jnp.diagonal(jax.scipy.linalg.cho_solve((L, True), jnp.eye(x.shape[0]))))
        return x, fx, certified, sd

    return jax.vmap(climb)


def _valley_between(a, b, value_a, value_b, logp_batch, valley: float = VALLEY, n_points: int = 20) -> bool:
    """True if the log density falls more than `valley` below the lower end somewhere on the straight line from a to b, or
    is not finite there (for instance where the model has no solution)."""
    t = np.linspace(0.0, 1.0, n_points + 2)[1:-1, None]
    values = np.asarray(logp_batch(jnp.asarray(a + t * (b - a))))
    return bool(np.any(~np.isfinite(values)) or np.min(values) < min(value_a, value_b) - valley)


class _Peaks:
    """Points grouped into peaks, each represented by its highest point. Points are taken highest first. A point joins the
    first peak (highest first) that it coincides with (only when standard deviations are given: within `SAME_PEAK` of the
    representative, in the representative's standard deviations) or that it reaches without a valley on the straight line;
    otherwise it starts a new peak."""

    def __init__(self, logp_batch, valley: float = VALLEY):
        self.logp_batch, self.valley = logp_batch, valley
        self.points, self.values, self.sds = [], [], []

    def add(self, points, values=None, sds=None) -> tuple[np.ndarray, int]:
        """Add points; returns the peak of each (-1 where the density is not finite) and how many new peaks they revealed."""
        points = np.asarray(points, dtype=float).reshape(-1, np.shape(points)[-1])
        values = np.asarray(self.logp_batch(jnp.asarray(points)) if values is None else values, dtype=float)
        labels, before = np.full(len(points), -1), len(self.points)
        finite = np.flatnonzero(np.isfinite(values))
        for i in finite[np.argsort(-values[finite])]:
            order = sorted(range(len(self.points)), key=lambda g: -self.values[g])
            near = [g for g in order if self.sds[g] is not None and np.sum(((points[i] - self.points[g]) / self.sds[g]) ** 2) < SAME_PEAK]
            g = near[0] if near else next((g for g in order if not _valley_between(self.points[g], points[i], self.values[g], values[i],
                                                                                    self.logp_batch, self.valley)), None)
            if g is None:
                g = len(self.points)
                self.points.append(points[i]), self.values.append(float(values[i])), self.sds.append(None if sds is None else sds[i])
            elif values[i] > self.values[g]:
                self.points[g], self.values[g], self.sds[g] = points[i], float(values[i]), (None if sds is None else sds[i])
            labels[i] = g
        return labels, len(self.points) - before


def _group_peaks(points: np.ndarray, logp_batch, valley: float = VALLEY) -> tuple[np.ndarray, int]:
    """Labels of the peaks the points lie in (-1 where the density is not finite), and the number of peaks, by the valley
    test alone."""
    peaks = _Peaks(logp_batch, valley)
    labels, n = peaks.add(points)
    return labels, n


@dataclass
class _Search:
    n_starts: int                 # climbs made
    n_certified: int              # climbs that ended at a certified peak
    values: np.ndarray            # (peaks,): each peak's highest log density
    members: list                 # per peak: points from which a climb reached it
    seconds: float
    stopped: str                  # why the search stopped
    skipped: str | None = None    # why there was no search, if there was none

    @property
    def n_peaks(self) -> int:
        return len(self.values)

    @property
    def detection_limit(self) -> float | None:
        """The smallest share of starting draws from which a peak must be reached to be found with 95% probability."""
        return float(np.log(1.0 / (1.0 - DETECTION_CONFIDENCE)) / self.n_starts) if self.n_starts else None

    def summary(self) -> dict:
        return {"starts": self.n_starts, "certified": self.n_certified, "peaks": self.n_peaks, "peak_log_densities": self.values.tolist(),
                "detection_limit": self.detection_limit, "seconds": self.seconds, "stopped": self.stopped, "skipped": self.skipped}


def _search_statement(search: "_Search | None", starts: str, proper: bool) -> str:
    """What the peak search can and cannot vouch for, in a sentence."""
    if search is None:
        return f"no peak search (starts='{starts}'): peaks the chains did not reach could have been missed"
    if search.skipped:
        return f"no peak search ({search.skipped}): peaks the chains did not reach could have been missed"
    draws = "prior draws" if proper else "standard normal draws"
    found = (f"{search.n_peaks} peak{'s' if search.n_peaks != 1 else ''} found" if search.n_certified
             else "no climb ended at a certified peak")
    return (f"searched {search.n_starts:,} starts ({found}): a peak reached from fewer than 1 in "
            f"{1.0 / search.detection_limit:,.0f} {draws} could have been missed")


class _Climber:
    """Climbs to peaks (`_climb_one`) through the runner, and groups the certified ones."""

    def __init__(self, problem, log_target, logp_batch, runner):
        self.problem, self.log_target, self.logp_batch, self.runner = problem, log_target, logp_batch, runner
        self.d, self.proper = problem.size, getattr(problem, "proper", True)
        self._draw = jax.jit(jax.vmap(problem.draw_prior)) if self.proper else None

    def _climb(self, batches: list, deadline: float) -> list:
        self.runner.ensure("climb", _climb_one(self.log_target), np.zeros((SEARCH_BATCH, self.d)))
        return self.runner.map("climb", [[b] for b in batches], deadline)

    def _starting_points(self, key) -> np.ndarray:
        if self.proper:
            return np.asarray(self._draw(jax.random.split(key, SEARCH_BATCH)))
        return np.asarray(jax.random.normal(key, (SEARCH_BATCH, self.d)))

    def search(self, key, deadline: float, target: int | None = None, keep: int = 64) -> _Search:
        """Climbs from starting draws, in rounds spread over the processes, until `target` starts (enough for the detection
        limit) are made and the last round found no new peak, or until `deadline` (wall-clock time)."""
        began = time.monotonic()
        if self.d > SEARCH_LIMIT:
            return _Search(0, 0, np.zeros(0), [], 0.0, "skipped",
                           f"{self.d} unknowns, more than {SEARCH_LIMIT}; certifying a peak needs the full Hessian")
        target = target or int(np.ceil(np.log(1.0 / (1.0 - DETECTION_CONFIDENCE)) / DETECTION / SEARCH_BATCH)) * SEARCH_BATCH
        peaks, members, n_starts, n_certified = _Peaks(self.logp_batch), [], 0, 0
        per_round, most = len(self.runner), 64 * len(self.runner)
        while True:
            batches = []
            for _ in range(per_round):
                key, k = jax.random.split(key)
                batches.append(self._starting_points(k))
            ends, starts_reached = [], []
            for batch, out in zip(batches, self._climb(batches, deadline)):
                if out is None:
                    continue
                x, value, certified, sd = out
                n_starts, n_certified = n_starts + len(batch), n_certified + int(certified.sum())
                ends.append((x[certified], value[certified], sd[certified]))
                starts_reached.append(batch[certified])
            x, value, sd = (np.concatenate(a) for a in zip(*ends)) if ends else (np.zeros((0, self.d)), np.zeros(0), np.zeros((0, self.d)))
            labels, new = peaks.add(x, value, sd)
            for start, label in zip(np.concatenate(starts_reached) if starts_reached else [], labels):
                members += [[] for _ in range(label + 1 - len(members))]
                if len(members[label]) < keep:
                    members[label].append(start)
            if time.time() > deadline:
                stopped = f"time limit for the search reached after {n_starts:,} starts"
                break
            if n_starts >= target and new == 0:
                stopped = f"{n_starts:,} starts made and the last {len(batches) * SEARCH_BATCH:,} found no new peak"
                break
            per_round = min(2 * per_round, most)
        return _Search(n_starts, n_certified, np.asarray(peaks.values), [np.asarray(m) for m in members], time.monotonic() - began, stopped)

    def count_peaks(self, points: np.ndarray, deadline: float) -> int:
        """How many separate certified peaks the climbs from these points reach."""
        points = np.asarray(points, dtype=float)
        padded = np.concatenate([points, np.repeat(points[-1:], -len(points) % SEARCH_BATCH, axis=0)])
        outs = [o for o in self._climb(list(padded.reshape(-1, SEARCH_BATCH, self.d)), deadline) if o is not None]
        if not outs:
            return 0
        x, value, certified, sd = (np.concatenate(a) for a in zip(*outs))
        return _Peaks(self.logp_batch).add(x[certified], value[certified], sd[certified])[1]


def _spread(search: _Search, n: int) -> np.ndarray:
    """n starting points for Pathfinder, taken round robin over the peaks (highest first), each a point from which a climb
    reached that peak."""
    order = np.argsort(-search.values)
    return np.stack([search.members[order[r % len(order)]][(r // len(order)) % len(search.members[order[r % len(order)]])] for r in range(n)])


# ---------------------------------------------------------------------------------------------------------------- warm-up

def _windows(n_chunks: int) -> list[tuple[int, str]]:
    """A fixed warm-up schedule in chunks (Stan's windows: a first window, slow windows doubling in length, a last window),
    used by the fixed-length rule that tests the machinery."""
    if n_chunks < 3:
        return [(n_chunks, "slow")]
    first, last = max(1, round(0.075 * n_chunks)), max(1, round(0.05 * n_chunks))
    slow, length, windows = n_chunks - first - last, 1, [(first, "first")]
    while slow > 0:
        this = slow if slow < 3 * length else length
        windows.append((this, "slow"))
        slow -= this
        length *= 2
    return windows + [(last, "last")]


def _respace(betas: np.ndarray, rejection: np.ndarray, n_new: int):
    """New betas spaced evenly in the cumulative rejection rate (the communication barrier), and the barrier itself."""
    barrier = np.concatenate([[0.0], np.cumsum(np.maximum(rejection, 1e-6))])
    targets = np.linspace(0.0, barrier[-1], n_new)
    new = np.interp(targets, barrier, betas)
    new[0], new[-1] = 0.0, 1.0
    return new, float(np.sum(rejection)), barrier


def _welford_final(count, total, second, dense: bool):
    """Stan's regularised (co)variance from sums of draws (centred at a fixed point): shrunk towards 1e-3 when there are few."""
    n = jnp.maximum(count, 1.0)
    mean = total / n
    correction = n / jnp.maximum(n - 1.0, 1.0)
    if dense:
        cov = (second / n - mean[..., :, None] * mean[..., None, :]) * correction
        eye = jnp.eye(mean.shape[-1])
        return (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * eye
    variance = jnp.maximum(second / n - mean**2, 0.0) * correction
    return (n / (n + 5.0)) * variance + 1e-3 * (5.0 / (n + 5.0))


# ---------------------------------------------------------------------------------------------------------- stopping rules

def _convergence(draws: np.ndarray, trips: int, n_rungs: int, precision: float, names=None) -> dict:
    """The convergence tests on the kept draws (chains, draws, unknowns): which are met, and the numbers behind them."""
    R, n, d = draws.shape
    names = names or tuple(f"z[{i}]" for i in range(d))
    if n < 4:
        return {"converged": False, "findings": [f"only {n} draws per chain"], "max_rhat": float("inf"), "min_ess": 0.0, "min_tail_ess": 0.0}
    rhat, ess, tail = split_rhat(draws), bulk_ess(draws), tail_ess(draws)
    rhat, ess, tail = np.where(np.isnan(rhat), np.inf, rhat), np.nan_to_num(ess, nan=0.0), np.nan_to_num(tail, nan=0.0)
    need = (1.0 / precision) ** 2
    unmet = []
    if np.max(rhat) >= RHAT_LIMIT:
        unmet.append(f"R-hat {np.max(rhat):.3f} for {names[int(np.argmax(rhat))]} (want below {RHAT_LIMIT}; "
                     f"{int(np.sum(rhat >= RHAT_LIMIT))} of {d} unknowns above)")
    if np.min(ess) < need:
        unmet.append(f"bulk effective sample size {np.min(ess):.0f} for {names[int(np.argmin(ess))]} (want at least {need:.0f})")
    if np.min(tail) < TAIL_ESS:
        unmet.append(f"tail effective sample size {np.min(tail):.0f} for {names[int(np.argmin(tail))]} (want at least {TAIL_ESS})")
    if n_rungs > 1 and trips < ROUND_TRIPS_PER_REPLICA * R:
        unmet.append(f"{trips} round trips between the reference and the posterior (want at least {ROUND_TRIPS_PER_REPLICA * R})")
    return {"converged": not unmet, "findings": unmet, "max_rhat": float(np.max(rhat)), "min_ess": float(np.min(ess)),
            "min_tail_ess": float(np.min(tail))}


class _UntilConverged:
    """The stopping rule of the public sampler: open-ended warm-up and sampling, judged by the convergence tests, with a
    deadline for the whole run."""

    def __init__(self, precision: float, max_seconds: float, clock: float):
        self.precision, self.max_seconds, self.deadline = precision, max_seconds, clock + max_seconds

    def out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def wall_deadline(self) -> float:
        return time.time() + (self.deadline - time.monotonic())

    def next_window(self, history: list[dict]):
        if not history:
            return FIRST_WINDOW, "first"
        last = history[-1]
        if last["window"] == "first":
            return FIRST_SLOW_WINDOW, "slow"
        if last["window"] == "last":
            return None
        slow = sum(h["window"] == "slow" for h in history)
        settled = slow >= 2 and last["rhat"] < WARMUP_RHAT_LIMIT and not last.get("rungs_changed", False)
        return (LAST_WINDOW, "last") if settled else (min(2 * last["planned"], MAX_WINDOW), "slow")

    def first_block(self) -> int:
        return FIRST_BLOCK

    def next_block(self, n_kept: int, verdict: dict) -> int:
        return max(MIN_BLOCK, n_kept // 2)

    def verdict(self, draws, trips, n_rungs, names) -> dict:
        return _convergence(draws, trips, n_rungs, self.precision, names)

    def status(self, verdict: dict, phase: str, n_kept: int) -> str:
        if verdict["converged"]:
            return f"converged after {n_kept} draws per chain"
        cause = f"time limit of {self.max_seconds:g} s reached during {phase}" if self.out_of_time() else f"stopped during {phase}"
        return f"NOT CONVERGED: {cause} with {n_kept} draws per chain; " + "; ".join(verdict["findings"])


class _FixedLength:
    """A fixed warm-up and draw count. Used only by the unit tests of the ladder's machinery, which compare against formulas
    for fixed lengths; the public sampler never stops on a count."""

    def __init__(self, n_warmup: int, n_draws: int, precision: float = 0.025):
        self.windows = [(n * CHUNK, kind) for n, kind in _windows(max(1, -(-n_warmup // CHUNK)))]
        self.n_draws, self.precision = n_draws, precision

    def out_of_time(self) -> bool:
        return False

    def wall_deadline(self) -> float:
        return float("inf")

    def next_window(self, history):
        return self.windows[len(history)] if len(history) < len(self.windows) else None

    def first_block(self) -> int:
        return self.n_draws

    def next_block(self, n_kept, verdict):
        return None

    def verdict(self, draws, trips, n_rungs, names) -> dict:
        return _convergence(draws[:, :self.n_draws], trips, n_rungs, self.precision, names)

    def status(self, verdict, phase, n_kept) -> str:
        return f"fixed length ({self.n_draws} draws per chain; for testing)"


# ---------------------------------------------------------------------------------------------------------------- sampler

def _take_replicas(carry, part: slice):
    """The carry of some of the replicas (the betas and the draw count are shared)."""
    z, step_size, inv_mass, da, betas, labels, last_end, trips, sums = carry
    sums = (sums[0], sums[1][part], sums[2][part], sums[3][part])
    return (z[part], step_size[part], inv_mass[part], jax.tree.map(lambda a: a[part], da), betas, labels[part], last_end[part],
            trips[part], sums)


def _join_replicas(parts):
    """The carries of groups of replicas joined back into one."""
    cat = lambda *xs: np.concatenate([np.asarray(x) for x in xs])
    column = lambda i: [p[i] for p in parts]
    first = parts[0]
    sums = (first[8][0],) + tuple(cat(*[p[8][i] for p in parts]) for i in (1, 2, 3))
    return (cat(*column(0)), cat(*column(1)), cat(*column(2)), jax.tree.map(cat, *column(3)), first[4], cat(*column(5)),
            cat(*column(6)), cat(*column(7)), sums)


class _Record:
    """What is kept from the scans: the posterior rung's draws and (with a ladder) every rung's log-likelihood ratio, thinned
    to at most `MAX_KEPT` per chain; and totals over every scan for the swap acceptance and the divergences."""

    def __init__(self, ladder: bool, divergences: bool, limit: int | None = MAX_KEPT):
        self.ladder, self.count_divergences, self.limit = ladder, divergences, limit
        self.top, self.ell, self.index = [], [], []
        self.stride, self.scans, self.accept_sum, self.divergences = 1, 0, 0.0, 0

    def add(self, outs):
        for o in outs:
            top = np.asarray(o["top"])
            index = self.scans + np.arange(top.shape[0])
            keep = index % self.stride == 0
            self.top.append(top[keep])
            self.index.append(index[keep])
            if self.ladder:
                self.ell.append(np.asarray(o["ell"])[keep])
                self.accept_sum = self.accept_sum + np.asarray(o["accept_prob"]).sum(axis=0)    # (replicas, rungs - 1)
            if self.count_divergences:
                self.divergences += int(np.sum(np.asarray(o["divergent"])[:, :, -1]))
            self.scans += top.shape[0]
        while self.limit is not None and sum(len(i) for i in self.index) > self.limit:
            self.stride *= 2
            keep = [i % self.stride == 0 for i in self.index]
            self.top = [t[k] for t, k in zip(self.top, keep)]
            self.ell = [e[k] for e, k in zip(self.ell, keep)] if self.ladder else []
            self.index = [i[k] for i, k in zip(self.index, keep)]

    def draws(self) -> np.ndarray:
        return np.swapaxes(np.concatenate(self.top), 0, 1)                                    # (replicas, kept, d)

    def ratios(self) -> np.ndarray:
        return np.swapaxes(np.concatenate(self.ell), 0, 1)                                    # (replicas, kept, rungs)

    def swap_acceptance(self) -> np.ndarray:
        return np.mean(self.accept_sum, axis=0) / self.scans


@dataclass
class _Escalation:
    """Warm-up without a ladder found that one is needed: where the chains are, their scales, why, and what it cost."""
    positions: jnp.ndarray
    scales: jnp.ndarray
    reason: str
    history: list
    cost: dict = field(default_factory=dict)


class _Sampler:
    def __init__(self, size, step, reference, ell_fn, log_target, target_acceptance, hmc_max_steps, random_walk_steps,
                 dense: bool = False, names=None, logp_batch=None, runner: _Runner | None = None, climber: _Climber | None = None):
        self.size, self.step, self.reference, self.ell_fn, self.log_target = size, step, reference, ell_fn, log_target
        self.da_init, self.da_update, self.da_final = dual_averaging_adaptation(target_acceptance)
        self.hmc_max_steps, self.random_walk_steps, self.dense = hmc_max_steps, random_walk_steps, dense
        self.names = names or tuple(f"z[{i}]" for i in range(size))
        self.logp_batch = logp_batch if logp_batch is not None else jax.jit(jax.vmap(log_target))
        self._chunk_fns, self._out_keys = {}, {}
        self.runner, self.climber = runner if runner is not None else _Runner(Local()), climber

    # one chain, one scan ---------------------------------------------------------------------------------------------
    def _tempered(self, beta, single: bool):
        """The density at one rung: reference^(1 - beta) x (prior x likelihood)^beta, written so that beta = 0 never
        multiplies an infinite log density. A lone rung is the posterior itself."""
        if single:
            return self.log_target
        log_ref, log_target = self.reference.log_density, self.log_target
        return lambda z: (1.0 - beta) * log_ref(z) + jnp.where(beta > 0, beta * log_target(z), 0.0)

    def _move(self, key, z, beta, step_size, inv_mass, single: bool):
        """One scan of the step rule at one rung. Returns the new position, the acceptance rate, whether the trajectory
        diverged, and the number of gradient and density evaluations. `inv_mass` is a vector (one scale per unknown) or a
        full matrix."""
        log_density = self._tempered(beta, single)
        if self.step == "random_walk":
            factor = jnp.linalg.cholesky(inv_mass) if inv_mass.ndim == 2 else None

            def one(carry, k):
                z, lp = carry
                k_prop, k_accept = jax.random.split(k)
                noise = jax.random.normal(k_prop, z.shape)
                proposal = z + step_size * (factor @ noise if factor is not None else jnp.sqrt(inv_mass) * noise)
                lp_new = log_density(proposal)
                alpha = jnp.where(jnp.isnan(lp_new - lp), 0.0, jnp.minimum(1.0, jnp.exp(lp_new - lp)))
                accept = jax.random.uniform(k_accept) < alpha
                return (jnp.where(accept, proposal, z), jnp.where(accept, lp_new, lp)), alpha
            (z, _), alphas = jax.lax.scan(one, (z, log_density(z)), jax.random.split(key, self.random_walk_steps))
            return z, jnp.mean(alphas), jnp.array(False), jnp.array(0), jnp.array(self.random_walk_steps + 1)
        state = blackjax.nuts.init(z, log_density) if self.step == "nuts" else blackjax.hmc.init(z, log_density)
        if self.step == "nuts":
            state, info = blackjax.nuts.build_kernel()(key, state, log_density, step_size, inv_mass)
        else:
            k_length, key = jax.random.split(key)
            length = jax.random.randint(k_length, (), 1, self.hmc_max_steps + 1)
            state, info = blackjax.hmc.build_kernel()(key, state, log_density, step_size, inv_mass, length)
        rate = jnp.where(jnp.isnan(info.acceptance_rate), 0.0, info.acceptance_rate)
        return state.position, rate, info.is_divergent, info.num_integration_steps + 1, jnp.array(0)

    # one chunk of scans, all replicas and rungs ------------------------------------------------------------------------
    def _chunk_fn(self, n_rungs: int, length: int):
        """A compiled call running `length` scans. Its switches `adapt` (tune step sizes by dual averaging) and `accumulate`
        (sum the draws for new mass matrices) are run-time arguments, so one program serves warm-up and sampling; the work
        of a switch that is off is skipped, not done and discarded."""
        cache_key = (n_rungs, length)
        if cache_key in self._chunk_fns:
            return self._chunk_fns[cache_key]
        moved = slice(1, n_rungs) if n_rungs > 1 else slice(0, 1)  # rung 0 of a ladder draws exactly from the reference
        single = n_rungs == 1
        move = jax.vmap(jax.vmap(lambda k, z, b, e, m: self._move(k, z, b, e, m, single)))
        ell_all = None if single else jax.vmap(jax.vmap(self.ell_fn))
        draw_ref = None if single else jax.vmap(self.reference.draw)
        dense = self.dense

        def scan_one(carry, inputs, adapt, accumulate):
            key, parity = inputs
            z, step_size, inv_mass, da, betas, labels, last_end, trips, sums = carry
            R = z.shape[0]
            k_move, k_ref, k_swap = jax.random.split(key, 3)
            sub = z[:, moved]
            keys = jax.random.split(k_move, R * sub.shape[1]).reshape(R, sub.shape[1])
            new, rate, divergent, grads, densities = move(keys, sub, jnp.broadcast_to(betas[moved], sub.shape[:2]), step_size[:, moved], inv_mass[:, moved])
            z = z.at[:, moved].set(new)
            def tune(state):  # step-size adaptation at the rungs that moved
                da, _ = state
                da_new = jax.vmap(jax.vmap(self.da_update))(jax.tree.map(lambda a: a[:, moved], da), rate)
                da = jax.tree.map(lambda a, b: a.at[:, moved].set(b), da, da_new)
                return da, jnp.exp(da.log_step_size)
            da, step_size = jax.lax.cond(adapt, tune, lambda state: state, (da, step_size))
            out = {"rate": rate, "divergent": divergent, "grads": jnp.sum(grads), "densities": jnp.sum(densities)}
            if n_rungs > 1:
                z = z.at[:, 0].set(draw_ref(jax.random.split(k_ref, R)))
                ell = ell_all(z)                                                    # (R, N)
                dbeta = betas[1:] - betas[:-1]
                log_a = dbeta * (ell[:, :-1] - ell[:, 1:])
                accept_prob = jnp.where(jnp.isnan(log_a), 0.0, jnp.minimum(1.0, jnp.exp(jnp.minimum(log_a, 0.0))))
                pairs = jnp.arange(n_rungs - 1)
                attempt = (pairs % 2) == parity
                swap = attempt & (jax.random.uniform(k_swap, accept_prob.shape) < accept_prob)
                shift = jnp.pad(swap, ((0, 0), (0, 1))).astype(int) - jnp.pad(swap, ((0, 0), (1, 0))).astype(int)
                perm = jnp.arange(n_rungs)[None, :] + shift                          # non-overlapping pairs
                z = jnp.take_along_axis(z, perm[:, :, None], axis=1)
                ell = jnp.take_along_axis(ell, perm, axis=1)
                labels = jnp.take_along_axis(labels, perm, axis=1)
                bottom, top = labels[:, 0], labels[:, -1]
                rows = jnp.arange(R)
                came_back = last_end[rows, bottom] == TOP
                trips = trips + came_back.astype(int)
                last_end = last_end.at[rows, bottom].set(BOTTOM)
                last_end = last_end.at[rows, top].set(jnp.where(last_end[rows, top] == BOTTOM, TOP, last_end[rows, top]))
                out |= {"accept_prob": accept_prob, "ell": ell, "densities": out["densities"] + R * n_rungs}
            def add(sums):  # sums of each rung's draws, centred at the window's starting positions, for its mass matrix
                count, origin, total, second = sums
                x = z - origin
                return count + 1.0, origin, total + x, second + (x[..., :, None] * x[..., None, :] if dense else x**2)
            sums = jax.lax.cond(accumulate, add, lambda sums: sums, sums)
            out["top"] = z[:, -1]
            return (z, step_size, inv_mass, da, betas, labels, last_end, trips, sums), out

        @jax.jit
        def chunk(carry, key, start, adapt, accumulate):
            keys = jax.random.split(key, length)
            parity = (start + jnp.arange(length)) % 2
            return jax.lax.scan(lambda c, x: scan_one(c, x, adapt, accumulate), carry, (keys, parity))

        self._chunk_fns[cache_key] = chunk
        return chunk

    # the whole run -----------------------------------------------------------------------------------------------------
    def run(self, key, start, scales, n_rungs, auto_rungs, max_rungs, rule, may_escalate: bool = False):
        """Warm-up and sampling under a stopping rule. Returns a Posterior, or an _Escalation when `may_escalate` and
        warm-up without a ladder finds that one is needed."""
        R, d = start.shape
        betas = np.array([1.0]) if n_rungs == 1 else (np.arange(n_rungs) / (n_rungs - 1)) ** 3
        initial_step = 2.38 / np.sqrt(d) if self.step == "random_walk" else 0.5
        z = jnp.broadcast_to(start[:, None, :], (R, n_rungs, d))
        step_size = jnp.full((R, n_rungs), initial_step)
        inv_mass = jnp.broadcast_to(jnp.diag(scales) if self.dense else scales, (R, n_rungs) + ((d, d) if self.dense else (d,)))
        state = {"key": key, "scan": 0}
        cost = {"gradients": 0, "densities": 0}
        track = None  # (labels, last_end, trips): kept for as long as the ladder is not respaced

        def next_key():
            state["key"], k = jax.random.split(state["key"])
            return k

        def carry_for(z, step_size, inv_mass, betas, track):
            n = len(betas)
            da = jax.vmap(jax.vmap(self.da_init))(step_size)
            labels, last_end, trips = track if track is not None else (jnp.broadcast_to(jnp.arange(n), (R, n)), jnp.full((R, n), NEVER), jnp.zeros(R, int))
            mass_shape = (d, d) if self.dense else (d,)
            sums = (jnp.zeros(()), z, jnp.zeros((R, n, d)), jnp.zeros((R, n) + mass_shape))
            return (z, step_size, inv_mass, da, jnp.asarray(betas), labels, last_end, trips, sums)

        def run_scans(carry, n_scans, adapt, accumulate):
            """Chunks until n_scans are done or time runs out. Returns the carry, the chunks' outputs, and whether time ran out."""
            keys = [next_key() for _ in range(-(-n_scans // CHUNK))]
            carry, outs, out_of_time = self._run_chunks(carry, keys, state["scan"], adapt, accumulate, rule.wall_deadline())
            state["scan"] += CHUNK * len(outs)
            cost["gradients"] += int(sum(np.sum(o["grads"]) for o in outs))
            cost["densities"] += int(sum(np.sum(o["densities"]) for o in outs))
            return carry, outs, out_of_time or rule.out_of_time()

        def top_draws(outs):
            return np.swapaxes(np.concatenate([np.asarray(o["top"]) for o in outs]), 0, 1)   # (R, draws, d)

        # warm-up: windows until the rule says the chains have settled
        history, began = [], time.monotonic()
        cost["seconds"] = 0.0
        while (window := rule.next_window(history)) is not None:
            n_scans, kind = window
            accumulate = kind == "slow"
            carry = carry_for(z, step_size, inv_mass, betas, track)
            carry, outs, out_of_time = run_scans(carry, n_scans, True, accumulate)
            z, step_size, _, da, _, labels, last_end, trips, sums = carry
            track = (labels, last_end, trips)
            if kind == "last":
                step_size = jnp.exp(da.log_step_size_avg)
            draws = top_draws(outs)
            entry = {"window": kind, "planned": n_scans, "scans": draws.shape[1], "rungs": len(betas)}
            if out_of_time:
                history.append(entry)
                cost["seconds"] = time.monotonic() - began
                record = _Record(len(betas) > 1, self.step != "random_walk")
                record.add(outs)
                return self._result(record, betas, step_size, cost, history, track, rule, "warm-up", keep_ladder=False)
            if kind == "slow":
                count, _, total, second = sums
                inv_mass = _welford_final(count, total, second, self.dense)
                entry["rhat"] = float(np.max(split_rhat(draws))) if draws.shape[1] >= 4 else float("inf")
                if n_rungs == 1:
                    entry["peaks"] = self._peaks_among_chains(draws, np.asarray(z[:, -1]), entry["rhat"], rule.wall_deadline())
                if may_escalate and n_rungs == 1:
                    reason = self._needs_ladder(history + [entry])
                    if reason:
                        cost["seconds"] = time.monotonic() - began
                        scales = jnp.diagonal(inv_mass[:, -1], axis1=-2, axis2=-1).mean(axis=0) if self.dense else inv_mass[:, -1].mean(axis=0)
                        return _Escalation(jnp.asarray(z[:, -1]), jnp.asarray(scales), reason, history + [entry], dict(cost))
            if len(betas) > 1 and kind in ("first", "slow"):
                rejection = 1.0 - np.mean(np.concatenate([np.asarray(o["accept_prob"]) for o in outs]), axis=(0, 1))
                n_new = len(betas)
                if auto_rungs:
                    n_new = int(np.clip(np.ceil(2.0 * np.sum(rejection)) + 2, 3, max_rungs))
                    if np.max(rejection) > 0.9:  # saturated: the barrier is underestimated, so grow quickly
                        n_new = int(min(max_rungs, max(n_new, 2 * len(betas))))
                new_betas, _, cumulative = _respace(betas, rejection, n_new)
                nearest = np.abs(np.interp(new_betas, betas, cumulative)[:, None] - cumulative[None, :]).argmin(axis=1)
                z, step_size, inv_mass = z[:, nearest], step_size[:, nearest], inv_mass[:, nearest]
                # the estimate behind the count is noisy, so a change of one rung (or 10%) is not counted as unsettled
                entry["rungs_changed"] = abs(n_new - len(betas)) > max(1, round(0.1 * len(betas)))
                betas, track = new_betas, None
            history.append(entry)

        # sampling: blocks until the convergence tests are met or time runs out
        cost["seconds"], began = time.monotonic() - began, time.monotonic()
        carry = carry_for(z, step_size, inv_mass, betas, track)
        record = _Record(len(betas) > 1, self.step != "random_walk", None if isinstance(rule, _FixedLength) else MAX_KEPT)
        n_scans = rule.first_block()
        while True:
            carry, new, out_of_time = run_scans(carry, n_scans, False, False)
            record.add(new)
            track = carry[5:8]
            verdict = rule.verdict(record.draws(), int(np.sum(np.asarray(track[2]))), len(betas), self.names)
            seen = max((h.get("peaks") or 0 for h in history), default=0)
            if len(betas) == 1 and seen >= 2:
                # Chains without a ladder cannot weigh separate peaks, and chains that all ended in one peak would pass the
                # tests above; what warm-up saw decides.
                verdict |= {"converged": False, "findings": verdict["findings"] + [
                    f"the chains were seen in {seen} separate peaks during warm-up, and chains without a ladder cannot weigh "
                    "separate peaks (use rungs='decide' or 'auto')"]}
                break
            if verdict["converged"] or out_of_time:
                break
            n_scans = rule.next_block(record.scans, verdict)
            if n_scans is None:
                break
        cost["sampling_seconds"] = time.monotonic() - began
        return self._result(record, betas, step_size, cost, history, track, rule, "sampling", verdict=verdict)

    def _run_chunks(self, carry, keys, first_scan, adapt, accumulate, deadline):
        """Chunks of scans, with the replicas split equally over the processes (one process: all of them here). The chunk
        program is compiled once, here, for one process's share of the replicas."""
        executor = self.runner.executor
        W, R = len(executor), carry[0].shape[0]
        r = R // W
        parts = [_take_replicas(carry, slice(k * r, (k + 1) * r)) for k in range(W)]
        leaves, treedef = jax.tree.flatten(parts[0])
        n_rungs = len(carry[4])
        name = ("chunk", id(self), n_rungs, r)   # one program per sampler, rung count and share of replicas
        if not executor.loaded(name):
            chunk = self._chunk_fn(n_rungs, CHUNK)
            out_keys = sorted(jax.eval_shape(chunk, parts[0], jax.random.key(0), 0, True, True)[1])

            def flat(*args):
                new, out = chunk(jax.tree.unflatten(treedef, args[:-4]), jax.random.wrap_key_data(args[-4]), *args[-3:])
                return tuple(jax.tree.leaves(new)) + tuple(out[k] for k in out_keys)

            raw = np.asarray(jax.random.key_data(jax.random.key(0)))
            self.runner.ensure(name, flat, *leaves, raw, np.int64(0), np.bool_(True), np.bool_(True))
            self._out_keys[name] = out_keys
        steps = [[[np.asarray(jax.random.key_data(jax.random.fold_in(key, k))), np.int64(first_scan + i * CHUNK), np.bool_(adapt),
                   np.bool_(accumulate)] for i, key in enumerate(keys)] for k in range(W)]
        results = executor.iterate(name, [jax.tree.leaves(p) for p in parts], steps, deadline)
        done = min(len(outputs) for _, outputs in results)
        carry = _join_replicas([jax.tree.unflatten(treedef, state) for state, _ in results])
        outs = []
        for i in range(done):
            pieces = [[outputs[i][j] for _, outputs in results] for j in range(len(self._out_keys[name]))]
            outs.append({k: (np.concatenate(p, axis=1) if p[0].ndim >= 2 else np.sum(p, axis=0))
                         for k, p in zip(self._out_keys[name], pieces)})
        return carry, outs, done < len(keys)

    def _peaks_among_chains(self, draws, positions, rhat: float, deadline: float):
        """How many separate peaks the chains are in, checked only when they disagree (R-hat over the window at least the
        warm-up limit) and a valley lies between their window means: then the number of certified peaks reached by climbing
        from the chains' positions. None when not checked. Climbing first, rather than testing the means directly, keeps a
        single curved peak (where the straight line between two of its points can dip) from counting as several."""
        if rhat < WARMUP_RHAT_LIMIT or self.climber is None or _group_peaks(draws.mean(axis=1), self.logp_batch)[1] < 2:
            return None
        return self.climber.count_peaks(positions, deadline)

    def _needs_ladder(self, history) -> str | None:
        """After a slow warm-up window without a ladder: does the evidence call for one?"""
        if (history[-1].get("peaks") or 0) >= 2:
            return f"the chains settled in {history[-1]['peaks']} separate peaks during warm-up"
        slow = sum(h["scans"] for h in history if h["window"] == "slow")
        if slow >= ESCALATE_AFTER and history[-1]["rhat"] > WARMUP_RHAT_LIMIT:
            return f"the chains still disagreed after {slow} warm-up scans (R-hat {history[-1]['rhat']:.2f})"
        return None

    def _result(self, record, betas, step_size, cost, history, track, rule, phase, verdict=None, keep_ladder=True) -> Posterior:
        draws = record.draws()
        n_rungs = len(betas)
        trips = int(np.sum(np.asarray(track[2]))) if track is not None else 0
        if isinstance(rule, _FixedLength):
            draws = draws[:, :rule.n_draws]
        R, n, d = draws.shape
        if verdict is None:
            verdict = rule.verdict(draws, trips, n_rungs, self.names)
        ladder, swap_acceptance, barrier = None, (), None
        if n_rungs > 1:
            accept = record.swap_acceptance()
            swap_acceptance = tuple(float(a) for a in accept)
            barrier = float(np.sum(1.0 - accept))
            if keep_ladder:
                ladder = Ladder(np.asarray(betas), record.ratios()[:, :n], self.reference.name)
        status = rule.status(verdict, phase, record.scans)
        stopped_by = ("converged" if verdict["converged"] else "fixed length" if isinstance(rule, _FixedLength)
                      else "time limit" if rule.out_of_time() else "separate peaks seen")
        return Posterior(draws.reshape(-1, d), np.full(R * n, 1.0 / (R * n)), None,
                         f"mcmc ({self.step}, {n_rungs} rung{'s' if n_rungs > 1 else ''})",
                         {"chains": draws, "replicas": R, "step": self.step, "rungs": n_rungs, "betas": tuple(float(b) for b in betas),
                          "reference": self.reference.name if self.reference is not None else None, "swap_acceptance": swap_acceptance,
                          "round_trips": trips if n_rungs > 1 else None, "barrier": barrier, "divergences": record.divergences,
                          "gradient_evaluations": cost["gradients"], "density_evaluations": cost["densities"],
                          "step_sizes": np.asarray(step_size), "mass_matrix": "full" if self.dense else "diagonal",
                          "warmup": tuple(history), "ladder": ladder, "draws_per_chain": record.scans, "draws_kept_per_chain": n,
                          "thinned_by": record.stride, "phase": phase,
                          "stopped_by": stopped_by, "status": status, "timings": {"warm-up": cost.get("seconds", 0.0),
                                                                                  "sampling": cost.get("sampling_seconds", 0.0)}, **verdict})
