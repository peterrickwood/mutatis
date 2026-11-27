"""General MCMC, built from four interchangeable parts: a step rule, a tempering ladder, independent replicas, and checks.

- **Step rule**: how one chain moves. `nuts` (the No-U-Turn Sampler: gradient-based, chooses its own trajectory length),
  `hmc` (gradient-based, random trajectory length up to `hmc_max_steps`), or `random_walk` (no gradients; `random_walk_steps`
  Metropolis steps per scan). Each rung tunes its own step size (dual averaging, as in Stan) and scales (one per unknown, from
  windows of its own draws).
- **Ladder** (parallel tempering): chains at "inverse temperatures" beta from 0 to 1. The chain at beta targets
  reference^(1 - beta) x posterior^beta, so beta = 1 is the posterior and beta = 0 the reference: the prior, or a normal
  distribution fitted by Pathfinder ("fitted", needed when the prior is improper). The beta = 0 rung draws exactly from the
  reference. After every scan, neighbouring rungs propose to exchange their states, alternating even and odd pairs (the
  non-reversible scheme of Syed, Bouchard-Cote, Deligiannidis and Doucet, JRSS B 2022). During warm-up the betas are respaced
  so that every exchange is rejected about equally often, and with `rungs="auto"` the number of rungs is set from the
  estimated "communication barrier" (the sum of rejection rates along the ladder; about twice it is enough). States carried
  from the reference end let the posterior chain move between separated peaks. One rung (`rungs=1`) is ordinary MCMC.
- **Replicas and starts**: `replicas` independent copies of the whole ladder, each started from its own point: a prior draw,
  or a draw from Pathfinder's normal approximation (Zhang, Carpenter, Gelman and Vehtari, JMLR 2022), which runs an
  optimiser from a prior draw and draws a spread-out start near where the posterior's mass is, rather than at the peak.
- **Checks**: rank-normalised split R-hat and bulk effective sample size across replicas (Vehtari and others, 2021);
  divergent trajectories; round trips of states from the reference end of the ladder to the posterior end and back; too few
  rungs for the barrier. Any finding makes `converged` False.

NUTS as usually run is the special case `mcmc(problem, key, rungs=1, starts="prior")`.

The ladder's record (each rung's log-likelihood ratios) is kept so the evidence can be computed afterwards by stepping stones
(`mutatis.infer.evidence`); computing it is a separate, optional step.
"""

from collections.abc import Callable
from dataclasses import dataclass

import blackjax
import blackjax.diagnostics
import blackjax.vi.pathfinder
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation

from mutatis import _jax_config  # noqa: F401
from mutatis.infer.posterior import Posterior

STEPS = ("nuts", "hmc", "random_walk")
TARGET_ACCEPTANCE = {"nuts": 0.8, "hmc": 0.8, "random_walk": 0.234}
RHAT_LIMIT = 1.01
ESS_PER_REPLICA = 100        # bulk effective sample size wanted per replica (Vehtari and others, 2021)
ROUND_TRIPS_PER_REPLICA = 2  # fewer means the ladder may not connect the reference to the posterior
NEVER, BOTTOM, TOP = -1, 0, 1


class MCMCError(RuntimeError):
    pass


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


def mcmc(problem, key, step: str = "nuts", rungs="auto", reference: str = "auto", replicas: int = 4, starts: str = "pathfinder",
         n_warmup: int = 1000, n_draws: int = 1000, initial_rungs: int = 8, max_rungs: int = 64, hmc_max_steps: int = 10,
         random_walk_steps: int = 10, target_acceptance: float | None = None) -> Posterior:
    """Sample a posterior problem (contract 2: `log_prior`, `log_likelihood`, `draw_prior`, `size`, `proper`)."""
    if step not in STEPS:
        raise MCMCError(f"unknown step rule {step!r}: use one of {', '.join(STEPS)}")
    if starts not in ("pathfinder", "prior"):
        raise MCMCError(f"unknown starts {starts!r}: use pathfinder or prior")
    if not getattr(problem, "gradient_ready", True):
        raise MCMCError("this problem has unknowns that cannot be moved (a rotation, or no prior); use the exact formula")
    if rungs != "auto" and (not isinstance(rungs, int) or rungs < 1):
        raise MCMCError("rungs is 'auto' or a whole number of at least 1")
    proper = getattr(problem, "proper", True)
    if starts == "prior" and not proper:
        raise MCMCError("cannot start from prior draws: the prior is improper; use starts='pathfinder'")
    reference = ("prior" if proper else "fitted") if reference == "auto" else reference
    if reference == "prior" and not proper and rungs != 1:
        raise MCMCError("the ladder cannot end at an improper prior; use reference='fitted'")

    log_target = lambda z: problem.log_prior(z) + problem.log_likelihood(z)
    key_starts, key_run = jax.random.split(key)
    start, pathfinder_draws = _starts(problem, key_starts, log_target, replicas, starts)
    ref, ell_fn = None, None
    if rungs != 1:
        if reference == "prior":
            ref, ell_fn = Reference("prior", problem.log_prior, problem.draw_prior), problem.log_likelihood
        elif reference == "fitted":
            if pathfinder_draws is None:
                _, pathfinder_draws = _starts(problem, key_starts, log_target, replicas, "pathfinder")
            if pathfinder_draws is None:
                raise MCMCError("Pathfinder failed from every start, so no reference can be fitted")
            ref = _fitted_reference(pathfinder_draws)
            ell_fn = lambda z: log_target(z) - ref.log_density(z)
        else:
            raise MCMCError(f"unknown reference {reference!r}: use auto, prior or fitted")
    scales = np.var(pathfinder_draws, axis=0) if pathfinder_draws is not None else np.ones(problem.size)

    sampler = _Sampler(problem.size, step, ref, ell_fn, log_target, TARGET_ACCEPTANCE[step] if target_acceptance is None else target_acceptance,
                       hmc_max_steps, random_walk_steps)
    n_rungs = 1 if rungs == 1 else (initial_rungs if rungs == "auto" else rungs)
    return sampler.run(key_run, start, jnp.asarray(np.maximum(scales, 1e-12)), n_rungs, rungs == "auto", max_rungs, n_warmup, n_draws)


def nuts(problem, key, n_chains: int = 4, n_warmup: int = 1000, n_draws: int = 1000) -> Posterior:
    """NUTS as usually run: independent chains started from prior draws, no ladder. A named setting of `mcmc`."""
    return mcmc(problem, key, step="nuts", rungs=1, replicas=n_chains, starts="prior", n_warmup=n_warmup, n_draws=n_draws)


# ---------------------------------------------------------------------------------------------------------------- starts

def _finite_prior_draws(problem, key, log_target, n: int, tries: int = 50):
    """n prior draws at which the posterior density is finite (the model may have no solution at some prior draws)."""
    candidates = jax.vmap(problem.draw_prior)(jax.random.split(key, n * tries))
    finite = np.isfinite(np.asarray(jax.vmap(log_target)(candidates)))
    if finite.sum() < n:
        raise MCMCError(f"only {finite.sum()} of {n * tries} prior draws give a finite posterior density; need {n}")
    return candidates[np.flatnonzero(finite)[:n]]


def _starts(problem, key, log_target, replicas, how):
    """One start per replica, and (for Pathfinder) the pooled approximate draws used for scales and a fitted reference."""
    key_initial, key_paths = jax.random.split(key)
    if getattr(problem, "proper", True):
        initial = _finite_prior_draws(problem, key_initial, log_target, replicas)
    else:
        initial = jax.random.normal(key_initial, (replicas, problem.size))
    if how == "prior":
        return initial, None
    starts, pooled = [], []
    for r, k in enumerate(jax.random.split(key_paths, replicas)):
        k_fit, k_draw = jax.random.split(k)
        state, _ = blackjax.vi.pathfinder.approximate(k_fit, log_target, initial[r], num_samples=200)
        draws, _ = blackjax.vi.pathfinder.sample(k_draw, state, 200)
        draws = np.asarray(draws)
        ok = np.all(np.isfinite(draws), axis=1) & np.isfinite(np.asarray(jax.vmap(log_target)(jnp.asarray(draws))))
        if ok.sum() == 0:  # Pathfinder failed from this point: fall back to the plain start, and say so in the results
            starts.append(np.asarray(initial[r]))
            continue
        starts.append(draws[np.flatnonzero(ok)[0]])
        pooled.append(draws[ok])
    pooled = np.concatenate(pooled) if pooled else None
    return jnp.asarray(np.stack(starts)), pooled


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


# ---------------------------------------------------------------------------------------------------------------- warm-up

def _windows(n_chunks: int) -> list[tuple[int, str]]:
    """Warm-up in chunks, as Stan's windows: a first window, slow windows doubling in length (scales and the ladder are
    re-estimated at the end of each), and a final window where only the step size adapts."""
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


def _welford_final(count, total, total_sq):
    """Stan's regularised variance: shrunk towards 1e-3 when there are few draws."""
    n = jnp.maximum(count, 1.0)
    mean = total / n
    variance = jnp.maximum(total_sq / n - mean**2, 0.0) * n / jnp.maximum(n - 1.0, 1.0)
    return (n / (n + 5.0)) * variance + 1e-3 * (5.0 / (n + 5.0))


# ---------------------------------------------------------------------------------------------------------------- sampler

class _Sampler:
    def __init__(self, size, step, reference, ell_fn, log_target, target_acceptance, hmc_max_steps, random_walk_steps):
        self.size, self.step, self.reference, self.ell_fn, self.log_target = size, step, reference, ell_fn, log_target
        self.da_init, self.da_update, self.da_final = dual_averaging_adaptation(target_acceptance)
        self.hmc_max_steps, self.random_walk_steps = hmc_max_steps, random_walk_steps
        self._chunk_fns = {}

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
        diverged, and the number of gradient and density evaluations."""
        log_density = self._tempered(beta, single)
        if self.step == "random_walk":
            def one(carry, k):
                z, lp = carry
                k_prop, k_accept = jax.random.split(k)
                proposal = z + step_size * jnp.sqrt(inv_mass) * jax.random.normal(k_prop, z.shape)
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
        if (n_rungs, length) in self._chunk_fns:
            return self._chunk_fns[(n_rungs, length)]
        moved = slice(1, n_rungs) if n_rungs > 1 else slice(0, 1)  # rung 0 of a ladder draws exactly from the reference
        single = n_rungs == 1
        move = jax.vmap(jax.vmap(lambda k, z, b, e, m: self._move(k, z, b, e, m, single)))
        ell_all = None if single else jax.vmap(jax.vmap(self.ell_fn))
        draw_ref = None if single else jax.vmap(self.reference.draw)

        def scan_one(carry, inputs):
            key, parity = inputs
            z, step_size, inv_mass, da, betas, labels, last_end, trips, flags = carry
            adapt, accumulate = flags["adapt"], flags["accumulate"]
            R = z.shape[0]
            k_move, k_ref, k_swap = jax.random.split(key, 3)
            sub = z[:, moved]
            keys = jax.random.split(k_move, R * sub.shape[1]).reshape(R, sub.shape[1])
            new, rate, divergent, grads, densities = move(keys, sub, jnp.broadcast_to(betas[moved], sub.shape[:2]), step_size[:, moved], inv_mass[:, moved])
            z = z.at[:, moved].set(new)
            # step-size adaptation at the rungs that moved
            da_new = jax.vmap(jax.vmap(self.da_update))(jax.tree.map(lambda a: a[:, moved], da), rate)
            da = jax.tree.map(lambda a, b: jnp.where(adapt, a.at[:, moved].set(b), a), da, da_new)
            step_size = jnp.where(adapt, jnp.exp(da.log_step_size), step_size)
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
            # running sums of each rung's draws, for its scales
            flags = flags | {"count": flags["count"] + accumulate, "total": flags["total"] + accumulate * z,
                             "total_sq": flags["total_sq"] + accumulate * z**2}
            out["top"] = z[:, -1]
            return (z, step_size, inv_mass, da, betas, labels, last_end, trips, flags), out

        @jax.jit
        def chunk(carry, key, start):
            keys = jax.random.split(key, length)
            parity = (start + jnp.arange(length)) % 2
            return jax.lax.scan(scan_one, carry, (keys, parity))

        self._chunk_fns[(n_rungs, length)] = chunk
        return chunk

    def run(self, key, start, scales, n_rungs, auto_rungs, max_rungs, n_warmup, n_draws):
        R, d = start.shape
        chunk_len = max(5, min(25, n_warmup // 10, max(n_draws // 10, 5)))
        n_draw_chunks = -(-n_draws // chunk_len)
        betas = np.array([1.0]) if n_rungs == 1 else (np.arange(n_rungs) / (n_rungs - 1)) ** 3
        initial_step = 2.38 / np.sqrt(d) if self.step == "random_walk" else 0.5
        z = jnp.broadcast_to(start[:, None, :], (R, n_rungs, d))
        step_size = jnp.full((R, n_rungs), initial_step)
        inv_mass = jnp.broadcast_to(scales, (R, n_rungs, d))
        barrier, scan_index, history = None, 0, []
        keys = iter(jax.random.split(key, 10_000))

        def fresh(n):
            return {"count": jnp.zeros(()), "total": jnp.zeros((R, n, d)), "total_sq": jnp.zeros((R, n, d)),
                    "adapt": jnp.array(True), "accumulate": jnp.zeros(())}

        def new_carry(z, step_size, inv_mass, betas, adapt, accumulate):
            n = len(betas)
            da = jax.vmap(jax.vmap(self.da_init))(step_size)
            labels = jnp.broadcast_to(jnp.arange(n), (R, n))
            last_end = jnp.full((R, n), NEVER)
            flags = fresh(n) | {"adapt": jnp.array(adapt), "accumulate": jnp.array(float(accumulate))}
            return (z, step_size, inv_mass, da, jnp.asarray(betas), labels, last_end, jnp.zeros(R, int), flags)

        cost = {"gradients": 0, "densities": 0}

        def run_chunks(carry, n_chunks):
            nonlocal scan_index
            outs = []
            fn = self._chunk_fn(len(carry[4]), chunk_len)
            for _ in range(n_chunks):
                carry, out = fn(carry, next(keys), scan_index)
                scan_index += chunk_len
                cost["gradients"] += int(np.sum(np.asarray(out["grads"])))
                cost["densities"] += int(np.sum(np.asarray(out["densities"])))
                outs.append(out)
            return carry, outs

        # warm-up
        n_warmup_chunks = max(1, -(-n_warmup // chunk_len))
        windows = _windows(n_warmup_chunks)
        slow_windows = [i for i, (_, kind) in enumerate(windows) if kind == "slow"]
        for i, (n_chunks, kind) in enumerate(windows):
            carry = new_carry(z, step_size, inv_mass, betas, True, kind == "slow")
            carry, outs = run_chunks(carry, n_chunks)
            z, step_size, inv_mass, da, _, _, _, _, flags = carry
            step_size = jnp.exp(da.log_step_size_avg) if kind == "last" else step_size
            if kind == "slow":
                inv_mass = _welford_final(flags["count"], flags["total"], flags["total_sq"])
            if len(betas) > 1 and kind != "last" and (not slow_windows or i < slow_windows[-1]):
                rejection = 1.0 - np.mean(np.concatenate([np.asarray(o["accept_prob"]) for o in outs]), axis=(0, 1))
                n_new = len(betas)
                if auto_rungs:
                    n_new = int(np.clip(np.ceil(2.0 * np.sum(rejection)) + 2, 3, max_rungs))
                    if np.max(rejection) > 0.9:  # saturated: the barrier is underestimated, so grow quickly
                        n_new = int(min(max_rungs, max(n_new, 2 * len(betas))))
                new_betas, barrier, cumulative = _respace(betas, rejection, n_new)
                nearest = np.abs(np.interp(new_betas, betas, cumulative)[:, None] - cumulative[None, :]).argmin(axis=1)
                z, step_size, inv_mass = z[:, nearest], step_size[:, nearest], inv_mass[:, nearest]
                betas = new_betas
            history.append({"window": kind, "rungs": len(betas)})

        # draws
        carry = new_carry(z, step_size, inv_mass, betas, False, False)
        carry, outs = run_chunks(carry, n_draw_chunks)
        trips = int(np.sum(np.asarray(carry[7])))
        cat = lambda name: np.concatenate([np.asarray(o[name]) for o in outs])
        draws = np.swapaxes(cat("top"), 0, 1)[:, :n_draws]                       # (R, n_draws, d)
        divergences = int(np.sum(cat("divergent")[:, :, -1])) if self.step != "random_walk" else 0
        ladder, swap_acceptance = None, ()
        if len(betas) > 1:
            accept = cat("accept_prob")                                         # (scans, R, N-1)
            swap_acceptance = tuple(float(a) for a in np.mean(accept, axis=(0, 1)))
            rejection = 1.0 - np.mean(accept, axis=(0, 1))
            barrier = float(np.sum(rejection))
            ladder = Ladder(np.asarray(betas), np.swapaxes(cat("ell"), 0, 1)[:, :n_draws], self.reference.name)
        checks = _checks(draws, divergences, len(betas), trips, barrier)
        return Posterior(draws.reshape(-1, d), np.full(R * n_draws, 1.0 / (R * n_draws)), None,
                         f"mcmc ({self.step}, {len(betas)} rung{'s' if len(betas) > 1 else ''})",
                         {"chains": draws, "replicas": R, "step": self.step, "rungs": len(betas), "betas": tuple(float(b) for b in betas),
                          "reference": self.reference.name if self.reference is not None else None, "swap_acceptance": swap_acceptance,
                          "round_trips": trips if len(betas) > 1 else None, "barrier": barrier, "divergences": divergences,
                          "gradient_evaluations": cost["gradients"], "density_evaluations": cost["densities"],
                          "step_sizes": np.asarray(step_size), "warmup": tuple(history), "ladder": ladder, **checks})


# ---------------------------------------------------------------------------------------------------------------- checks

def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Rank-normalised split R-hat per unknown, from draws of shape (chains, draws, unknowns)."""
    return np.asarray(jax.vmap(blackjax.diagnostics.rhat, in_axes=2)(jnp.asarray(chains)))


def bulk_ess(chains: np.ndarray) -> np.ndarray:
    """Bulk effective sample size per unknown, from draws of shape (chains, draws, unknowns)."""
    return np.asarray(jax.vmap(blackjax.diagnostics.ess_bulk, in_axes=2)(jnp.asarray(chains)))


def _checks(draws, divergences, n_rungs, trips, barrier) -> dict:
    R, n, d = draws.shape
    rhat = split_rhat(draws)
    ess = bulk_ess(draws)
    findings = []
    if np.max(rhat) > RHAT_LIMIT:
        findings.append(f"R-hat above {RHAT_LIMIT} for {int(np.sum(rhat > RHAT_LIMIT))} of {d} unknowns (largest {np.max(rhat):.3f}): replicas disagree")
    if np.min(ess) < ESS_PER_REPLICA * R:
        findings.append(f"effective sample size below {ESS_PER_REPLICA * R} for {int(np.sum(ess < ESS_PER_REPLICA * R))} of {d} unknowns (smallest {np.min(ess):.0f})")
    if divergences:
        findings.append(f"{divergences} divergent trajectories at the posterior rung: some regions may be missed")
    if n_rungs > 1:
        if trips < ROUND_TRIPS_PER_REPLICA * R:
            findings.append(f"{trips} round trips between the reference and the posterior (want at least {ROUND_TRIPS_PER_REPLICA * R}): the ladder may not connect them")
        if barrier is not None and n_rungs < 2 * barrier:
            findings.append(f"{n_rungs} rungs for a communication barrier of {barrier:.1f}; about {int(np.ceil(2 * barrier)) + 2} are needed")
    return {"max_rhat": float(np.max(rhat)), "min_ess": float(np.min(ess)), "findings": findings, "converged": not findings}
