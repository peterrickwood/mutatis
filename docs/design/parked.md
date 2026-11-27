# Parked questions

Questions raised in discussion and set aside, to come back to. When one is settled, record the outcome in
[decisions.md](decisions.md) or [PLAN.md](../../PLAN.md) and remove it here.

1. **How the forward-looking solve reports failure (parked 2026-11-26).** Counting the model's roots, to tell "no stable
   solution" from "many stable solutions", is 85 to 90% of the solve's time: the complex eigenvalues of a 2n × 2n matrix (40
   variables: 1.10 s of 1.24 s for 100 solves; 80 variables: 5.51 s of 6.75 s). Judging the outcome from the solution itself
   would make the solve about 4.5 times faster at 80 variables, but when cyclic reduction does not settle it cannot say which
   of the two failures happened. Options: a single status "no unique solution" during estimation (both failures make the
   likelihood minus infinity anyway), with the full root count kept as a separate check run when diagnosing a model; or a
   fallback to counting roots for the failing draws only. Suggested: the first. Source: the numerical audit, decisions 34
   and 35.

2. **The default sampler: what was left open when it was settled (raised 2026-11-26; default settled 2026-11-27, decision
   43; parked so that the project moves on).** Settled: the general sampler (decision 41); its data-driven settings (decision
   42); always a ladder, after the peak search (decision 43). Still open: whether `estimate()`'s automatic choice (exact
   regression, NUTS for sampled shocks, otherwise SMC) moves to it. Evidence below is from the short suite
   (`benchmarks/samplers/results/short-4`, report there) and the small held-out suite (`results/held-out-2`).
   - **An early stop between peaks of unequal width.** On two_peaks_d5 (weights 0.3 and 0.7, the heavier peak narrow), seed
     1 stopped as converged after 1,025 draws per chain, with 2,570 effective draws, and weights 0.25 / 0.75. Seed 0, same
     settings, ran 200,075 draws per chain to the time limit: weights right (0.303), but only 359 effective draws. So moving
     weight between the peaks is slow, the short run did not see it, and the stopping rule took the first passing check. Likely
     the known weakness of tempering between peaks of unequal width (Woodard, Schmidler and Huber 2009); not tested. To try:
     requiring the tests to pass in consecutive blocks. The ladder from a fitted normal got both seeds right (not understood).
   - **Long ladders from the prior.** The barrier grows with how much the data narrow the prior, about as the square root of
     the number of unknowns (normal means: 3.9, 9.8, 22.3 at 2, 10, 50 unknowns). The rung count starts at 8 and doubles
     whenever an exchange is rejected more than 90% of the time; on normal_means_d50 and conjugate_regression_n2_p4 warm-up
     reached 64 rungs and did not finish in 600 s; on conjugate_regression_n2_p1 it went to 64, then back to 22, with 444 of
     527 s in warm-up. Why a 64-rung scan is so slow was not measured. Each change of rung count also recompiles.
   - **Where the ladder starts.** From a normal fitted by Pathfinder (`reference="fitted"`), short-4: 10 rungs and right on
     normal_means_d50 in about 230 s, and both two_peaks_d5 seeds right; but confidently wrong on 3 of 4 banana runs (spread 8
     to 23% too narrow: a normal fitted at the peak does not reach the curved arms), and 64 rungs on scales_d20 (barrier 13
     against 6.6 from the prior). Untested: the two-leg version, one leg from the prior and one from the fitted normal
     (Surjanovic, Syed, Bouchard-Cote and Campbell, NeurIPS 2022); a start built from the peaks the search certified.
   - **The funnel** (funnel_d10): every MCMC method wrong or out of time; the default reports not converged on both seeds.
   - **Divergent trajectories do not count against convergence.** Plain chains on the bananas were confidently wrong with 186
     to 450,900 of them; the sampled-shocks model has 35 to 60 in accurate runs.
   - Round trips fall well below the formula for exact moves (0 against about 46 on the 49-rung ladder). Investigated
     2026-11-27; not a defect. Mainly, trips are counted only within the 1,000 sampling scans, from a fresh start, and on a
     49-rung ladder one state needs about 4,200 scans per trip on average: with exact moves the same ladder also gives 0
     and 2 trips over 1,000 draws, and 85 to 87% of the formula over 20,000. Secondly, NUTS moves do not fully refresh a
     rung's state between exchanges (lag-1 autocorrelation of the log likelihood about 0.27, against 0 for exact moves),
     which gives 67 to 83% of the formula's trips over 5,000 draws at 2 and 10 unknowns, against 95 to 97% for exact
     moves. So the round-trip check asks for more trips than a long ladder can show in 1,000 draws, even when it works as
     the theory says; what the check should compare against is for the design session.
   - **The search's limits.** No search above 200 unknowns (the certificate needs the full Hessian). On expensive models the
     10% time cap leaves it weak: 8 starts on the 103-unknown stochastic volatility model of the held-out set (could miss a peak
     reached from 1 in 3 prior draws), 64 on the forward-looking model. Compiling is serial, in the main process.
   - **The 20-dimension narrow peak** (two_peaks_d20): beyond every method, and beyond any search from prior draws (its
     catchment holds 1 in 9 x 10^13 prior draws), as the detection limit states.
   - **SMC is dropped as a default candidate (2026-11-27).** At 2,000 particles it was confidently wrong on 13 of 34 runs
     in short-1, and it has no convergence test of its own. The test of whether 10 times the particles changes that: 4
     independent runs of 20,000 particles each, pooled, with a convergence test across the runs (R-hat across runs below
     1.01, and the pooled mean's standard error, from the spread of the runs' means, at most 0.025 sd: the default's
     tests). It ran on the 7 targets where SMC had failed (two peaks in 5 and 20 dimensions, funnel, hidden level with
     sampled shocks, Student t, mild banana in 10 dimensions, correlated normal with condition number 10,000), seeds 0
     and 1, against the new default's runs in short-2. Criterion, fixed before running: no more confidently wrong runs
     and no fewer accurate runs than the default. Result over the 14 runs: SMC ×10 accurate 3, flagged by its test 8,
     confidently wrong 3 (Student t seed 1, and two peaks in 20 dimensions twice); the default accurate 8, flagged 2,
     confidently wrong 4 (two peaks in 5 and in 20 dimensions). The errors did not shrink with 10 times the particles
     (the funnel's scale unknown off by 1.8 to 1.9 sd, the hidden level's noise size by 0.9 to 1.0 sd, Student t's spread
     65 to 80% too wide), so they come from SMC's moves, not from too few particles. Worth keeping from it: comparing
     independent runs caught 8 of SMC ×10's 11 wrong answers, including both two-peak runs in 5 dimensions that the
     default got confidently wrong. The test was a scratch script, not kept in the repo.
   - **Held-out check.** The small held-out suite (`python -m benchmarks.samplers --suite held_out`, 11 Inference Gym targets,
     2 seeds) was run with decision 42's rule only (`results/held-out-2`): 7 accurate, 4 accurate but out of time, 6 wrong and
     reported not converged, 5 confidently wrong (banana seed 0; Brownian motion with unknown scales, both seeds, the
     observation noise's mean off 0.26 to 0.32 sd; the small stochastic volatility model, both seeds, the spread of the
     Cauchy-prior mean 21 to 25% too wide, where the reference spread may itself be off). Not yet rerun with the default of
     decision 43 (agreed 2026-11-27 not to now). Before the project is finished, on a machine with many more cores: the full
     held-out check, the Inference Gym targets in the long suite (all four main methods, 4 seeds, including the 2,519-unknown
     stochastic volatility model).

3. **Range acceptance criterion for Read (2022) (decision 32, raised 2026-11-26).** Stage 1's test requires our range
   bounds to be exact rather than to match Read's published numbers, because his are not always exact (decision 28). This
   changes the plan's acceptance criterion and needs confirmation.

4. **Computing power for DSGE-sized estimation (raised 2026-11-26).** At 50 states one likelihood gradient costs about 21 ms
   per particle on this machine's CPU, so an SMC run on a DSGE of that size would take hours. Options to weigh before
   stage 4: a GPU, fewer particles or moves, a cheaper filter (see decision 39), or accepting overnight runs.

Also deferred, recorded elsewhere: the written report on the replication of Read (2022) (PLAN.md, end of stage 1).
