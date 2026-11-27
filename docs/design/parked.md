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

2. **The default sampler, and which one `estimate()` chooses automatically (raised 2026-11-26; reopened 2026-11-27 for
   a separate design session).** Settled: a general sampler with selectable step rule, ladder, replicas, starts and
   checks (decision 41); the evidence as a separate, optional step; SMC kept for comparison. Open: which settings make a
   good default, given the short benchmark suite (`python -m benchmarks.samplers`, report in
   `benchmarks/samplers/results/short-1/`), and only then whether `estimate()`'s automatic choice moves to it. From that
   suite:
   - Accurate: the ladder 26/33, single-rung NUTS 23/34, SMC 21/34. Confidently wrong (wrong with checks passing): 2, 5
     and 13. Median time per run: 67 s, 20 s and 10 s; the ladder costs 3 to 30+ times NUTS where the prior is much wider
     than the posterior (34 rungs and over 1,500 s on the 21-unknown regression).
   - The ladder does not join two peaks of unequal width in 20 dimensions (all methods put every draw in the wide peak,
     and the ladder's checks passed): a known limit of tempering (Woodard, Schmidler and Huber 2009). More starts would
     raise the chance that the checks see it, not fix it.
   - The ladder's checks raised false alarms on 8 of its 26 accurate runs: too few round trips on long ladders, and
     divergent trajectories on the banana and hidden-level targets.
   - Round trips fall well below the formula for exact moves (0 against about 46 on the 49-rung ladder). Investigated
     2026-11-27; not a defect. Mainly, trips are counted only within the 1,000 sampling scans, from a fresh start, and on a
     49-rung ladder one state needs about 4,200 scans per trip on average: with exact moves the same ladder also gives 0
     and 2 trips over 1,000 draws, and 85 to 87% of the formula over 20,000. Secondly, NUTS moves do not fully refresh a
     rung's state between exchanges (lag-1 autocorrelation of the log likelihood about 0.27, against 0 for exact moves),
     which gives 67 to 83% of the formula's trips over 5,000 draws at 2 and 10 unknowns, against 95 to 97% for exact
     moves. So the round-trip check asks for more trips than a long ladder can show in 1,000 draws, even when it works as
     the theory says; what the check should compare against is for the design session.

3. **Range acceptance criterion for Read (2022) (decision 32, raised 2026-11-26).** Stage 1's test requires our range
   bounds to be exact rather than to match Read's published numbers, because his are not always exact (decision 28). This
   changes the plan's acceptance criterion and needs confirmation.

4. **Computing power for DSGE-sized estimation (raised 2026-11-26).** At 50 states one likelihood gradient costs about 21 ms
   per particle on this machine's CPU, so an SMC run on a DSGE of that size would take hours. Options to weigh before
   stage 4: a GPU, fewer particles or moves, a cheaper filter (see decision 39), or accepting overnight runs.

Also deferred, recorded elsewhere: the written report on the replication of Read (2022) (PLAN.md, end of stage 1).
