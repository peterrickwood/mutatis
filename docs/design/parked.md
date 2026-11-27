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

2. **Which sampler `estimate()` chooses automatically (raised 2026-11-26).** Decided in discussion: the default is the
   general sampler (decision 41: NUTS steps, a tempering ladder, 4 replicas, checks), with the evidence as a separate,
   optional step, and SMC kept for comparison. Open: switching `estimate()`'s automatic choice from SMC and NUTS to it, once
   the sampler benchmark (`python -m benchmarks.samplers`) has been reviewed. SMC's failures with many unknowns are in
   decision 38.

3. **Range acceptance criterion for Read (2022) (decision 32, raised 2026-11-26).** Stage 1's test requires our range
   bounds to be exact rather than to match Read's published numbers, because his are not always exact (decision 28). This
   changes the plan's acceptance criterion and needs confirmation.

4. **Computing power for DSGE-sized estimation (raised 2026-11-26).** At 50 states one likelihood gradient costs about 21 ms
   per particle on this machine's CPU, so an SMC run on a DSGE of that size would take hours. Options to weigh before
   stage 4: a GPU, fewer particles or moves, a cheaper filter (see decision 39), or accepting overnight runs.

Also deferred, recorded elsewhere: the written report on the replication of Read (2022) (PLAN.md, end of stage 1).
