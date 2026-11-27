# Decisions made during implementation

Decisions taken while building, without stopping to ask, as agreed on 2026-11-26. Each says what was decided, why, and what
the alternative was, so it can be reviewed and reversed. Decisions agreed in discussion are recorded in
[specification-sketches.md](specification-sketches.md) and [PLAN.md](../../PLAN.md) instead.

## Stage 0

**1. Distributions are a thin layer over `jax.scipy.stats` and `jax.random`, not NumPyro.** The set needed is small (about
a dozen families), each is a few lines over JAX's own functions, and every one is tested against SciPy. NumPyro would bring a
whole probabilistic-programming framework for this one use. Alternative: `numpyro.distributions`.

**2. Equations use mutatis's own small parser and syntax tree, not sympy.** Equations need time shifts (`x(-1)`), marked
expectations (`E(...)`), elements and slices of vectors (`y[a]`, `B[l]`), and sums over sets (`sum(l in lags: ...)`). Sympy's
parser handles none of these directly. Owning the tree also gives every node its position in the text, which `name_number` needs
to rewrite one number without disturbing the rest. JAX evaluates the tree directly, and derivatives come from JAX's automatic
differentiation, so neither sympy nor sympy2jax is needed. PLAN.md's library table expected sympy for trees and derivatives;
sympy remains available if symbolic work is needed later (deriving equations from optimisation problems, for example).

**3. A block's equations may use only its own ports' variables.** Any variable an equation mentions must be one of the block's
inputs or outputs. This makes the ports a complete statement of what a block depends on, which is what `replace_block` and
*isolate* rely on.

**4. Block names and data series names must be distinct.** A target such as `y.sigma_eps` then names one thing: a block's
parameter or a data link's parameter.

**5. Parameters and shocks declared in a template block are shared by all its instances.** This matches how the RBA housing model
is written: one investment-adjustment cost for non-traded, manufacturing and commodity investment, and one investment shock. An
instance gets its own value with `untie`; a shock or parameter that differs by element is declared with the set as its shape
(`"u[sectors]"`) and used as `u[j]`.

**6. Numbers that state structure are not offered for naming.** `find_numbers` leaves out time shifts (the 1 in `x(-1)`), element
positions, ranges and dates. A lag length is changed with `replace_equation`.

**7. `diff` gives back the same edit for every single edit except `name_number`.** Naming a number shows up as an added parameter
and a replaced equation, which has the same effect. Making it recognise `name_number` would mean guessing intent from text.

**8. Every value from the future must be written inside `E(...)`.** `p(+1)` outside an expectation is an error. The
expectation schemes work by rewriting `E(...)` terms, so an unmarked future value would silently escape them. A Dynare
importer will wrap leads in `E(...)` automatically. Leads of more than one period, and leads of shocks or outside variables,
are not supported yet.

**9. Lags longer than one period become chains of extra variables, named `x@lag1`, `x@lag2`, ...; a shock or outside variable
that appears lagged first gets a copy in the state, `b@now`.** This is the standard reduction to a system with one lag, and it
keeps the solved model's step simple: next state from this state, this period's shocks and this period's outside variables.

**10. The discounting factor of `discounted(m)` must be a model-level parameter.** It applies to every expectation in every
block, so it belongs to the model, not to one block.

**11. Compiled equations are evaluated by walking the syntax tree inside JAX, not by generating code.** JAX traces the walk once
when a function is compiled, so there is no cost per evaluation, and the same tree serves for evaluation, derivatives (by
automatic differentiation) and counting equations.

**12. Solved models carry their own noise: the shock and measurement-error distributions, built for the parameter values.**
Contract 1 then answers everything estimation needs from one object, including the optional abilities "linear" and "normal
noise" that select the Kalman filter.

**13. Whether shocks and measurement errors are normal with mean zero is decided from the specification's text, before any
parameter values exist.** `normal(0, sigma)` qualifies; `normal(m, sigma)` with a parameter m does not, even if m happens to
be zero. The choice of method must not depend on a parameter value, or it could change during estimation.

**14. Starting distribution, for now:** the `initial` priors if every variable has one (they must be normal for now); otherwise
the long-run distribution, when every shock is normal and the model has no outside variables; otherwise none, and estimation
must start from the data. Giving `initial` for some variables but not others is an error.

**15. The posterior problem gets its solver as an argument (a function from parameter values to a solved model).** This keeps
the estimation layer independent of how models are solved, as the import rules require. The wiring lives in
`mutatis/versions/pipeline.py`, the one module that knows every layer.

**16. The method for handling shocks is chosen automatically from the specification and the data alone:** read the shocks off
the data if every series is observed exactly, there are as many shocks as series, the starting state can be read from the data,
and nothing is missing; otherwise the Kalman filter if the model is linear with normal noise; otherwise sample the shocks if
every series has measurement error. A method can also be forced, which is how the methods are tested against each other.

**17. The starting state is read from the data whenever it can be** (every part of it is a variable observed exactly, an
outside variable, or a lag of one), and the model explains the periods after it. Otherwise it comes from the model's starting
distribution. This matches standard practice for VARs and equation-by-equation models, and lets read_off and Kalman be
compared on exactly the same periods.

**18. When the shocks are sampled and the starting state is not read from the data, the sampler moves a standard normal vector
u, and the state is mean + L u.** The starting covariance is often singular (the price in the forward-looking toy is an exact
multiple of demand), so a density for the state itself need not exist. L is a Cholesky factor with a jitter of 1e-12 relative
to the largest variance.

**19. The stationary covariance uses a fixed 60 doubling steps instead of a loop that stops when settled.** JAX cannot
differentiate backwards through a loop whose length depends on the data, and gradient samplers need these derivatives.

**20. Choosing a sampler:** sequential Monte Carlo when every prior is proper; the exact regression formula when a prior is
improper (flat or Jeffreys) and the model qualifies; otherwise an error that says why. SMC cannot start from an improper prior,
because it draws from it.

**21. Whether the exact regression formula applies is checked numerically on the model, not inferred from its kind.** The checks:
shocks read off the data; one-step-ahead errors exactly linear in the flat-prior coefficients (compared at a second random
point); every coefficient used by exactly one series' error; every series using the same regressors; the errors' covariance equal
to the one covariance parameter. This keeps the formula free of any notion of "VAR" while covering VARs and SVARs.

**22. A stored result is keyed by the specification after edits, a fingerprint of the data, every sampler setting including the
defaults in force, and the seed.** Running the same thing again returns the stored result. Including the defaults matters: a
cached result must not be reused after a default changes, which happened once during development (the number of SMC moves).

**23. SMC defaults: 5 Hamiltonian moves of 5 leapfrog steps per temperature step.** Down from 10 and 10, which took 5 minutes on
the three-parameter forward-looking toy (150 periods, 1,000 particles); now 1.5 minutes, with the same posterior and the sampler
tests on known answers still passing. Most of the remaining time is the Kalman filter's gradient (0.35 s per gradient over 1,000
particles), so speed is a known issue for later stages.

**24. Data transforms are expressions in the raw series `x`,** such as `"100 * log(x)"` or `"d(log(x))"`, instead of a fixed list
of named transforms. Read (2022) needs 100 × log, and a fixed list would keep growing.

**25. Data are not stored in the repository; a demonstration's manifest says where to download them.** The manifest gives a URL,
a checksum, the file inside a zip, the sheet, and the sample. Files are cached outside the repository (~/.cache/mutatis). The
first case is Read's VARData.xlsx: the RBA licenses its papers under CC BY 4.0 but excludes third-party material, and the file
compiles ABS, US and academic series. Checksums must be quoted in the YAML, because a checksum made only of digits would
otherwise be read as a number.

**26. Problems with sampled shocks use NUTS chains, not SMC; their marginal likelihood is not reported.** Evidence (risk
experiment 1, `experiments/sample_route_diagnosis.py`): on the hidden-level toy with 200 periods, a NUTS chain on the sampled-shock
posterior matched the Kalman route (sigma_eta 0.232 against 0.233), while SMC gave 0.097, 0.132 and 0.172 with moves of 5x5,
20x10 and 50x20 steps. On a 200-dimension normal target SMC's posterior was roughly right but its log marginal likelihood was off
by +19.7. Standardising the shocks (now done anyway) did not change this. The cost: NUTS does not move between separated peaks
and gives no marginal likelihood, so versions estimated this way cannot be compared by marginal likelihood until SMC improves.

## Stage 1

**27. Whether ranges can use the exact one-column method is checked by evaluating at rotations only, never by derivatives in
every direction.** Read's sign normalisation reads a row of the inverse impact matrix, (chol(Sigma) Q)^-1 = Q' chol(Sigma)^-1,
which depends on one column of Q only because Q is orthogonal. Derivatives with respect to unrestricted changes in Q show it
depending on every column, which is what the first version of the check did, wrongly refusing Read's model. The check now fits
each function as linear in one column by least squares at n + 6 random rotations, and verifies the fit at fresh ones.

**28. Read's published range results are exactly zero at some horizons where ours are slightly positive; ours are taken as
correct.** Under Restriction (1), the set of posterior means for the cash rate's response starts at exactly 0 in Read's figure
data at horizons 0 to 3; here it starts at 0.0009, 0.0043, 0.0003 and 0.0001. The bounds computed here are exact: each is attained
by an allowed unit vector (tested), and an independent optimiser (SciPy's SLSQP from 200 starting points) found exactly the same
minima on every draw checked, for example 0.026 rather than 0 for one draw at horizon 1. Read's `analyticalBounds.m` tests
`v == 0` with exact floating-point equality where a restricted response's value function is zero in theory but about 1e-17 in
practice, and then uses a direction computed from rounding noise; it also writes the zero-value case into the slot for no
active restrictions (`fMax(hh,1,ii)`) inside the loop over active sets. Either could produce exact zeros (moderate confidence;
MATLAB was not available to step through it). The stage 1 test accepts only this kind of difference, and prints each one.

**29. An edit file may include another (`- include: restriction1.yaml`),** so a restriction set built on another is written as
"that set plus these", not copied.

**30. Interval summaries follow Read's definitions exactly** (shortest centred interval over the draws; robust credible interval
on a 1,000-point grid of centres; MATLAB's percentile definition, "hazen" in NumPy), so results can be compared number for number.
Tolerances for comparing with a reference come from sampling error on both sides; the ends of 68% intervals use the standard
error of the 16th and 84th percentiles of a normal (1.51 sd / sqrt(N)). Observed on Restriction (1): posterior means agree with
standardised differences of sd 0.82; interval ends with sd 1.7, which suggests this standard error understates the noise of
shortest-interval ends.

**31. README's paragraph on the contract between a model and its estimation now describes what was built:** separate log prior
and log likelihood functions, prior draws, and a layout of the unknowns, instead of one joint-probability function. This was
flagged on 2026-11-25 as out of date after the plan changed the contract, and had no answer; it was updated under the instruction
to decide, document and move on.

**32. Stage 1's test for range results requires our bounds to be exact, not to match Read's published range statistics; this
changes PLAN.md's acceptance criterion for ranges (confirmed in discussion 2026-11-27: exact bounds; where Read's are found by
random rotations, as for Restriction (6), ours must contain his).** Evidence (`experiments/read2022_differences.py`):
under Restriction (2), on 15 posterior draws and 6 horizons, an independent optimiser never found an allowed value outside our
bounds (largest gap 1.8e-11), while Read's published lower ends at restricted horizons are about 0 where the exact bounds are
clearly positive (0.276 in one draw for the exchange rate at horizon 2). Since his range numbers are not the exact bounds, they
cannot be the standard. The comparison with them is still printed for every run, so the size of the difference stays visible.

**33. Comparison tolerances come from a bootstrap over our own stored draws, not a formula.** The formula used before (1.51 sd /
sqrt(N) for the ends of 68% intervals) understated the noise about threefold: across five seeds, our own interval ends for
Restriction (3) moved with sd 0.025, 0.022 and 0.016 where the formula said 0.0078, 0.0078 and 0.0059. Draws of every reported
quantity are now stored with each result (`quantities.npz`) for this.

## Numerical audit (2026-11-26)

**34. The derivative of the forward-looking solve solves its matrix equation by doubling, not by forming an n² × n² matrix.**
Differentiating A P² + B P + C = 0 gives (A P + B) dP + A dP P = R. It was solved by building the n² × n² matrix of that map
and factoring it, whose cost grows with n⁶ and memory with n⁴ (328 MB for one model of 80 variables). It is now rewritten as
dP = G + F dP P and summed by doubling (`numerics.linear_re.solve_sylvester`), using only n × n products; the sum converges
exactly when the solution's roots have a gap at the unit circle, which the solve needs anyway, and takes about as many steps
as cyclic reduction (8 for well-separated roots, 24 for roots 1e-6 apart around 1). Evidence
(`experiments/forward_looking_solve_cost.py`): one solve with its gradient took 1,320 ms at 80 variables and 69 ms at 40
before, 58 ms and 14 ms after, so the gradient now adds 6% or less to the solve; 1,000 solves with gradients at 20 variables
took 4.6 s before and 1.5 s after; at 80 they now take 65 s, and before they did not fit in memory. Against the dense form,
the answers differ by at most 0.6 machine epsilon times the larger of the dense system's condition number and the series'
error amplification, on 12 cases from 1 to 20 variables with root gaps down to 1e-4. The alternative was Bartels–Stewart
through Schur decompositions, also n³, but JAX's Schur decomposition runs only on the CPU and in a sequential loop over a batch.
The dense form stays only as a test reference (`tests/numerics/reference.py`).

**35. The forward-looking solve is checked against SciPy's ordered QZ (Klein's method) at up to 80 variables; where the two
largest-and-smallest roots are 1e-4 apart around the unit circle, only the residual is required.** Medium-tier test
`test_agrees_with_ordered_qz`: complex roots, static equations and unlagged variables, unit roots, root gaps of 1e-2 to 1e-4,
and both kinds of failure, at 10, 40 and 80 variables. In a wider scratch run (144 cases, 2 to 80 variables), the
classification agreed every time and P agreed with QZ to 1e-8 relative except at gaps of 1e-4, where it differed by up to
3.6e-8. That difference is the problem's, not either method's: computing the exact answer for those systems (Newton's method
with residuals in 40-place arithmetic) put this solve 1e-9 to 7e-9 and QZ 1e-9 to 3e-8 away from it, with the derivative's
equation having condition numbers of 1e10 to 1e13. Elsewhere this solve is less accurate than QZ at 40 to 80 variables: 1e-10
to 1e-9 away from the exact answer where QZ was 4e-12 to 9e-11, with residuals about 50 times larger. That is far below what
estimation can notice, and one Newton step did not reliably improve it, so it is recorded rather than fixed.

**36. The Kalman filter recomputes each period's step when taking derivatives, instead of storing its intermediate
matrices.** Reverse-mode derivatives through the filter kept about seven n × n matrices per period per particle. Evidence (XLA's
own memory estimate of the compiled program): for 50 states, 150 periods and SMC's default 2,000 particles, the gradient needed
43 GB of working memory, more than this machine's 30 GB, so estimating a model of DSGE size would have failed. With
`jax.checkpoint` on the period step it needs 6.5 GB (one covariance per period, plus a little). Time was unchanged at 20 and 50
states (4.66 s against 4.68 s, 4.31 s against 4.28 s for a batch of gradients) and 7% slower on the 2-state toy; gradients
agreed to 1e-13 relative. A fast test checks the compiled program's memory so this cannot silently return. The alternative,
evaluating particles in smaller batches, would bound memory too, but it lives in the sampler and leaves each gradient's
memory as large.

**37. The stationary covariance stops when settled and takes its derivatives from the equation it solves; this replaces
decision 19.** Decision 19 fixed 60 doubling steps so that reverse-mode derivatives could pass through the loop, which meant
storing every step: for 50 states and 2,000 particles the gradient needed 9.9 GB and 7.4 s. Now the sum stops once a block no
longer changes it (`linear_re.sum_by_doubling`, the same routine the forward-looking solve's derivative uses), and the
derivative solves dX = T dX T' + dT X T' + T X dT' + dW by the same doubling (a custom derivative rule, as for the
forward-looking solve). Evidence: 0.36 GB and 0.93 s for that gradient; the value took 0.48 s instead of 1.38 s. Accuracy
against SciPy's solve_discrete_lyapunov is unchanged (residuals at rounding level up to a largest root of 0.99999).
Derivatives match finite differences and agree between forward and reverse mode (fast tests), and are zero, not NaN, where
the covariance does not exist.

**38. Why SMC fails with many unknowns, as far as found; SMC is left unchanged pending a design decision.** Evidence on
Gaussian targets with a known marginal likelihood (`experiments/smc_dimension_study.py`, `tests.infer.targets.NormalMean`, 2,000
particles, 3 seeds): the log marginal likelihood is right to within 0.4 up to 50 unknowns, then +27 to +29 too high at 100 and
+4 to +21 at 200, while posterior means stay within 0.08 posterior standard deviations. Found, one change at a time (scratch
experiments, recorded in the audit report; 100 unknowns and one seed unless stated):
- *Not the adaptive temperatures:* fixing the schedule from an earlier run gave +27 to +31 (4 seeds).
- *Not the moves' validity:* BlackJAX's HMC move left exact draws of the tempered posterior unchanged (spread 100.00 against
  100), and one SMC step from exact draws kept the spread within 0.3%.
- *The trajectory length:* the step size settles near 0.65 to 0.85, so 5 leapfrog steps make a trajectory of 3.3 to 4.3, close to
  half the orbit period (pi) of a normal posterior in the mass matrix's units. A move then nearly reflects each particle through
  the centre and keeps its distance from it: after 5 moves that distance's correlation with its starting value was 0.87 (0.00 for
  a trajectory of 1.5). The acceptance rate swings between 0.4 and 0.95 as the step size crosses this, so the step-size rule
  oscillates. The cloud ends up 3 to 10% too close to the centre (squared standardised distance) and each step's increment
  comes out +0.1 to +0.6 too high. With the exact mass matrix and a fixed step: a 3.25 trajectory gave -17 to -33 (high
  variance), a 1.5 trajectory -0.07 to -0.23 (3 seeds each).
- *A random trajectory length is not enough:* drawing each move's number of leapfrog steps uniformly from 1 to ceil(pi / step
  size) gave -1.73, -2.37 and +0.22 at 100 unknowns and within 0.7 at 1 to 50, but +39 to +47 at 200 (3 seeds), worse than now.
  At 200 the cause is the mass matrix copying each unknown's spread from the particles: with the exact mass matrix, or the
  particles' spread averaged over unknowns, the same run gave -5.2, while 8,000 particles did not help (+33). A plausible reading
  (not proven): an unknown in which the cloud is by chance too narrow gets shorter moves and so stays narrow.
- *The sampled-shock route (decision 26) fails differently:* random lengths alone changed nothing (sigma_eta 0.127 against 0.232
  from the Kalman route). A full covariance mass matrix from the particles, with random lengths, gave 0.193 with 5 moves and
  0.215 with 20 (log marginal likelihood -210.9 against -210.3): the posterior ties sigma_eta to all 200 standardised shocks,
  which a diagonal mass matrix cannot follow.
Nothing is changed in SMC: every fix tried helps one case and not another, and the remaining choices (full covariance mass
matrices, regularising the per-unknown scales, a number of moves set by a measure of mixing, or NUTS for these problems) change
every SMC result and need a decision.

**39. The Kalman filter's covariance update: switch from the Joseph form to the standard form only if the standard form
passes checks with exactly observed series (agreed in discussion 2026-11-26; checked 2026-11-26: it failed, so the Joseph form
stays).** The audit found that the standard update makes likelihood gradients 1.6 to 1.7 times faster at 20 to 50 states. The
Joseph form is in use because it keeps the covariance positive semi-definite when some series are observed exactly, where the
standard form can lose that through rounding. So the switch is made only if, on cases with exactly observed series, singular
state noise, roots near the unit circle and long samples, the standard form keeps the covariance valid and gives the same
likelihood and gradients as the Joseph form. Otherwise the Joseph form stays.
The check (scratch; the standard form written as P - U'U with U = L^-1 Z P, F = L L'): 160 cases with 2, 20 and 50 states, 1 to 49
exactly observed series, state noise of rank 1 to full, largest roots 0.99 to 0.99999, 1,000 and 5,000 periods, none or 20%
missing, starting from the stationary covariance. Required in advance: every filtered covariance's smallest eigenvalue at least
-1e-10 times its largest; likelihood within 1e-9 and gradients within 1e-7 of the Joseph form's; nothing NaN or infinite.
- *Covariance: failed.* In 11 cases the standard form broke the limit where the Joseph form did not. Worst: 20 states, 19 observed
  exactly, noise of full rank, root 0.99999, 1,000 periods. Smallest eigenvalue -1.2e-8 times the largest, and past the limit
  in 153 of the 1,000 periods, not only the first; the Joseph form's worst there was -4.5e-16. Also 50 states with 49 observed
  exactly: -7.0e-10 (full-rank noise), -6.0e-9 (rank 49).
- *Cause:* the standard form subtracts from the predicted covariance, so its rounding error is a fraction of the predicted
  covariance (at most 7.4e-13 of it in the 11 cases). Exactly observed series can leave a filtered covariance a thousand to a
  million times smaller than that, and then the same error is large next to it. The Joseph form's error stays proportional to
  its result.
  One update checked against 50-digit arithmetic (2 states, root 0.99999; the exact smallest eigenvalue is 0): Joseph +2e-19,
  standard -1.0e-11 (-1.07e-10 of its largest), and the textbook way of writing it, P - K Z P, -1.7e-11.
- *Where both fail:* with as many exact series as shocks, the filtered covariance can shrink towards zero as the state becomes
  known, and then both forms break the limit (19 cases). In one such case per size, the Joseph form did so only in periods where
  the filtered covariance was below 2e-13 (2 states), 1e-10 (20) or 5e-10 (50) of the predicted one; the standard form already
  at 1.3e-6, 5.2e-6 and 2.5e-6. With more exact series than shocks the model has no likelihood; both forms gave NaN within 7
  periods.
- *Likelihood, gradients, NaN: passed.* Likelihoods agreed to 1.8e-12 relative; gradients to 4.6e-8 of each block's largest
  entry; nothing NaN or infinite. Entry by entry, small gradient entries differ by up to 4.4e-4 of themselves; finite
  differences were too coarse to say which form is right there.
- *Speed* (gradient with respect to T, W, Z and H, vmapped over 2,000 parameter values, 150 periods): 0.73 s against 0.49 s at 2
  states, 9.1 s against 5.8 s at 20, 43.1 s against 28.6 s at 50 (Joseph against standard: 1.5 to 1.6 times). Compile times
  similar (0.4 to 1.9 s).
Revisit if gradient time at DSGE size becomes the limit. Alternative: the standard form, 1.5 times faster, with covariances
accurate relative to the predicted covariance rather than to themselves.

## General sampler (2026-11-26)

**40. The evidence from a tempering ladder uses a geometric bridge between neighbouring rungs, and standard errors come from
effective sample sizes, not from the spread of replicas.** First version: the average of the one-sided "forward" and
"backward" stepping-stone estimates. On a 5-unknown normal target with exact draws at every rung it was 0.098 below the exact
log evidence with a reported standard error of 0.021. Per pair of rungs, the forward estimates summed to an error of -0.004 and
the backward ones to -0.162. Cause: the backward average, of exp(-dbeta x ell) over the hotter rung's draws, has infinite
variance unless that rung's beta minus dbeta stays above about -1/(n x prior variance) (here -2e-4), which a ladder spaced
to even out its exchanges breaks at most pairs. The geometric bridge (exp(+dbeta/2 x ell) over the cooler rung, exp(-dbeta/2 x
ell) over the hotter) has finite variance on both sides whatever the spacing, and both sides see only places with positive
likelihood, so the reference end needs no special case. Over 8 seeds its errors were -0.059 to +0.060. Second finding: a
standard error from the spread of 4 replicas has 3 degrees of freedom and is itself too noisy (one seed gave 3.5 standard
errors by chance); the delta method with each rung's effective sample size uses all the draws. Bridge sampling likewise uses
Fruhwirth-Schnatter's (2004) error approximation. Alternative: forward-only stepping stones (Xie and others, 2011), which
also has finite variance here but uses one side of each pair.

**41. The general sampler: step rule x tempering ladder x replicas and starts x checks (agreed in discussion 2026-11-26).**
`mutatis.infer.mcmc.mcmc`. Defaults, chosen to work rather than to be fast: NUTS steps; a parallel-tempering ladder whose
spacing and number of rungs are set during warm-up from the communication barrier (non-reversible exchanges, Syed and others
2022), ending at the prior when it is proper and otherwise at a normal distribution fitted by Pathfinder (variance inflated
fourfold); 4 replicas started from Pathfinder draws; 1,000 warm-up and 1,000 recorded scans. Checks: rank-normalised R-hat
above 1.01, bulk effective sample size below 100 per replica, any divergent trajectory at the posterior rung, fewer than 2
round trips per replica, fewer rungs than twice the barrier; any finding sets `converged` False. NUTS as usually run is the
named setting `nuts` (one rung, prior starts), replacing the separate NUTS module. The evidence is a separate, optional step
(`mutatis.infer.evidence`, decision 40). SMC is kept, for comparison. Not yet changed: `estimate()`'s automatic choice
(exact regression, NUTS for sampled shocks, otherwise SMC), pending the benchmark. Alternatives: SMC with better moves; NUTS
alone, which cannot weigh separated peaks.

**42. The sampler's default settings are chosen from the problem and the hardware, and it never stops on a draw count (a
try-out, 2026-11-27, pending review of the short benchmark suite).** Asked for in discussion: as many choices as possible from
the data; stopping only by convergence tests, with a time limit that, when reached, returns the results loudly marked as
suspect; the hardware used to the full. In `mutatis.infer.mcmc.mcmc`, by default:
- *What counts as a peak* (revised 2026-11-27 after the first short suite): a climb from a prior draw (L-BFGS, then Newton
  steps) whose end point is certified: minus the Hessian is positive definite there, so the density falls in every direction,
  and half the Newton decrement, the further rise the curvature predicts, is below 1e-8. End points that cannot be certified
  are left out. The first try-out counted every optimiser end point, and reported false peaks: 2 or 3 on the strong banana and
  6 on the 21-unknown regression, from optimisations that had stopped partway (5 of the 6 had log densities of -843 to -1,023
  against -393.7 at the true peak). With the certificate: 1 peak on the bananas, the funnel and the regression, 2 on
  two_peaks_d1 (tests/infer/test_mcmc.py). On the sampled-shocks model (hidden_level_shocks_T50) no climb is certified (0 of
  1,024): climbs stall where the observation noise is tiny (sd 0.004 to 0.009) at log densities of 127 to 160, a different
  value for each start, against about -5,000 at prior draws. That fits a density that rises without bound as the noise goes
  to zero with the hidden path matching the data (likely, not proven); no peak is reported rather than a false one. The certificate needs the full Hessian, so there is no search above 200 unknowns (the same limit
  as the full mass matrix); a matrix-free certificate (Newton steps by conjugate gradients) would lift that, not built.
- *Telling peaks apart*: two certified peaks are separate if the log posterior density dips more than 2 below the lower of
  the two somewhere on the straight line between them (or is not finite there). 2 log units is a factor of about 7 in
  density: small enough to separate peaks that matter, large enough that the ripples of one peak do not count. Because only
  certified peaks are compared, a single curved peak (one highest point) cannot count as several, though the straight line
  between two of its other points can dip.
- *How much to search*: climbs in rounds spread over the processes, until there have been enough starts to find, with 95%
  probability, any peak reached from at least 1 in 10,000 prior draws (N = ln 20 / 10^-4, about 30,000) and the last round
  found no new peak; or until 10% of `max_seconds` has gone (time for compiling the climb included). The reason for sizing
  it from a detection limit rather than a fixed count: the number of starts then says what the search can promise. With N
  starts, a peak whose catchment (the prior draws from which a climb reaches it) holds a share p of the prior is missed by
  every start with probability (1 - p)^N, at most exp(-p N); so a peak with p of at least ln 20 / N is found with 95%
  probability. Every result states it: "searched N starts: a peak reached from fewer than 1 in M prior draws could have been
  missed" (M = N / ln 20), or that there was no search. The narrow peak of two_peaks_d5 holds 1 in 8,800 prior draws, so
  30,000 starts find it with about 97% probability; in 20 dimensions it holds 1 in 9 x 10^13, beyond any search from prior
  draws, and the statement says so. Cost, measured: 30,000 climbs took 10 s on two_peaks_d1 in one process (climb
  compiled once, 2.6 s), 8 s in 8; about 0.2 to 0.6 ms a climb on the cheap targets, 30 ms on the 21-unknown regression, and
  0.4 s on the forward-looking model (whose climb takes 28 s to compile), where the 10% cap, not the count, ends the search.
- *Ladder or not* (replaced as the default by decision 43; kept as `rungs="decide"`): plain chains if the search found one peak or none; the ladder (as decision 41) if it found several, with the
  replicas started round robin over the peaks. During warm-up without a ladder, the ladder is switched on (and warm-up
  restarts from the chains' positions) if the chains' window means lie in separate peaks by the same valley test, or if
  after 1,000 scans of slow windows they still disagree (R-hat over the window above 1.05). Only during warm-up, so that no
  kept draw comes from a sampler that was still changing. Rung count and spacing: as decision 41.
- *Mass matrix*: full (correlations included) up to 200 unknowns, one scale per unknown above that; Stan's regularisation.
- *Warm-up length*: Stan's windows, open-ended: first 75 scans, slow windows from 25 doubling up to 3,200, last 50; warm-up
  ends after a slow window (at least two) in which the chains agree (R-hat over the window below 1.05) and the automatic rung
  count has settled (changed by no more than 1 rung or 10%). Without the cap, chains that never agree (NUTS in two peaks)
  had planned windows of 13 million scans.
- *Chains without a ladder seen in separate peaks*: after a slow window in which the chains disagree (R-hat over the window
  at least 1.05) and the valley test puts their window means in separate peaks, the sampler climbs from each chain's position
  and counts the certified peaks reached. Testing the window means directly, as first tried, switched the ladder on for the
  mild banana (one curved peak) in both seeds. With `rungs="decide"` two or more peaks switch the ladder on; where no ladder
  is allowed (`nuts`, `rungs=1`) the run is reported as not converged, because such chains cannot weigh peaks, and chains
  that later all fall into one peak would pass every other test (seen in the benchmark: NUTS on two peaks in one dimension
  did exactly that once windows were capped).
- *Memory*: at most 10,000 draws are kept per chain; past that the kept draws are thinned to every second one, and every
  second scan is kept from then on (doubling again as needed). The tests are judged on the kept draws, which can only make
  them stricter; divergences and swap acceptance are counted over every scan. Without this, cheap targets that do not
  converge kept millions of draws per chain within 600 s (3.6 to 4.1 GB per run, and later steps on them took longer than
  the run).
- *Run length*: blocks of at least 100 scans, each at least half of all before it, until every unknown has R-hat below 1.01,
  bulk effective sample size at least (1 / precision)^2 = 1,600 at the default precision 0.025 (Monte Carlo error of every
  posterior mean at most 2.5% of its posterior sd, chosen by Peter so that the benchmark's accuracy threshold of 0.1 sd is
  four Monte Carlo errors), tail effective sample size at least 400, and with a ladder at least 2 round trips per replica,
  counted from the ladder's last respacing (not reset at the start of sampling; see parked item 2 on why the old count ran
  short). `max_seconds` (default 3,600, the whole run): when reached, the draws so far are returned with `converged` False, a
  `status` naming the unmet tests, and a `NotConvergedWarning` starting "NOT CONVERGED — RESULTS ARE SUSPECT"; the versions
  summary records the status and the command line prints it in a banner.
- *Chains and cores*: one replica per usable CPU (at least 4, at most 64), each group of replicas in its own process pinned
  to its own CPU (`mutatis.numerics.parallel`); 4 in this process when there is one CPU; 64 in this process on a GPU or TPU (a
  rule of thumb, not measured). Measured on 8 CPUs (4 cores with 2 threads each): in one process, vmapped chains do not
  spread over the cores (time per chain-step flat from 1 to 64 chains, CPU over wall time 1.0); splitting them over XLA host
  devices in one process was slower than one device for a Kalman-filter likelihood (4 chains on 4 devices 1.0 s, against
  0.17 s for one chain alone), because its small linear-algebra calls contend between threads (a loop of 2 x 2 Cholesky
  solves: 4 devices from one thread 4 times as long as one, from 4 threads 6 times); separate pinned processes ran at full
  speed (one chain 0.17 s alone; 7 or 8 at once 0.23 to 0.42 s each).
  Programs are compiled once, in the main process, and sent compiled (`jax.experimental.serialize_executable`); the search's
  climbs and the Pathfinder runs go through the same processes. The first try-out sent each worker a `jax.export`ed program
  to compile itself, and a whole run with 6 workers was no faster than with one process. Measured cause, on the forward-
  looking model: export 4.9 s, then compiling took 17 s in one pinned worker and 29 s in each of 6 compiling at once, against
  12 s in the unpinned main process. Not the warm-up windows: chunks always have 25 scans, so a run with one rung needs one
  program. Loading the compiled program takes a worker 0.5 to 0.8 s, and it gives bit-identical results. Two things the
  workers need: LAPACK loaded before the first call (JAX loads it when it compiles a linear-algebra call; a worker that only
  loads compiled programs crashed without it), and programs compiled with JAX's persistent compilation cache bypassed (a
  program loaded from the cache is sent incompletely, and fails in the worker with "Function ... not found"). So with
  workers there is no reuse of compiled programs between runs; in one process the cache is used as configured.
  Whole runs (max_seconds 600, as in the benchmark), one process (4 chains) against 8 (8 chains):
  normal_means_d10 69 s against 28 s (search 58 s for 30,200 starts against 18 s for 32,704);
  hidden_level_shocks_T50 92 s against 57 s (search stopped by its 60 s cap after 15,824 starts, against 32 s for 32,704);
  forward_looking 195 s against 142 s (search capped at 60 s either way: 64 starts against 192; sampling 67 s against 35 s
  for more draws, about 2.5 times the throughput; warm-up 44 s against 26 s). The rest is serial: compiling the climb (28 s
  on the forward-looking model), Pathfinder (13 s) and chunk (12 s) programs in the main process, one after another; and
  sampling speeds up by about 2.5, not 8, because 8 CPUs are 4 cores with 2 threads each and each chunk waits for the
  slowest worker. Not yet tried: compiling the programs side by side while the search runs.
- The stopping rule is a parameter of the one sampling loop: a fixed-length rule exists only for the unit tests of the
  ladder's machinery, which compare against formulas for fixed lengths.
- *The convergence diagnostics* (R-hat, bulk and tail effective sample sizes) are computed in NumPy
  (`mutatis.infer.convergence`), a port of BlackJAX's that matches it to 1e-9 on independent, autocorrelated, tied, stuck
  and constant draws (tested). BlackJAX's JAX versions compiled afresh for every new number of draws, about 11 s a call
  (7 s under `jax.jit`), which made the checks after each block cost more than the sampling on cheap targets; NumPy takes
  milliseconds.
Named settings: `nuts(problem, key)` (one rung, prior starts, 4 replicas) and the previous default as `mcmc(rungs="auto",
starts="pathfinder")`, both with the same stopping. `estimate()`'s automatic choice is still unchanged (parked item 2).
Alternatives: a fixed draw count (rejected in discussion); one replica count for all hardware; running chains on several XLA
devices in one process (slower for the Kalman filter, as measured).

**43. The default sampler always uses a ladder, and keeps the peak search (agreed in discussion 2026-11-27, after the short
benchmark suite).** In `mutatis.infer.mcmc.mcmc`, `rungs="auto"` is now the default; decision 42's rule (a ladder only if
the search finds several peaks, or warm-up sees the chains in separate peaks) stays available as `rungs="decide"`. Why:
correctness before speed. In the short suite (`benchmarks/samplers/results/short-3`, 38 runs per method), decision 42's rule
was confidently wrong on 5 runs: the 20-dimension narrow peak twice (every method was), the 5-dimension narrow peak once
(both peaks found and a ladder used, but weights 0.25 / 0.75 against 0.3 / 0.7, in a run that stopped as converged after
1,025 draws per chain), and each banana once. On the bananas the certified search rightly found one peak, so there was no
ladder, and plain chains passed every convergence test with the spread 13 to 14% too narrow (with 186 to 450,900 divergent
trajectories, which the tests do not count); in the run before, false peaks had switched the ladder on and all four banana
runs were right. On the held-out Inference Gym targets (`results/held-out-2`) the same rule was confidently wrong on the
banana (one seed of two), Brownian motion with unknown scales (both seeds: the observation noise's mean off by 0.26 to 0.32
sd, the scales' spread about 20% too narrow) and the small stochastic volatility model (both seeds, in the spread of one
quantity with a Cauchy prior, whose reference spread may itself be off). A ladder always (decision 41's default) was
confidently wrong in short-3 only on the 20-dimension narrow peak. Cost: a ladder in every run, about 1.75 times decision
42's total run time in short-3 (2.6 h against 1.5 h); where the prior is much wider than the posterior the ladder needs
many rungs (barrier 22 on 50 normal means, about 47 rungs), and on two short-suite targets warm-up did not finish in 600 s
(open: how many rungs, and where the ladder starts; parked item 2). The search is kept: it puts the starts on the peaks found
and states its detection limit, at a cost of at most 10% of `max_seconds`. When this was decided, the search with a ladder
always had not been benchmarked together; the short suite is rerun with it, against the same with the ladder from a fitted
normal (`reference="fitted"`). Alternatives: decision 42's rule with divergent trajectories also switching the ladder on
(not tried; it would switch it on for the sampled-shocks model too, where the ladder takes about 4 times as long); a ladder
always without the search (decision 41).

**44. The exact regression formula works block by block, each block with its own sample (agreed in discussion 2026-11-27, for
Read's Restriction (6)).** `mutatis.infer.regression` reads each block's errors from its own equations at the data, with its
shocks set to zero, instead of from the solved model's one-step-ahead errors. Conditions, each checked numerically on the model:
every variable can be read from the data; within a block the errors are linear in coefficients with flat priors and every
equation uses the same regressors; each block's errors have a covariance parameter of their own (Jeffreys or inverse Wishart)
and are independent of other blocks'; each block's equations give its outputs one for one, and no blocks depend on each other's
current values in a circle (so the change of variables from errors to data has a Jacobian of one); anything else leaves the
likelihood unchanged. A block uses the periods in which every value it needs is present; a variable with missing values must be
an output of one block, used by it only in the current period and by no other block, so that leaving out that block's periods
drops exactly the missing values' densities. A VAR is the one-block case (all earlier tests unchanged). Checked: with a series
missing in 40 of 159 periods, the product of the block densities equals the Kalman filter's likelihood of the whole model to
1e-8, for a VAR with a survey block and for a chain of two single-equation blocks (MARTIN's kind of model); for Read's model the
VAR block (140 quarters, 32 regressors) and the proxy block (1994Q1 to 2018Q4, 100 quarters, 36 regressors) match his own
construction to 1e-12. The same mechanism is meant for *isolate* (stage 5) and MARTIN's equation-by-equation estimation
(stage 6). Alternative: a special path for a first-stage regression, rejected as model-specific.

**45. Ranges whose restrictions involve several columns of the rotation come from a search over rotations.** When the exact
one-column method does not apply, `mutatis.infer.identification.rotation_bounds` finds each bound by a log-barrier method with
Newton steps on the rotations themselves (local coordinates Q cayley(theta), re-centred after each step; positive curvature has
its sign flipped), from the best few of 20,000 random rotations that satisfy the restrictions, or, if none does, from a first
phase that maximises the smallest restriction value. Restrictions and quantities linear in the rotation (every SVAR restriction
met so far) are fitted as matrices per draw, so the search is compiled once. Checked against the exact one-column method where
both apply (agreement to 2e-12, never outside), against an exact reduction of Read's proxy restrictions to a convex problem in
the first column (an orthonormal completion with |a'q_j| <= a'q_1 exists exactly when a'q_1 >= |a| / sqrt(n)), including which
sets are empty, and on a set holding about one rotation in a million (found by the first phase). It replaces the earlier
SLSQP prototype (`optimised_bounds`), which used the matrix exponential and recompiled for every call. A bound can still be
missed where every start leads to a lower local best, and on Restriction (6) it was (decision 53 adds a second search); an
allowed set too small for the search is reported as empty.

**46. Read's published results for Restriction (6) come from replication code that pairs the proxy with the wrong quarters; the
demonstration follows the paper, and the stage 1 test reproduces his figure by giving the same pairing as data.** In
`mainfile_proxy.m` the proxy is first shortened to its 100 values (`proxy = proxy(~isnan(proxy))`), and the next line selects
the VAR's rows with `YY(~isnan(proxy),:)`: a mask of 100 true values, which MATLAB applies to the first 100 of the 140 rows. So
the first-stage regression pairs the proxy for 1994Q1-2018Q4 with the VAR's values for 1985Q1-2009Q4, 36 quarters earlier. The
paper describes the aligned regression ("use the sample over which the proxy is available to estimate the first-stage
regression"). Evidence: a line-by-line NumPy port of his code (`mainfile_proxy.m`, `drawQ0.m`, `genVMA.m`; a scratch script,
not kept), 400 draws: aligned, plausibility 0.83 and impact responses cash rate 0.28, GDP -0.20, TWI 2.42, as our model gives
(0.876; 0.287, -0.195, 2.40); misaligned, plausibility 0.40 (20,000 rotations per draw) and cash rate 0.16, 0.21, 0.20, 0.23 at
horizons 0 to 3, GDP -0.16 and -0.31 at 0 and 8, against his figure's 0.15, 0.19, 0.20, 0.23 and -0.14, -0.30, and the paper's
plausibility of "around 45 per cent". Through the generic path, with the proxy moved 36 quarters earlier in the data (the same
pairing), all 252 conventional statistics of his figure 6 agree within sampling error and the plausibility is 0.454; aligned,
101 of 252 disagree. The model in `demos/read2022/edits/restriction6.yaml` pairs them as the paper describes; the test gives
the published pairing as data, so it checks that our machinery reproduces his published numbers. Fewer rotation tries per draw
were ruled out as the cause first: they lower the plausibility (0.45 at about 70 tries) but leave the responses unchanged.

**47. A specification's restrictions truncate the prior, in every sampler.** Until now only the exact regression formula applied
restrictions (to the unknowns the likelihood ignores); every other sampler silently ignored them, and so did the formula for a
model without such unknowns. Now the posterior problem evaluates them (`PosteriorProblem.restriction_values`): the log prior is
minus infinity wherever one fails, `draw_prior` draws until all hold (at most 10,000 tries per draw), and the regression formula
drops the draws that break restrictions on what it estimates. The ranges step uses the same evaluation. The truncated prior is
not renormalised, so a marginal likelihood computed with restrictions leaves out log P(restrictions hold under the prior).
Checked on the forward-looking toy with a restriction on the price's response (`tests/infer/test_restrictions.py`): the log prior
is unchanged where it holds and minus infinity elsewhere, prior draws follow the truncated prior, and by SMC the restricted
posterior is the plain posterior cut down (medium tier); and on a VAR whose restriction involves only its coefficients.
Alternative: refusing to run when a sampler cannot apply restrictions (safe, but leaves restricted models without a sampler).

**48. Ranges with any sampler (agreed in discussion 2026-11-27).** For a model the exact regression formula does not cover,
whose only unknown without a prior is a rotation the likelihood ignores (checked at prior draws), and whose other unknowns have
proper priors: those are sampled by the chosen sampler with the rotation held fixed and the restrictions left out, and for each
of `n_draws` posterior draws the ranges are computed as the formula computes them (`mutatis.infer.restricted.ranges_posterior`;
`estimate()` routes there). Checked on a forward-looking model whose two shocks are mixed by a rotation with no prior: the
bounds of the price's response equal the closed form rho^h sqrt(S11) / (1 - a rho) and 0 at every draw, to 1e-9. Ranges over
unknowns that are not rotations remain unsupported.

**49. Stage 2's textbook model is Herbst and Schorfheide's small New Keynesian model (agreed in discussion 2026-11-27).**
`demos/herbst_schorfheide2016`: the model of "Bayesian Estimation of DSGE Models" (Princeton University Press, 2016), chapters
1.1 and 4, written from the book's equations and the authors' public code (none of their files is copied), with their data
downloaded from the book website by the manifest (no licence for redistribution is stated). Checked against the authors' own
code, run in Octave 10.3: the log-likelihood and the log prior density at two parameter points agree to 1e-12
(`reference/likelihood.csv`). The model is also written in Dynare's language (`model.mod`); Dynare 6.5's first-order solution
of it at the authors' posterior mode agrees with ours, all 56 coefficients to 1e-10 (`reference/dynare_solution.csv`). The
posterior from the default sampler agrees with the authors' random-walk Metropolis-Hastings run, 200,000 draws in Octave
(`reference/posterior.csv`): every parameter's mean and standard deviation within 5 standard errors of the difference (the
largest, 1.2 for a mean and 2.7 for a standard deviation), in 35 minutes on 4 processes. Dynare 6.5 runs here from its source
release in a separate conda environment (Octave 10.3 with its statistics package; the preprocessor built from the same
release, since conda-forge's writes code for a different Dynare version; compiled helpers not built, Dynare's own Octave
fallbacks standing in, enough for first-order work). Nothing of Dynare is in the repository.

**50. Nonlinear models are solved to first order around their steady state, found from values the specification gives.** A
block may state `steady_state` values for its outputs: expressions in parameters and in the values given before them, where
Newton's method starts (`mutatis.numerics.rootfind`, derivatives through the implicit function theorem). The linear solve then
takes its derivatives at that point instead of at zero, with the constant f - (A + B + C) x_ss, so linear models are unchanged
and a nonlinear model's solution rests at its steady state; observation equations are linearised at the same point. Every
solution is therefore linear for now, and the choice of route no longer asks whether the equations are. Checked: a growth model
k = (1 - delta) k(-1) + s k(-1)^alpha + e against its closed form (steady state, persistence, linearised log observation, and
the derivative of the steady state with respect to s); a real business cycle model whose Euler equation takes the expectation
of a nonlinear expression, against Dynare 6.5 (steady state and solution, 20 values to 1e-9). Not yet: `steady_state` in
template blocks; nonlinear models with outside variables.

**51. Dynare files are imported through Dynare's own preprocessor (`mutatis.io.dynare`).** The preprocessor (GPL, a separate
program found from $MUTATIS_DYNARE_PREPROCESSOR or the path) writes the parsed file as JSON, which becomes a specification: one
block holding every equation (a Dynare file does not say which equation determines which variable); leads x(1) as E(x(+1)),
the same to first order; parameters fixed at the file's values or given its `estimated_params` priors (beta, gamma, normal,
uniform, and the inverse gamma for standard deviations, whose (s, nu) are found from Dynare's mean and standard deviation);
`steady_state_model` or `initval` as steady-state values; `varobs` as data series of the same names, with the measurement
errors the shocks block declares. Checked: the imported real business cycle model solves as Dynare does and as the
hand-written specification does (to 1e-12); the imported Herbst and Schorfheide model gives the authors' log-likelihood (to
1e-12) and log prior density (to 1e-8, through Dynare's mean-and-sd form of the inverse gamma). Not yet: leads beyond one
period, auxiliary variables, shocks given by variances or correlations, shifted priors.

**52. Two smaller additions for stage 2.** The prior family `inverse_gamma_sd(s, nu)`, a standard deviation whose square is
inverse gamma (Schorfheide's IG(s, nu), Dynare's `inv_gamma_pdf`), tested against SciPy and the authors' formula; and data
manifests for CSV files with no date column (`first_date`: rows are consecutive quarters from it).

**53. The search over rotations also treats the restrictions as a penalty that hardens step by step (agreed in discussion
2026-11-27).** On Restriction (6) the barrier search of decision 45 fell short of the exact bound on 25 of 504 bounds (3
posterior draws, 84 responses, both ends), by up to 11% of the range's width. There the extreme lies where every proxy
restriction binds at once (a'q1 = |a| / 2), and a barrier, which keeps every step allowed, pushes the search away from such
places: all 232 allowed starts of one draw ended at the same lower local best. (Read's own method for this set, the smallest and
largest response over 5,000 random allowed rotations, falls short by 10% of the width on average; his paper's Appendix A.2 says
it is too narrow.) `rotation_bounds` now also runs a softened search from 16 random rotations per bound, allowed or not
(`_soften`): the restrictions become a penalty, half the sum of squares of each one's shortfall below a margin of 1e-6 (in units
of its spread over random rotations); the quantity is divided by a weight that rises from 1 to 10^8 in nine stages; the steps
are decision 45's Newton steps on rotations (`_newton_on_rotations`, now shared by both searches). The end point, inside every
restriction by about the margin, is finished by the barrier search with a small barrier, so every reported bound is attained at
an allowed rotation. For each bound the more extreme allowed result of the two searches is reported. Evidence (scratch
experiments, not kept): on the 504 bounds, SLSQP over whole rotations from 20 random starts fell short on 2 and the softened
search (BFGS, then SLSQP) on none, 94% of its starts reaching the exact bound (68% on the 25 hard ones); the built-in version
falls short on none (largest gap 4.5e-11 of the width). Tests: two random cases with proxy restrictions where the barrier search
alone falls short by more than 1e-3 now match the exact reduction to 1e-8. Alternatives, rejected in discussion: an exact rule
for proxy restrictions (specific to one kind of restriction), and accepting bounds that may be too narrow (they understate
uncertainty without saying so). Parked item 4 reviews the optimisation methods together.

**54. Dynare files are imported as published, at full size; the housing model's likelihood equals Dynare's (stage 4, first
part).** The importer (decision 51) now handles what Gibbs, Hambur and Nodari's (2018) code uses: files included with
`@#include` (read from the model file's folder) and macro variables; model-local variables (`#`) as the block's definitions;
estimated shock standard deviations as parameters named `stderr_<shock>` with their priors; parameter values worked out
statement by statement in the file's order, as Dynare does, with values the file computes in MATLAB code (here 31, from a
steady-state solve) supplied by the caller, for example from the authors' stored results (`stored_parameter_values`); a
declared shock that no equation uses left out (as Dynare's `nostrict` does); `initval` entries for shocks ignored; and the
model read as it stands at the `estimation` statement (this file's later calibration checks reset two shock sizes).
`import_dynare_source` imports a model from a published archive described by a small YAML file; the archive is downloaded into
the data cache and nothing of it enters the repository. Data manifests gain an unnamed date column and `demean: true`
(Dynare's `prefilter=1`). Checked (`tests/test_stage4_gibbs_hambur_nodari2018.py`, slow tier): 150 variables, as in Dynare (its
7 auxiliary variables come from our own lag handling); every fixed value equals the stored `M_.params`; at the stored posterior
mode the log-likelihood (-4909.6624259464) and the log prior density equal Dynare 6.5's to 8e-11 and 1e-10. Dynare was run with
the stored parameter values (`demos/gibbs_hambur_nodari2018/reference/README.md`): its own solve of the steady state, which is
singular in one direction, moves the log-likelihood by 1.4e-6. A small file with all these features solves as the same model
written by hand, to 1e-14 (fast tier). The stored run used 99 quarters, 1992Q2 to 2016Q4, not the paper's 1992Q1 start. Not
yet: the steady state and calibration as part of a specification (the values are taken as computed); the model written as
blocks, and the step from the base model; estimation at this size (parked item 3).

**55. The root finder's derivative is our own implicit-function rule, so that a failed steady-state search stays a status.**
Optimistix's own rule raised an error ("a linear solver received non-finite input") when a gradient passed through a search
that had run off to values that are not finite, or whose root had a Jacobian that is not finite: the real business cycle toy at
a capital share of 0.98 or more, which prior draws reach, stopped SMC and NUTS (found while building stage 3). Now
`mutatis.numerics.rootfind.find_root` treats a root or Jacobian that is not finite as a failure: the status is NOT_CONVERGED,
the value is a finite stand-in (the start, with entries that are not finite set to zero), and the derivative, dx = -J^-1
(d fn / d args) dargs, is zero there. Downstream, the log-likelihood is minus infinity and its gradient not finite, which the
samplers already treat as a rejected move. Checked: first and second derivatives of a root against the closed form; a failed
search, alone and batched, gives a zero derivative without raising; SMC now runs on the real business cycle toy.

**56. The web inspector is a top layer, `mutatis.web`, started by `python -m mutatis inspect STORE` (stage 3).** It may import
every other layer; only the command line imports it, since the `inspect` command starts the server. The family-name rule
applies to it. The server is the standard library's `ThreadingHTTPServer`, GET only (other methods are refused), listening on
localhost by default; it serves plain HTML, CSS and JavaScript with no build step, and a JSON API (`/api/runs`,
`/api/runs/<key>/{model,parameters,fit,quantities}`, `/api/compare/<a>/<b>`). Each answer is computed on first request and
kept, once even under concurrent requests; static files are served only from `static/`; NaN becomes null in the JSON; a run's
key is its folder name. Everything shown is computed in Python and tested there; the browser only draws it. Alternative: a
separate entry point for the server.

**57. Each run keeps its data, as `data.npz` (dates, names, values), checked against the run's fingerprint when loaded.** The
inspector may read only the store, and `run.json` held only a fingerprint. npz rather than CSV because it is bit-exact (a CSV
round trip turns -NaN into NaN, which breaks the fingerprint check; tested). `save_result` is split out of `run`, so any
posterior can be stored through the store's own code (the fast tests store draws from the prior; no sampler runs in them);
`load_data`, `load_posterior` and `problem_of` read a run back. Runs stored before this show their summaries and priors, with a
note saying why the rest is missing. Alternative: storing the manifest's path (fails for CSV data, breaks when files move).

**58. "Fitted" means the one-step-ahead prediction of each observed series** (`mutatis.infer.states.one_step_predictions`),
from the route's estimate of the state one period earlier, with that period's shocks and measurement errors at zero: kalman,
the filter's predicted state (`routes.kalman_run`, now shared by the likelihood, the smoother and this); read_off, the state read
exactly from the data; sample, the state the sampled shocks imply (drawn given all the data, which the page says). Shown as
the median and a 90% band over 200 posterior draws spaced evenly through the stored draws; draws with no usable solution are
dropped and counted. The band is parameter (and, when sampled, state) uncertainty only, not next period's shocks; including
those would need a particle filter whenever shocks are not normal. Alternative rejected: smoothed values, which equal the data
for any series observed exactly.

**59. Priors against posteriors, element by element.** The posterior is a 40-bin histogram of the stored draws in the
parameter's own values. A prior that acts element by element and needs no estimated parameter is drawn as its exact density on
a 121-point grid; any other proper prior (the even prior over rotations, inverse Wishart, a prior with estimated arguments) as a
histogram of 4,000 draws (`draw_prior`); flat and Jeffreys priors are not drawn, with a note. The axis is the posterior's
0.1-99.9% range plus 15% each side, widened to the prior's central 96% unless that is more than 12 times wider (then flagged),
clipped to allowed values. Up to 16 elements get a panel each, a matrix parameter laid out as its matrix; larger ones (the
SVAR's 64 lag coefficients) become a filterable table with a small chart per row. The summary numbers come from
`summary.json`, as `compare` prints them.

**60. Equations are written as LaTeX in Python from our own expression trees (`mutatis.spec.latex`), not with sympy.** Sympy
reorders terms and has no notion of time shifts or expectation terms. Variables carry the period (x(-1) is x_{t-1}, E(...) is
E_t[...]); Greek names become Greek letters; brackets appear only where the order of operations needs them (checked on 400
random expressions by evaluating the LaTeX back). KaTeX 0.16.9 is loaded from cdnjs with integrity hashes; without it the page
shows the specification's text.

**61. The block graph is laid out in Python, layered (Sugiyama style).** Series that feed the model come first, observed and
restriction-only series last; blocks are placed by longest path after loops are broken (Eades-Lin-Smyth); long edges pass
through slots of their own (tested: no edge crosses a box); four barycentre sweeps order each column. One node per
specification block (a template's instances are listed in it). The same specification always gives the same layout. Known
limits: edge labels can overlap lines in dense graphs, and a loop arc can cross a box below its source in the same column.

**62. Comparing two runs** shows the edits from `diff`; which blocks, series, restrictions, shared parameters, sets and
expectations setting differ (found by comparing the two specifications directly); the second version's graph with the changed
parts marked; parameters side by side with a 90%-interval chart per row; the reported quantities both runs have, overlaid; and
both log marginal likelihoods, with the caveat about prior widths. It warns when the data differ, and says so when `diff`
refuses because the two are not versions of one model. Reported quantities also have a tab of their own for a single run.
