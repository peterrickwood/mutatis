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
changes PLAN.md's acceptance criterion for ranges and needs confirmation.** Evidence (`experiments/read2022_differences.py`):
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
