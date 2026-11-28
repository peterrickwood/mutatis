# Plan

How mutatis gets built, and in what order. What mutatis is, and why, is in [README.md](README.md).

**The framework is the product.** The aim is a set of small building blocks that combine to express, solve, estimate and compare many kinds of economic model. The demonstrations below are **milestones that stress-test that framework**, not things to build one at a time. If each demonstration needed its own code there would be no framework, only a collection of models, which people already have.

Two things follow, and they shape every stage:

- **Modularity comes first.** The question to get right is: what are the building blocks, and how do they combine? Small parts behind narrow contracts are what make an assumption or a method swappable.
- **Every method is tested alone.** Numerical methods especially get unit tests that need no whole model. Whole-model tests are few and slow.

## What the project's success depends on

Four things do most to decide whether mutatis gets used. The plan should be built around them.

1. **Import Dynare model files.** Most published models of this kind are distributed as Dynare files. Reading them means existing models come in without being retyped, and anyone with a Dynare model can try mutatis on it.
2. **Reproduce published results exactly.** Two or three well-known published models, matched number for number against the original code. This is how the tool earns trust.
3. **One flagship demonstration that people care about.** A real question about the Australian economy, answered under many variants of the assumptions, showing the change-and-compare workflow doing something that is tedious by hand. The chosen question is below, under [The flagship demonstration](#the-flagship-demonstration).
4. **Economists as co-owners.** The library of blocks, the choice of demonstrations and the field's trust all need people from inside the field. The first users in view work with SVARs, so SVARs need to be supported early and well.

## Demonstrations, in order

The order is agreed. Which papers to use is not yet confirmed. Everything is built from published papers, their public replication files and public data only.

1. **An SVAR from published RBA research.** Candidate: Read (2022), "Estimating the Effects of Monetary Policy in Australia Using Sign-restricted Structural Vector Autoregressions", RBA Research Discussion Paper 2022-09. MATLAB code and data are public. It reports results two ways: under one conventional prior, and as the range of answers across every prior consistent with the restrictions. Reproducing the second needs restrictions with no prior behind them, reported as ranges (README, "When the data cannot decide").
2. **A simple textbook DSGE.** A small model whose answers are well known, to prove the path from specification through solving to estimation before anything large is attempted. Which textbook model is to be chosen.
3. **A larger DSGE, as close as public sources allow to what the RBA uses.** Candidates: Rees, Smith & Hall (2015), "A Multi-sector Model of the Australian Economy", RBA Research Discussion Paper 2015-07, and Gibbs, Hambur & Nodari (2018), "DSGE Reno: Adding a Housing Block to a Small Open Economy Model", RBA Research Discussion Paper 2018-04, which adds a housing block to it. The housing model's Dynare code and data are public; its read-me says "All files are available for external release". The base model's code is not public: the RBA's supplementary page for RDP 2015-07 returns "404 Page Not Found", and the University of Sydney model hub offers only the housing version (checked 2026-11-26). But the base paper prints the whole log-linearised model (Appendix A, equations A1 to A61), its calibration (Table 1), its estimation results (Tables 3 and 4) and its data definitions (Appendix B). So the base side is rebuilt from the paper and matched approximately, because the data have been revised since; the housing side can be matched exactly. The RBA now calls its DSGE model DINGO; no public paper documenting it under that name was found, but a 2026 RBA paper (RDP 2026-02) still runs its DSGE exercises on the RDP 2018-04 model.
4. **A block estimated from data on individual households or firms.** One block (for example, how household spending or firm investment responds to interest rates) is estimated from microdata and then plugged into one of the earlier models behind the same interface. Synthetic or public data is enough: the point is to show what can be done, not to build the right component. Which block, which data and which model it plugs into are to be chosen. The README's caution about learned blocks applies.
5. **MARTIN, the RBA's macroeconometric model.** Ballantyne and others (2019), "MARTIN Has Its Place: A Macroeconometric Model of the Australian Economy", RBA Research Discussion Paper 2019-07. This is a third family of model: a large system of equations, mostly looking backward, with the estimated ones fitted one at a time by least squares. Start with a handful of core equations and expand from there. What the public material allows (checked 2026-11-26):
   - The online appendix prints every equation, around 140 of them. About 50 carry estimated coefficients, each with standard errors and its sample period (the paper itself counts "over 30" behavioural equations). So the model can be written down with the published values without re-estimating anything.
   - The public files are an EViews workfile holding the model and data, and EViews programs that run five scenarios, the first being a rise of one percentage point in the cash rate held for a year. They are released "for academic and personal use only". They can be used to check against, but not copied into this repository.
   - About twenty data series are proprietary and are replaced by simulated data in the public version: housing prices, bond yields, oil prices and most of the world variables. Equations that use them cannot be re-estimated from public data.
   - The model is nonlinear, so its responses depend on where the economy starts. The read-me says the public version will not exactly match the figures in the paper. The data behind the paper's figures are public.
   - Two things mutatis would need for it: estimating one block at a time while the rest is held fixed, and solving a path with chosen variables held at given values.

   So the realistic target is to write the model from the published equations, re-estimate the equations that use only public data, and match the published responses approximately. Exact reproduction of the whole model is not possible from public sources.

## The flagship demonstration

**What does a rise in the cash rate do to output and inflation?** The question is answered by three families of model inside one framework: the SVAR (demonstration 1), the larger DSGE (demonstration 3) and MARTIN (demonstration 5). The differences between the answers are then traced to the assumptions that produce them, by changing those assumptions one at a time.

The RBA has published on this question, which gives a benchmark: Mulqueeney, Ballantyne & Hambur (2025), "Monetary Policy Transmission through the Lens of the RBA's Models", RBA Bulletin, April 2025. It compares MARTIN with the RBA's DSGE and notes that the two give noticeably different answers.

## Decided

- **Our own core on JAX, not built on gEconpy.** gEconpy already parses models, derives equations from agents' optimisation problems, and solves and estimates DSGE models. It is not the base because it is licensed GPL-3.0 (mutatis is Apache-2.0), is built on PyTensor and PyMC, centres its estimation on the Kalman filter, and has no SVAR support. It is to be used as a reference to test against, alongside Dynare. "Our own core" means a thin layer over existing JAX libraries, not everything from scratch.

## Rules

1. **A demonstration contains no Python.** It is a folder holding a specification, a list of edits, a data manifest and stored reference results, run by one generic command.
2. **Every capability has two consumers.** Anything added because a demonstration needs it must also be exercised, in a fast test, by a toy model from a *different* family before the demonstration is accepted. A sign restriction built for the SVAR is also tested on a forward-looking toy.
3. **No model-family names in the core.** A check in the test suite fails if `svar`, `var`, `dsge` or `martin` appears in an identifier outside the block library, the import code and the demonstrations. Code chooses what to do by asking what a model *can do*, never what kind it is.
4. **Lower layers do not know about higher ones.** Enforced by an automatic import check (see Layout).
5. **Stop and discuss** whenever a demonstration seems to need a special case, or a building block's contract has to change.

## The building blocks

Two contracts sit in the middle. Everything else is on one side or the other.

```
 specification ─► equation system ─► SOLVED MODEL ─► POSTERIOR PROBLEM ─► draws ─► quantities ─► compare
      ▲  (edits)     (expectation      (contract 1)     (contract 2)                  (responses,
      │               rewrites)                                                        forecasts, ...)
      └──────────────────────────────── versions and results ◄──────────────────────────────┘
```

| Building block | What it is |
|---|---|
| **Specification** | Pure data: blocks, each with ports (the variables it takes in and gives out), parameters, shocks and equations or a named JAX function; data links; restrictions; the expectations setting. The internal data structure is the real model. YAML is the first way to write one; a Python interface can follow. No JAX in this layer |
| **Equation** | A syntax tree with time shifts and explicit "expected value of" nodes |
| **Edit** | One of a fixed list of pure functions from specification to specification: set a prior, fix or free a parameter, set a shock's distribution, set the expectations setting, add a restriction, add a data link, add, remove or replace a block, replace one equation inside a block. A block may only be replaced by one with matching ports. A version is a specification; the difference between two versions is a list of edits |
| **Expectation scheme** | A rewrite of the equation system. Rule-of-thumb replaces each "expected value" node with an expression in past values; discounted scales it; rational leaves it. What is left decides whether a solve is needed |
| **Solver** | Turns an equation system and parameter values into a solved model. Kinds: none needed, resting point (steady state), linear forward-looking, whole path. Reports failure (no stable solution, or many) as a status, never by crashing |
| **Solved model** (contract 1) | Can give the starting state, step the state forward given shocks and any outside variables, and say what would be observed. May also declare optional abilities: "I am linear with bell-curve shocks, here are my matrices"; "I can recover the shocks from observed values"; "I can solve a path with some variables held at given values" |
| **Shock handling** | Joins the two contracts. Three ways, chosen from the solved model's declared abilities: *sample* the shocks as unknowns; *average them out* exactly (Kalman filter); *read them off* the data |
| **Posterior problem** (contract 2) | The log prior and the log likelihood as two separate functions, a way to draw from the prior, and a layout. The layout names every unknown, gives its shape and allowed range, and gives its role: sampled, fixed, or restricted with no prior behind it. The two functions are kept separate because bringing the data in gradually needs them separately |
| **Sampler** | Draws from a posterior problem. Kinds: gradual introduction of the data with gradient-based moves (sequential Monte Carlo); an exact formula for the regression case; a quick approximation around a peak |
| **Quantity** | A named result computed from a solved model and parameter values: a response to a shock, a forecast, a structural coefficient, the response in a scenario. One definition, four users: reports, restrictions, priors on behaviour, comparison |
| **Restriction** | An inequality or equality on a quantity |
| **Bounds** | For unknowns with the role "restricted with no prior": the smallest and largest value of a quantity over everything the restrictions allow, for each posterior draw. Includes a check that the likelihood really does ignore that unknown, and refuses otherwise |
| **Isolate** | Cut one block out at its ports and feed its inputs from data, giving a small model that can be estimated alone. With "post the estimates back as priors", this is equation-by-equation estimation, and also the microdata block |
| **Scenario** | A statement of what is held at what values and which shocks adjust to make that so. Needed so that "a rise in the cash rate" means the same thing in all three model families |
| **Check** | A function from a specification (or results) to a list of findings: counting, unique stable solution, identification, interchangeable parts |
| **Versions and results** | A store keyed by specification, settings and random seed, plus comparison of quantities between versions |

How the awkward cases fit the same blocks, with no subsystem of their own:

- **A sign-restricted SVAR** is a block with three parameters: the lag coefficients, the size of the forecast errors, and the split of those errors into shocks (a rotation). Sign and zero restrictions are restrictions on quantities. Giving the rotation an even prior produces the conventional result; giving it the role "restricted with no prior" produces the range. Read's two sets of results differ by **one edit**.
- **MARTIN** is many blocks, each estimated through *isolate* by the exact regression formula, then run together as one nonlinear step-forward model. Its scenarios use *scenario*.
- **A DSGE** is blocks whose equations contain "expected value" nodes, the rational scheme, and the linear forward-looking solver. The housing expansion is a *replace block* edit.

## Layout

```
mutatis/
  numerics/     arrays in, arrays out: root finding, linear forward-looking solve, Kalman filter and smoother, rotations
  dist/         distributions and their transforms
  spec/         pure data: specification, blocks, ports, equation trees, edits, diff, YAML reader   (no JAX)
  compile/      specification -> equation system; expectation rewrites; isolate
  solve/        equation system -> solved model
  measure/      data links, transforms, missing values
  quantities/
  infer/        shock handling, posterior problem, samplers, bounds, checks
  versions/     results store, compare
  library/      ready-made blocks
  io/           Dynare import
demos/          outside the package: specifications, edit lists, data manifests, reference results
tests/
```

Import rules, checked automatically: `numerics` and `dist` import nothing else of ours. `spec` imports no JAX. `infer` sees a solved model only through contract 1. Nothing imports `library`, `io` or `demos`.

## Testing

**Three tiers.**

| Tier | What | When |
|---|---|---|
| Fast | Unit tests of single building blocks. The great majority of tests | Every change; only the area touched |
| Medium | Agreement between methods on small models; calibration of samplers | On request and before a milestone is accepted |
| Slow | Replication of a demonstration against stored reference results | At milestones only |

**Kinds of test, by building block.**

- **Numerical methods**: cases with a known formula (one-variable Kalman filter; the forward-looking toy whose answer is 1/(1 − a·ρ)); a brute-force alternative (the likelihood computed as one large bell curve, against the Kalman filter); automatic derivatives against finite differences.
- **Properties that must hold for any input**, tested on many random inputs: the likelihood does not change if the hidden variables are relabelled or reordered; rotating the split of shocks leaves the likelihood unchanged.
- **Shock handling**: the three ways, and brute force, give the same likelihood at fixed parameter values on small samples.
- **Samplers**: simulate data from known parameters many times and check the posterior covers the truth as often as it should; targets with a known answer, including two-peaked and ridge-shaped ones.
- **Bounds**: two- and three-variable cases where the allowed set has a formula, and a dense grid of rotations as a cross-check.
- **Specification**: read, write and read again gives the same thing; the difference between a specification and its edited version is the edit.
- **Reference results from Dynare and MATLAB**: solution matrices, likelihood values and responses at fixed parameter values, stored with a note of where they came from. These are deterministic and must match to numerical precision.

**Where exact agreement is and is not possible.** Deterministic pieces must match exactly. Results that come from random sampling cannot match number for number; they are compared within a tolerance derived from the sampling error itself, which is stated in the test. No other loosening of tolerances.

**Removing special structure must not break anything.** For each shortcut there is a test that removes the structure it relies on (for example, give one shock fat tails) and checks that the general route takes over and approaches the shortcut's answer as the structure is restored.

## Stages

Each stage lists the *general* capabilities it adds. A demonstration is accepted when its replication test passes and every new capability has its second, different-family test.

### Stage 0: foundations, before any demonstration

Four toy models go through **one** code path, so that the building blocks cannot shape themselves around SVARs:

- a hidden-level model (one hidden variable seen with error);
- a two-variable VAR;
- a one-equation forward-looking model with a known solution;
- one backward-looking error-correction equation.

Order of work:

1. Project skeleton: package, test tiers, automatic checks for the import rules and the family-name rule.
2. **A paper exercise, reviewed with you before any parser is written**: the specifications written out by hand for the four toys, a VAR with lags and a rotation, three MARTIN equations, and the housing expansion as a list of edits. This is the cheapest test of whether the specification can express what is needed.
3. Distributions and transforms.
4. Numerical methods, on arrays only: root finding, the linear forward-looking solve, the Kalman filter and smoother, simulation.
5. Specification data structure, equation trees, YAML reader, edits and diff. Sets, vector variables and matrix parameters are in from the start, because a VAR needs them. Block templates over sets, with the `extend_set`, `bind` and `untie` edits, are built here too, so the machinery the housing expansion needs is tested from the start. Numbers written directly in equations are allowed: they are found automatically, listed as unnamed fixed values, included in sensitivity checks, and can be made into named parameters with one edit (see docs/design/specification-sketches.md).
6. Compile and the expectation rewrites.
7. Solvers and contract 1.
8. Shock handling and contract 2.
9. Samplers: gradual introduction of data with gradient moves; the exact regression formula.
10. Quantities: responses, forecasts, estimates of hidden variables.
11. Results store, comparison and the single generic run command, all without a web app.

Accepted when: all four toys run through the same code with no branch on model family; an edit to each toy produces a comparable second version; and the two experiments under Risks have been run and reported.

**Status (2026-11-26): built.** Steps 1 to 11 are done, with a fifth fixture model written with templates. Both risk experiments have been run (see Risks): the forward-looking solve is fine; sampling the shocks needed NUTS chains instead of SMC. Acceptance: `tests/test_stage0_acceptance.py` (slow tier) estimates every toy through the one generic path and compares each with an edited version; see the decisions log for choices made along the way.

### Stage 1: Read's SVAR (demonstration 1)

What the paper needs (from its public pages and read-me; not yet checked against the data file): four variables (cash rate, real GDP, trimmed-mean inflation, the trade-weighted exchange rate), four lags, quarterly data from 1984 to 2019, three outside variables that feed in but are not modelled (terms of trade, US GDP, the US policy rate), and six sets of restrictions. The sets restrict responses to a shock, coefficients of the interest-rate rule, and the relation of the shock to an outside measure of policy surprises. The data are public and ship with the paper. The original MATLAB code may be used with attribution; its read-me says the richest set of restrictions took about 18 hours to run.

Adds: lags at realistic size; outside variables that enter a model without being modelled; parameters with special allowed ranges (a rotation); restrictions on quantities, including structural coefficients; the role "restricted with no prior" and the check that the likelihood ignores the unknown; drawing ignored unknowns from their prior; bounds and ranges; data links to public series.

Second-consumer tests: a sign restriction on the forward-looking toy's response; bounds on a toy with a parameter the data cannot inform; the exact regression formula on the error-correction toy; an outside variable feeding the hidden-level toy.

Accepted when: Read's results under the conventional prior and as ranges are both reproduced, and the paper's six sets of restrictions exist as six lists of edits to one base specification. Running time for the richest set is measured and reported; an 18-hour original means this may be a slow test even when correct.

**Status (2026-11-28): accepted. All six sets of restrictions reproduced, conventionally and as ranges.** `demos/read2022` holds only data: the model, a data manifest, the six restriction sets as edit lists, the edit that removes the rotation's prior, a report list, and Read's published figure data. `tests/test_stage1_read2022.py` (slow tier) passes, 12 tests in 4 hours 6 minutes on this 8-core machine (shared with other jobs for part of it):
- Conventional results: all 1,512 published statistics (6 sets, 4 responses, 21 horizons, mean and 68% interval) agree with Read's within sampling error, estimated by a bootstrap. For Restriction (6), Read's published figure comes from replication code that pairs the proxy with the VAR's values from 36 quarters earlier (decision 46): the test gives that pairing as data, and the demonstration's model pairs them as the paper describes.
- Ranges: our bounds are exact. On sampled draws an independent optimiser never finds an allowed value outside them (largest gap 1.8e-11 for sets 1 to 5, by the exact one-column method; 1.3e-11 for set 6, by the search over rotations, decisions 45 and 53, with 1,213 optimiser solutions checked). Against Read's published range statistics, 1,989 of 2,016 agree within sampling error; the others are where his bounds are not the exact ones (decisions 28 and 53; for set 6 his come from 5,000 random rotations, which his paper says are too narrow). The criterion (exact ranges rather than Read's numbers) is confirmed (decision 32).
- Running time of the richest set, (6) as ranges: about 3 hours for 1,000 draws, most of it the softened search's 16 starts for each of 168 bounds per draw (decision 53); before that search was added it took an estimated 25 minutes (about 1.5 s per draw), with bounds that sometimes fell short. Read's read-me gives about 18 hours for his code.

Added for Restriction (6), all general: the exact regression formula block by block, each block with its own sample (decision 44); restrictions that truncate the prior in every sampler (decision 47); ranges with any sampler (decision 48); the search over rotations for restrictions on several columns, with its softened form (decisions 45 and 53).

**To do later: a thorough written report on the replication of Read (2022).** What was reproduced and how it was checked, set by set; every difference from the published results and its cause (with the evidence and our confidence in it), including where his range bounds are not exact; what could not be reproduced and why; running times against the original's.

### Stage 2: a textbook DSGE (demonstration 2)

Adds: steady state at realistic size; linearisation by automatic derivatives; the linear forward-looking solve with failure statuses, in a sampler that crosses the boundary of the stable region; priors on parameters at scale; Dynare import for a simple model; expectation schemes beyond rational.

Second-consumer tests: an expectation rewrite on a backward-looking toy is a no-op; the imported Dynare model and the hand-written specification give identical results.

Accepted when: solution matrices, likelihood and responses match Dynare at fixed parameter values, and switching expectations from rational to rule-of-thumb is one edit.

**Status (2026-11-27): its tests pass; two parts of the acceptance condition are met indirectly (below).** The model is Herbst and Schorfheide's small New Keynesian model (decision 49), in `demos/herbst_schorfheide2016`: a specification, the same model in Dynare's language, a data manifest and reference results. `tests/test_stage2_herbst_schorfheide.py` (slow tier) passes: the log-likelihood and log prior density at two points equal the authors' own code's (to 1e-12); the first-order solution equals Dynare 6.5's (56 coefficients, to 1e-10); the Dynare file imported through Dynare's preprocessor gives the same likelihood and priors; switching to rule-of-thumb expectations is one edit; the posterior from the default sampler agrees with the authors' own random-walk Metropolis-Hastings run (200,000 draws, in Octave): every parameter's mean and standard deviation within 5 standard errors of the difference (the largest, 1.2 for a mean and 2.7 for a standard deviation). Met indirectly: the likelihood is checked against the authors' code, not against Dynare's own likelihood routine, which was not run; responses are not compared separately, since they follow from the solution matrices, which match Dynare's. Added for it: nonlinear models solved to first order around a steady state found from values the specification gives (decision 50), checked against a closed-form growth model and against Dynare on a real business cycle model; the Dynare import (decision 51); the inverse gamma prior for standard deviations, and data files without dates (decision 52). Second consumers: the steady state on a backward-looking growth model; the expectation rewrite left a backward-looking toy unchanged; the Dynare import on a real business cycle model. Still to do: the steady state at realistic size waits for stage 4's model.

### Stage 3: read-only web inspector

Draws any specification as a graph of blocks; shows equations, priors against posteriors, data against fit; compares two versions. It reads only the specification and the results store, with no view written for a particular model family. That makes it a generality test in its own right: all four toys and both demonstrations must display.

### Stage 4: the larger DSGE and its housing expansion (demonstration 3)

Adds: Dynare import at full size; many blocks joined through ports, at full size; missing data; priors on behaviour; caching of compiled models. Templates, `extend_set`, `bind` and `untie` already exist from stage 0.

Run first, before building: compare the base model's printed equations (RDP 2015-07, Appendix A) one by one against the housing model's code, and count how many equations *outside* housing changed. If many did, ports alone are not enough to describe an expansion, and the block concept needs rethinking before this stage starts.

Accepted when: both published models are reproduced and the step from one to the other is a list of edits.

**Status (2026-11-27): the gate has been run and the housing model imported; the rest waits on three decisions.**
- The gate (`docs/design/stage4-gate.md`): of the base paper's 45 printed model equations, 24 are unchanged in the housing
  model's code, 2 differ only by misprints in the paper, 6 gain housing terms, 11 changed for reasons that have nothing to do
  with housing, and 3 changed only in ways that do nothing at the published values. The 11 are revisions of the base that came
  in the same paper: sticky wages, price indexation, a different foreign block, policy rule, interest parity, productivity
  process and current-account ratio, and weights in the hours aggregate. The housing paper says it compared against "a
  modified version of RSH" with the first three. Read literally the gate trips, but it counts two steps as one: the housing
  step itself reaches the rest of the model only through sums over the lists of sectors, so ports and set edits look enough
  for it (moderate to high confidence). The base paper's appendix also leaves out at least six equations of its model.
- Done: the housing model imported from the authors' own Dynare code, its log-likelihood and log prior density equal to
  Dynare's at the published posterior mode (decision 54).
- To decide before going on: (1) which base model to reproduce, and so what the edit list goes between; (2) whether the steady
  state and calibration targets become part of a specification (adding housing moves dozens of computed coefficients
  throughout the model; as plain numbers, the expansion must overwrite them all); (3) how a block replacement may take inputs
  the block it replaces did not have (the housing instances need extra inputs and their own shock, which `replace_block`
  refuses).

### Stage 5: a block estimated from microdata (demonstration 4)

Adds: *isolate*; data links to data on individual households or firms; posting estimates back as priors; the registry of named JAX blocks.

Second-consumer test: *isolate* applied to a block of the textbook DSGE.

### Stage 6: MARTIN, core equations (demonstration 5)

Adds: a nonlinear step that needs a solve within each period; ordering a large set of equations; scenarios with variables held at given values; starting conditions for data that trend.

Accepted when: the chosen core equations, re-estimated through *isolate*, match the published coefficients where public data allow, and the published response to a cash-rate rise is matched approximately (exactly is not possible from public sources; see PLAN.md).

### Stage 7: the flagship

Adds: one scenario definition that applies to all three families; shared names for the same concept across specifications; attributing a difference in results to individual edits along a path from one version to another.

### Stage 8: editing in the web app

The editor issues edits from the fixed list. It adds no modelling capability.

## Risks, and the cheapest experiment for each

| Risk | Experiment | When |
|---|---|---|
| Sampling the shocks as unknowns is too slow or unstable | Time it against the Kalman filter on the hidden-level toy and a three-equation model with 200 periods | Stage 0. **Done 2026-11-26.** The density is right: integrating out the sampled shocks equals the Kalman likelihood exactly, and a NUTS chain on the 203 unknowns of the hidden-level toy matches the Kalman route's posterior (sigma_eta 0.232 against 0.233) in 10 s. But SMC fails there: sigma_eta 0.097 to 0.172 depending on the strength of its moves, and log marginal likelihoods 3 to 8 too low; on a 200-dimension normal target its log marginal likelihood is off by +19.7. So sampled shocks use NUTS chains, with no marginal likelihood reported. **Open:** making SMC reliable with many unknowns (more particles, adapting the number of moves, or bringing data in period by period) |
| The forward-looking solve cannot be differentiated or run on thousands of parameter values at once. JAX lacks the standard routine, so this is our own code and the largest single piece of numerical risk | On the toy with a known answer: derivatives against finite differences, agreement with SciPy's routine, and 10,000 prior draws that straddle the stable region | Stage 0. **Done 2026-11-26:** all 10,000 draws classified exactly as theory predicts, derivatives match the closed form, 0.25 s for the batch. Cost grows steeply with size (40 variables: 13 ms to solve, 67 ms with derivatives), because derivatives use a dense matrix of size n² by n²; larger models (stage 4) need a solver that never forms it |
| The specification cannot express what is needed | The paper exercise | Stage 0, step 2 |
| The search for bounds stops at a local best, not the true one | Many starting points against formula cases and a grid | Stage 1 |
| Ports are not enough to describe an expansion | Diff the two public Dynare files | Before stage 4 |

## Libraries

A thin core over existing permissively licensed libraries. Checked against each project's own repository on 2026-11-26; "from source" means read, not run.

| Need | Choice | Why, and what was found |
|---|---|---|
| Sampling | **BlackJAX** (Apache-2.0, active) | Has gradual introduction of data with an automatically chosen schedule, gradient-based moves inside it, and the ordinary gradient sampler with tuning. It reports the pieces of the marginal likelihood step by step; we add them up |
| Root finding | **Optimistix** with **Lineax** (Apache-2.0, maintained) | Root finding and least squares, with derivatives through the solution by default |
| Linear forward-looking solve | **Our own**, in `numerics/` | JAX has no ordered matrix decomposition of the kind the standard algorithm uses, and its plain version has no derivatives. So this is written by us: solve the matrix equation by iteration, take derivatives through the equation it satisfies, and test against SciPy's standard routine as the reference |
| Kalman filter | **Our own**, in `numerics/` | It is small and central, and must handle missing observations. dynamax (MIT) does not handle them; cuthbert (Apache-2.0) does but is at version 0.1. Both, and statsmodels, serve as references to test against |
| Equations | **sympy** (BSD-3) for the trees, derivatives and display; conversion to JAX by sympy2jax (Apache-2.0) or sympy's own JAX output | Which conversion to use is settled in stage 0 by trying both on the toys |
| Typed containers | **Equinox** (Apache-2.0), **jaxtyping** (MIT) | Maintained. Array shapes stated in signatures and checked in tests |
| Dynare import | **Dynare's own preprocessor, run as a separate program** | It can write a parsed model as JSON: variables, parameters, equations as syntax trees, and the priors from the estimation section. That avoids writing a Dynare parser. It is GPL, but we run it and read its output; we do not include it. A BSD-licensed Python parser (dyno.py) exists but showed no support for priors |

Not checked by running: that sympy's JAX output is in the released version and not only in development; which Dynare releases write the JSON under which file name.

## Not in this plan

- Time estimates. None are given; stage 0 is the largest stage by far.
- Models with many household types, constraints that bind only sometimes, and simulation-only models. They stay in the README as future scope.
- Who writes the block library beyond what the demonstrations need.

## How we know it works

- Each stage has an acceptance condition above, and a slow-tier replication test for its demonstration.
- The automatic checks (import rules, no family names, no Python in `demos/`) run with the fast tests.
- At the end of each stage: a short report of which capabilities were added, which toy is the second consumer of each, and anything that needed a special case.
