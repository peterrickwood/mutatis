# Plan

> **Stub.** A proper implementation plan will replace this. What mutatis is, and why, is in [README.md](README.md); this file is for how and in what order it gets built.

## What the project's success depends on

Four things do most to decide whether mutatis gets used. The plan should be built around them.

1. **Import Dynare model files.** Most published models of this kind are distributed as Dynare files. Reading them means existing models come in without being retyped, and anyone with a Dynare model can try mutatis on it.
2. **Reproduce published results exactly.** Two or three well-known published models, matched number for number against the original code. This is how the tool earns trust.
3. **One flagship demonstration that people care about.** A real question about the Australian economy, answered under many variants of the assumptions, showing the change-and-compare workflow doing something that is tedious by hand.
4. **Economists as co-owners.** The library of blocks, the choice of demonstrations and the field's trust all need people from inside the field. The first users in view work with SVARs, so SVARs need to be supported early and well.

## Demonstrations, in order

The order is agreed. Which papers to use is not yet confirmed. Everything is built from published papers, their public replication files and public data only.

1. **An SVAR from published RBA research.** Candidate: Read (2022), "Estimating the Effects of Monetary Policy in Australia Using Sign-restricted Structural Vector Autoregressions", RBA Research Discussion Paper 2022-09. MATLAB code and data are public. It reports results two ways: under one conventional prior, and as the range of answers across every prior consistent with the restrictions. Reproducing the second needs restrictions with no prior behind them, reported as ranges (README, "When the data cannot decide").
2. **A simple textbook DSGE.** A small model whose answers are well known, to prove the path from specification through solving to estimation before anything large is attempted. Which textbook model is to be chosen.
3. **A larger DSGE, as close as public sources allow to what the RBA uses.** Candidates: Rees, Smith & Hall (2015), "A Multi-sector Model of the Australian Economy", RBA Research Discussion Paper 2015-07, and Gibbs, Hambur & Nodari (2018), "DSGE Reno: Adding a Housing Block to a Small Open Economy Model", RBA Research Discussion Paper 2018-04, which adds a housing block to it. Dynare code is public. The pair is an expansion with a published answer on both sides.
4. **A block estimated from data on individual households or firms.** One block (for example, how household spending or firm investment responds to interest rates) is estimated from microdata and then plugged into one of the earlier models behind the same interface. Synthetic or public data is enough: the point is to show what can be done, not to build the right component. Which block, which data and which model it plugs into are to be chosen. The README's caution about learned blocks applies.

## Draft staging

From the design discussion. Not agreed.

1. **Core.** The specification format. The steady-state and linear forward-looking solvers. The general estimation route and the two shortcuts. The first two demonstrations, written in the same format and estimated by the same code: the SVAR, then the textbook DSGE.
2. **Versions.** Edit, re-estimate, compare. The built-in checks. The web app in inspect-only form.
3. **Australia.** The third demonstration: the larger DSGE. A DSGE as a prior on a VAR. Priors on model behaviour. Switchable expectations.
4. **Editing in the web app.** The library of ready-made blocks.
5. **Further out.** Indexed variables for many goods and industries. Many household types. Constraints that bind only sometimes. Simulation-only models.

## To test early

- **What generality costs.** Time the general estimation route (sampling shocks together with parameters) against the Kalman-filter shortcut on the same linear model, at small and medium size. The design assumes the cost is tolerable; this has not been measured.

## Decided

- **Our own core on JAX, not built on gEconpy.** gEconpy already parses models, derives equations from agents' optimisation problems, and solves and estimates DSGE models. It is not the base because it is licensed GPL-3.0 (mutatis is Apache-2.0), is built on PyTensor and PyMC, centres its estimation on the Kalman filter, and has no SVAR support. It is to be used as a reference to test against, alongside Dynare. "Our own core" means a thin layer over existing JAX libraries, not everything from scratch.

## Implementation choices still to make

- Whether to build estimation on an existing library (NumPyro and BlackJAX are candidates) or write it.
- How equations are parsed.
- Where versions and estimation results are stored.
- How Australian data series are fetched, versioned and documented.
- Which browser libraries draw the graph and the mathematics.
