# mutatis

An open-source (Apache-2.0) framework for economic modelling in which the model can be expanded or restricted as the user wishes. It is meant to be a laboratory for rapid experimentation with different modelling approaches: something like a "model operating system", where the user collapses or expands parts of the model, adds constraints or data, re-estimates, and sees what difference the change made.

## The motivating example

Model the Australian economy. Start from a DSGE model. Then click on any part of it and propose:

- **an expansion** — relax an assumption, e.g. replace the single household with multiple household types, or replace rational expectations with bounded rationality;
- **a constraint** — add data or a restriction, e.g. data showing that a particular variable stayed within a certain range.

Re-estimate after each change and compare the results. Parts of the model can also be collapsed back to simpler forms.

## The name

From *mutatis mutandis*, "the necessary changes having been made": change one assumption, and the rest of the model is re-solved and re-estimated to match. The contrast is *ceteris paribus*, which holds everything else fixed.

## Status

Under construction: stages 0 to 3 are built (foundations, Read's SVAR, a textbook DSGE, the read-only web inspector); stage 4 (the larger DSGE and its housing expansion) has started and waits on decisions listed in PLAN.md. The design lives in these documents — read them before proposing or building anything:

- [README.md](README.md) — what mutatis is: the concepts and the framework choices. No planning or staging goes in it.
- [PLAN.md](PLAN.md) — the implementation plan and staging, with the status of each stage.
- [docs/design/decisions.md](docs/design/decisions.md) — decisions taken during implementation, each with its reason and alternative.
- [docs/design/parked.md](docs/design/parked.md) — questions set aside in discussion, to come back to.

Anything those documents mark as *proposed* or list as an open question is undecided — ask rather than assume.

## How to work on this project

- **The framework is the product.** The demonstrations in PLAN.md are milestones that stress-test generality. They are not deliverables to be built one by one.
- **No bespoke code per demonstration.** A demonstration is a specification, its data, and a comparison against published results. If one seems to need its own Python, a general building block is missing: stop and discuss.
- **Get the building blocks right.** The central design question is always which primitives exist and how they combine. Prefer small parts behind narrow contracts, so that assumptions and methods can be swapped.
- **Prefer the general mechanism.** Do not hard-wire a specific method where a general one will do, even at some cost in speed (README, core idea 4).
- **Test every method in isolation.** Each numerical method gets unit tests that need no whole model. Whole-model tests are few, slow, and not run by default.

## Development

- The environment is `.venv/` in the repo (CPU JAX, 64-bit floats). Install with `.venv/bin/pip install -e ".[dev]"`.
- Test tiers: plain `.venv/bin/python -m pytest` runs only the fast tier; `-m medium` and `-m slow` run the others. Run only the tests for the area you touched.
- `tests/architecture/` holds the automatic checks: import rules between layers, no model-family names in the core, no code in `demos/`.
