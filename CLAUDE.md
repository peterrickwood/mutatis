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

Nothing is designed yet. The repo holds an empty README.md and the LICENSE. Implementation language, how a model is represented, estimation methods and the user interface are all undecided — ask rather than assume.
