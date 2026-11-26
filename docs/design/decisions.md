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
