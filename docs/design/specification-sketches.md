# Specification sketches (the paper exercise)

*Stage 0, step 2 of [PLAN.md](../../PLAN.md). For review before any parser is written.*

This document writes out, by hand, the specifications mutatis would need for seven cases: four small toy models, the SVAR of Read (2022), an excerpt of MARTIN, and the housing expansion of the RBA's multi-sector DSGE. The point is to find out, before writing code, whether one specification format can express all of them, and what that format has to contain.

Everything here is a **proposal**. The syntax is illustrative. What matters is which concepts each case needs. Where a case forced something new into the format, it is marked **Added**. The questions that need your decision are collected at the end.

## The format in brief

A specification has these parts:

```yaml
model: <name>
time: quarterly
sets: {}          # named lists that variables, parameters and blocks can run over
calendar: {}      # variables that depend only on the date, such as a dummy for "before 2008Q2"
expectations: rational   # how every marked expectation term is computed
blocks: {}        # the parts of the model
data: {}          # links between data series and the model
restrictions: []  # requirements on quantities the model implies
```

A **block** has:

```yaml
<block>:
  inputs:  []     # variables it takes from other blocks, or from outside the model
  outputs: []     # variables its equations determine; there must be exactly one equation per output
  parameters: {}  # each one fixed, given a prior, or given no prior (see below)
  definitions: {} # named expressions of parameters
  shocks: {}      # each one with its distribution
  initial: {}     # starting values for variables whose level never settles
  equations: []   # or `code: <name>` for a block written as a registered JAX function
```

A **parameter** is one of:

```yaml
beta:  {fixed: 0.99}                           # an assumption: all weight on one value
sigma: {prior: gamma(mean=1.5, sd=0.4)}        # a belief the data can update
Q:     {prior: none}                           # restricted, with no prior behind it: results are reported as ranges
```

and may also have a `shape` (a list of sets) and a `domain` (`positive`, `unit_interval`, `positive_definite`, `orthogonal`).

**Equations** use `x(-1)` for last period, `x(+1)` for next period, `E(...)` for an expectation term, `d(x)` for `x - x(-1)`, `@` for matrix products, `sum(l in lags: ...)` for sums over a set, and `x[j]` for one element of a variable that runs over a set.

A **data link** does one of three things:

```yaml
<series>:
  observes: <expression in model variables>   # the model must explain this series
  error: none | <distribution>                # none means observed exactly
<series>:
  feeds: <variable>                           # supplies an outside variable; the model conditions on it
<series>: {}                                  # used only by restrictions
```

An **edit** is one entry from a fixed list. Each takes a specification and returns a new one:

| Edit | What it changes |
|---|---|
| `set_prior` | a parameter's prior, including `fixed` and `none` |
| `set_shock` | a shock's distribution |
| `set_expectations` | the expectation scheme for the whole model |
| `add_restriction`, `remove_restriction` | a restriction |
| `add_data`, `remove_data` | a data link |
| `add_block`, `remove_block` | a block |
| `replace_block` | a block, by another with the same inputs and outputs |
| `extend_set`, `shrink_set` | the elements of a set **(Added: housing)** |
| `bind`, `untie` | whether several block instances share one parameter **(Added: housing)** |

## 1. Hidden level (toy)

One variable that drifts, seen through noisy data.

```yaml
model: hidden-level
time: quarterly

blocks:
  level:
    outputs: [mu]
    parameters:
      sigma_eta: {prior: half_normal(scale=1)}
    shocks:
      eta: normal(0, sigma_eta)
    initial:
      mu: normal(0, 10)
    equations:
      - "mu = mu(-1) + eta"

data:
  y:
    observes: mu
    error: normal(0, sigma_eps)
    parameters:
      sigma_eps: {prior: half_normal(scale=1)}
```

- **Added: `initial`.** The level never settles, so where it starts is part of the model and needs a prior.
- **Added: parameters on data links.** The size of the measurement error belongs to the data link, not to a block.

## 2. Two-variable VAR (toy)

```yaml
model: two-variable
time: quarterly
sets:
  series: [a, b]

blocks:
  dynamics:
    outputs: ["y[series]"]
    parameters:
      c:     {shape: [series], prior: normal(0, 10)}
      B:     {shape: [series, series], prior: normal(0, 1)}
      Sigma: {shape: [series, series], domain: positive_definite, prior: inverse_wishart(df=4, scale=identity)}
    shocks:
      "u[series]": multivariate_normal(0, Sigma)
    equations:
      - "y = c + B @ y(-1) + u"

data:
  a: {observes: "y[a]", error: none}
  b: {observes: "y[b]", error: none}
```

- **Added: sets, vector variables, matrix parameters, domains.** A VAR cannot be written without them, so they are in from the start, not "further out".
- One vector equation counts as two equations, one per element of `series`. The count matches the two outputs.
- Both series are observed exactly, so the shocks are read off the data.

## 3. Forward-looking (toy)

The price depends on what people expect it to be next quarter. With rational expectations the answer is known: `p = x / (1 - a * rho)`.

```yaml
model: forward-looking
time: quarterly
expectations: rational

blocks:
  demand:
    outputs: [x]
    parameters:
      rho:     {prior: beta(mean=0.8, sd=0.1)}
      sigma_x: {prior: half_normal(scale=1)}
    shocks:
      e: normal(0, sigma_x)
    equations:
      - "x = rho * x(-1) + e"

  price:
    inputs: [x]
    outputs: [p]
    parameters:
      a: {prior: beta(mean=0.5, sd=0.1)}
    equations:
      - "p = x + a * E(p(+1))"

data:
  x_obs: {observes: x, error: normal(0, 0.1)}
  p_obs: {observes: p, error: normal(0, 0.1)}
```

Switching to rule-of-thumb expectations is one edit. The `E(p(+1))` term becomes `p(-1)`, and the model then needs no solving:

```yaml
- set_expectations: {scheme: naive}
```

- Fits without additions.

## 4. Error correction (toy)

One MARTIN-style equation: spending moves back towards a long-run relation with income. Income comes from data.

```yaml
model: error-correction
time: quarterly

blocks:
  spending:
    inputs: [income]
    outputs: [spend]
    parameters:
      alpha0: {prior: normal(0, 1)}
      gamma:  {prior: normal(0, 1)}      # how fast spending returns to the long-run relation
      beta1:  {prior: normal(1, 0.5)}    # the long-run relation between spending and income
      alpha1: {prior: normal(0, 1)}
      sigma:  {prior: half_normal(scale=1)}
    shocks:
      e: normal(0, sigma)
    equations:
      - "d(spend) = alpha0 + gamma * (spend(-1) - beta1 * income(-1)) + alpha1 * d(income) + e"

data:
  spend:  {observes: spend, error: none}
  income: {feeds: income}
```

- **Added: `feeds`.** Income is an outside variable: the model conditions on it and does not explain it.
- The equation multiplies two parameters (`gamma * beta1`). So it is not linear in its parameters, and the exact regression formula does **not** apply; the general route does. Fixing `beta1` makes it linear and the shortcut applies. This toy therefore tests that shortcuts switch on and off correctly, which is what the plan requires.

## 5. Read's sign-restricted SVAR

The model as described on the paper's public pages: four variables, four lags, three outside variables entering now and with four lags, 1984 to 2019. *The details of the data and of the six restriction sets come from an automated summary of the paper and have not been checked against the paper itself or its data file.*

### Base specification

```yaml
model: monetary-policy-svar
time: quarterly
sets:
  series:       [cash_rate, gdp, cpi, twi]   # cash rate, real GDP, trimmed-mean CPI, trade-weighted exchange rate
  outside:      [tot, us_gdp, us_rate]       # terms of trade, US real GDP, US federal funds rate
  shocks:       [policy, other1, other2, other3]
  lags:         1..4
  outside_lags: 0..4

blocks:
  economy:
    inputs:  ["x[outside]"]
    outputs: ["y[series]"]
    parameters:
      c:     {shape: [series], prior: flat}
      B:     {shape: [lags, series, series], prior: flat}
      D:     {shape: [outside_lags, series, outside], prior: flat}
      Sigma: {shape: [series, series], domain: positive_definite, prior: jeffreys}
      Q:     {shape: [series, shocks], domain: orthogonal, prior: uniform}   # how forecast errors split into shocks
    shocks:
      "eps[shocks]": normal(0, 1)
    equations:
      - "y = c + sum(l in lags: B[l] @ y(-l)) + sum(l in outside_lags: D[l] @ x(-l)) + chol(Sigma) @ Q @ eps"

data:
  cash_rate: {observes: "y[cash_rate]", error: none}
  gdp:       {observes: "y[gdp]", error: none}
  cpi:       {observes: "y[cpi]", error: none}
  twi:       {observes: "y[twi]", error: none}
  tot:       {feeds: "x[tot]"}
  us_gdp:    {feeds: "x[us_gdp]"}
  us_rate:   {feeds: "x[us_rate]"}
  proxy:     {}          # an outside measure of policy surprises, used only by restrictions
```

Where each series comes from, and how it is transformed (logs, for example), goes in the data manifest, not here.

### The restriction sets, as edits

Restriction set 1:

```yaml
- add_restriction: {name: rate_rises,   quantity: "response(y[cash_rate], eps[policy], horizons=0..3)", require: ">= 0"}
- add_restriction: {name: prices_fall,  quantity: "response(y[cpi], eps[policy], horizons=0..3)",       require: "<= 0"}
```

Set 2 is set 1 plus:

```yaml
- add_restriction: {name: currency_rises, quantity: "response(y[twi], eps[policy], horizons=?)", require: ">= 0"}
```

Set 3 restricts the interest-rate rule. That is the equation which defines the policy shock, read off the inverse of the impact matrix `chol(Sigma) @ Q`:

```yaml
- add_restriction: {name: rule_output,       quantity: "shock_equation(eps[policy], y[gdp])",                  require: "<= 0"}
- add_restriction: {name: rule_prices,       quantity: "shock_equation(eps[policy], y[cpi])",                  require: "<= 0"}
- add_restriction: {name: rate_rises_impact, quantity: "response(y[cash_rate], eps[policy], horizons=0)",       require: ">= 0"}
```

Set 4 is set 3 plus a restriction on the rule's coefficient on the exchange rate. Set 5 is sets 2 and 4 together. Set 6 is set 5 plus two restrictions linking the policy shock to the proxy series:

```yaml
- add_restriction:
    name: proxy_agrees
    quantity: "correlation(shock_estimate(eps[policy]), data(proxy))"
    require: "> 0"
- add_restriction:
    name: proxy_mostly_policy
    quantity: "abs(correlation(shock_estimate(eps[policy]), data(proxy))) - max(j in shocks except policy: abs(correlation(shock_estimate(eps[j]), data(proxy))))"
    require: ">= 0"
```

### Conventional result versus range

The base specification gives `Q` an even prior over all rotations, which produces the conventional result. The range result is one edit:

```yaml
- set_prior: {target: economy.Q, prior: none}
```

What this case added, or confirmed:

- **Added: a small expression language for quantities.** Read's set 6 needs `abs`, `max` over shocks and `correlation`. Without it, each restriction type would need its own code.
- **Added: restrictions can refer to data directly.** The proxy series is never explained by the model.
- **Added: `shock_equation`.** This is "the coefficient on a variable in the equation that defines a shock". It is defined for any model whose shocks can be read off the data, so it is not specific to VARs. Its sign depends on a normalisation rule that still has to be fixed.
- **Confirmed: one edit separates Read's two kinds of result.**
- **Confirmed: the six restriction sets are six edit lists over one base.**
- **Confirmed: `prior: flat` here relies on the exact formula** (README, "Flat needs care").

## 6. MARTIN excerpt

Four pieces from the online appendix of RDP 2019-07. Equation (8) for household consumption, and rule (128) for the cash rate, were checked against the rendered page. Identities (11), (12) and (129) were transcribed from extracted text. Lowercase names are logarithms of the uppercase ones; that is my reading of the appendix notation, not something I found stated.

```yaml
model: martin-excerpt
time: quarterly
calendar:
  before_2008q2: "date < 2008Q2"

blocks:
  consumption:                                     # equation (8)
    inputs: [hdy, hnw, RCR, hcoe, hoy, LUR, tdlla, tdllpop, tdllhpp, before_2008q2]
    outputs: [rc]
    parameters:
      alpha0: {prior: flat}
      gamma:  {prior: flat}
      beta1:  {prior: flat}
      alpha1: {prior: flat}
      alpha2: {prior: flat}
      alpha3: {prior: flat}
      alpha4: {prior: flat}
      sigma:  {prior: jeffreys}
      real_rate_weight:  {fixed: 0.5}      # printed as a number in the paper; named so it can be relaxed
      wage_income_share: {fixed: 0.15}
    shocks:
      e: normal(0, sigma)
    equations:
      - >-
        d(rc) = alpha0
        + gamma * (rc(-1) - beta1 * hdy(-1) - (1 - beta1) * hnw(-2) + real_rate_weight * RCR(-1) / 100)
        + wage_income_share * d(hcoe)
        + alpha1 * d(hoy(-2)) + alpha2 * d(hnw(-1)) + alpha3 * d(LUR(-2)) / 100
        + alpha4 * before_2008q2
        + (1 - wage_income_share - alpha1 - alpha2) * (tdlla(-1) + tdllpop(-1) + tdllhpp(-1))
        + e

  disposable_income:                               # identities (11) and (12), and the log that (8) uses
    inputs: [NHCOE, NHOY, PC]
    outputs: [NHDY, HDY, hdy]
    equations:
      - "NHDY = NHCOE + NHOY"
      - "HDY = NHDY / PC"
      - "hdy = log(HDY)"

  real_cash_rate:                                  # identity (129)
    inputs: [NCR, PTM]
    outputs: [RCR]
    equations:
      - "RCR = 100 * (1 + NCR / 100) / (PTM / PTM(-4)) - 100"

  cash_rate:                                       # rule (128); every coefficient is printed as a number
    inputs: [RSTAR, PTM, LURGAP, LUR]
    outputs: [NCR]
    parameters:
      smoothing:        {fixed: 0.7}
      level_weight:     {fixed: 0.3}     # equals 1 - smoothing, but printed separately
      inflation_weight: {fixed: 2}
      gap_weight:       {fixed: 2}
      target:           {fixed: 2.5}     # the inflation target; value assumed, not checked
      sigma:            {prior: half_normal(scale=1)}
    shocks:
      e: normal(0, sigma)
    equations:
      - >-
        NCR = smoothing * NCR(-1)
        + level_weight * (RSTAR + inflation_weight * (PTM / PTM(-1) * 100 - 100) - target - gap_weight * LURGAP)
        - (LUR - LUR(-2))
        + e
```

In this excerpt, the inputs that other parts of MARTIN would supply are outside variables fed from data. Estimating one equation alone is *isolate*: cut the block out and feed its inputs from data. Running the excerpt is the four blocks together, stepping forward one quarter at a time, with the identities solved within each quarter.

- **Added: `calendar`.** Equation (8) has a dummy for dates before 2008Q2.
- **Added: name every number that is a modelling choice.** The paper prints 0.5, 0.15, 0.7 and so on as numbers. As bare numbers they could not be relaxed by an edit. Named as fixed parameters, they can. Pure constants such as 100 stay as numbers.
- **Found: tied coefficients.** `(1 - wage_income_share - alpha1 - alpha2)` ties one coefficient to three others. Written as above, untying it means replacing the equation.
- **Found: `0.3` equals `1 - 0.7`** but is printed separately. Whether to tie the two is a modelling decision the person transcribing has to make, and it changes what a later edit to `smoothing` does.
- **Found: the exact regression formula does not apply to (8)** for the same reason as toy 4 (`gamma * beta1`). The published coefficients are least-squares estimates, which equal the posterior peak under flat priors. So reproducing them means comparing a peak, not a formula.
- **Interpretation to check:** "Δ₂LUR" is read as `LUR - LUR(-2)`.

## 7. Housing expansion

*Based on the public code for RDP 2018-04 ("DSGE Reno"). The RBA's supplementary page for the base model, RDP 2015-07, returned "404 Page Not Found" when I fetched it, so I have not seen the base model's code.*

The Reno code is a log-linearised Dynare model organised around lists of sectors:

- production sectors `[n, m, z, h]`: non-traded, manufacturing, commodities, housing;
- imperfectly competitive sectors `[n, m, h]`;
- consumed sectors `[n, m, f, h]`, where `f` is imports.

Most equations are written once and repeated for each sector in a list. For example, the hours aggregate sums over all production sectors. Housing appears partly as one more element in these lists, and partly through a few equations of its own:

- **investment demand:** a different equation for housing, because housing enters households' utility;
- **investment supply:** its own adjustment-cost parameter (`phik_h`, where the other sectors share `phik_n`) and its own investment shock;
- **demand schedule:** a housing demand equation;
- **preference shock:** a housing preference shock.

So the expansion, written as edits, looks like this:

```yaml
- extend_set: {set: production_sectors,              add: [h]}
- extend_set: {set: imperfectly_competitive_sectors, add: [h]}
- extend_set: {set: consumed_sectors,                add: [h]}
- replace_block:
    target: "investment_demand[h]"
    with: investment_demand_with_housing_in_utility      # same inputs and outputs
- untie: {parameter: phik, instance: "investment_supply[h]", new_name: phik_h, prior: "..."}
- add_block: housing_preference                           # the preference shock and its process
- add_data: {series: dwelling_investment, observes: "...", error: "..."}
- add_data: {series: housing_prices,      observes: "...", error: "..."}
```

That only works if blocks can be **templates** run over a set:

```yaml
blocks:
  "investment_supply[j in production_sectors]":
    inputs: [lambda, mu, "inve[j]", upsilon]
    outputs: ["lambda_k[j]"]
    parameters:
      phik: {prior: "..."}            # one parameter shared by every instance, until untied
    equations:
      - "lambda_k[j] = phik * mubar^2 * ((1 + beta) * inve[j] - inve[j](-1) - beta * inve[j](+1) - beta * mu(+1) + mu) + lambda - upsilon"
```

What this case added:

- **Added: block templates run over a set.** Each element gets one instance, and a single instance (`investment_demand[h]`) can be replaced on its own, through the same inputs and outputs as any other block.
- **Added: `extend_set`.** Adding an element to a set adds an instance of every block that runs over it, and a term to every sum over it. That is most of what "add a housing sector" means in this code.
- **Added: `bind` and `untie`.** Reno makes non-traded, manufacturing and commodity investment share one adjustment-cost parameter, while housing gets its own. Whether parts of a model share a parameter is itself an assumption, so it should be an edit.
- **Found: steady-state values computed outside Dynare.** The equations use steady-state values such as `k_h_ss`, which MATLAB code outside Dynare computes. An importer can bring these in as fixed numbers, but then changing a deep parameter would leave them stale. This needs a decision before stage 4.
- **Found: Reno estimates its foreign block first,** then fixes those parameters at their posterior means while estimating the rest. That is *isolate* followed by an edit that fixes parameters at estimated values. Those are general capabilities already in the plan, not special code.
- **Not confirmed:** that the published base model is Reno with housing removed. Without the base code, that cannot be checked.

## What the exercise found

One format covers all seven cases, with the additions below. Nothing needed a model-family-specific construct.

**Added to the format:**

1. Sets, vector variables, matrix parameters and domains (from the start, because a VAR needs them).
2. `initial`: priors on starting values of variables that never settle.
3. Data links in three roles: `observes` (with or without error), `feeds` an outside variable, or used only by restrictions; with their own parameters.
4. `calendar` variables.
5. Every number that is a modelling choice is a named parameter, fixed or not.
6. `definitions`: named expressions of parameters, as Dynare's model-local variables. Reno uses them; none of the sketches above needed one.
7. A small expression language for quantities, used by restrictions and priors on behaviour.
8. Block templates over sets; `extend_set`; `bind` and `untie`.

**Not tested by this exercise:**

- A nonlinear DSGE written from scratch, which mutatis would linearise. That is the textbook DSGE of stage 2.
- Dynare import.
- The microdata block.

## Questions for you

1. **One equation per output, per block.** Each block names the variables its equations determine, and must have exactly that many equations. That makes counting checkable block by block. The cost is that a market-clearing condition has to be assigned to some block, which then "determines" a price or quantity. Agree?
2. **Naming every number.** Should the convention be that every number that is a modelling choice is a named parameter, as in the MARTIN excerpt, while pure constants such as 100 stay as numbers? The alternative is to leave published numbers bare and accept that they cannot be relaxed by an edit.
3. **Small changes inside an equation.** Untying MARTIN's `(1 - 0.15 - α1 - α2)` currently means replacing the whole block. Should there also be a `replace_equation` edit for changes like this? I lean yes, since each MARTIN equation is its own block anyway.
4. **Templates, `extend_set`, `bind` and `untie`.** They make the housing expansion a short list of edits, but they are the most complex part of the format. Keep them in the design now, even though stage 4 is the first to need them?
5. **The base 2015 model.** Its code does not appear to be public from the RBA. PLAN.md says demonstration 3 has "a published answer on both sides". That may not hold. This does not block stage 0, but before stage 4 we need either the base code from elsewhere, or a different second side.

## Sources

- Read, M. (2022), RBA Research Discussion Paper 2022-09, public pages and read-me.
- Ballantyne and others (2019), RBA Research Discussion Paper 2019-07, online appendix, equations (8), (11), (12), (128), (129).
- Gibbs, Hambur & Nodari (2018), RBA Research Discussion Paper 2018-04, supplementary code: `Domestic Sector/model_equations_loglinear.mod`, `multisector.mod` and the read-me. The read-me says "All files are available for external release".
