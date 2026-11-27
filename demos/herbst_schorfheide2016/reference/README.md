# Reference results

- `likelihood.csv`: the log-likelihood of the data and the log prior density at two parameter points, computed with the
  authors' own code (`dsgeliki.m` and `prior.m` from "DSGE Estimation.zip" on the book website, run in Octave 10.3). The first
  point is the posterior mode their SMC code lists in `prior_dsge.txt`.
- `dynare_solution.csv`: the first-order solution of `../model.mod` at the parameter values written in it (the first point
  above), computed with Dynare 6.5 in Octave 10.3 (`oo_.dr.ghx` and `oo_.dr.ghu`): each variable's coefficient on each state
  variable's previous value and on each shock.
- `posterior.csv`: the posterior mean and standard deviation of each parameter from the authors' own random-walk
  Metropolis-Hastings code (`MetropolisHastings.m`, starting from their stored mode and Hessian, `MH_candidate.mat`), run in
  Octave 10.3 for 200,000 draws, the first half dropped as their code does; with each parameter's effective sample size (rank
  normalised, the 100,000 draws split into 4 consecutive chains; largest R-hat 1.006).
