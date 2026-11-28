# Reference results

At the posterior mode the authors stored (`Domestic Sector/multisector_mode.mat` in the supplementary archive: 58 values), with
every other parameter at the value stored in `Domestic Sector/multisector_results.mat` (`M_.params`):

- `log_prior_density`: Dynare 6.5's `priordens` at the mode.
- `laplace_approximation`: Dynare 6.5's Laplace approximation of the log marginal density, using the Hessian stored in the mode
  file; `laplace_approximation_stored` is the value the authors' run stored (`oo_.MarginalDensity.LaplaceApproximation`, Dynare
  4.4.0). They agree to 1e-10.
- `log_likelihood`: the Laplace approximation minus its other terms, n/2 log(2 pi) + 1/2 log det(V) with V the inverse of the
  stored Hessian (n = 58), minus the log prior density.

How Dynare 6.5 was run (in Octave 10.3, on a scratch copy of the authors' files; nothing of them is in this repository):
`estimate_mod = 1` in `multisector.mod`; the preprocessor option `nostrict` (the shock `eps_z_x` is declared but unused); the
calibration-check include left out; the data sheet written to a `.m` data file; `estimation(first_obs=2, nobs=99, prefilter=1,
prior_trunc=0, mode_compute=0, mode_file=multisector_mode, mh_replic=0)`, which is the stored run's sample (1992Q2 to 2016Q4,
each series demeaned over it); and, just before the `estimation` statement, `M_.params` replaced by the stored values. That
last step matters: the code computes 31 parameter values from its steady state in MATLAB, at the calibrated values, and Dynare
6.5's solve of that steady state differs from the stored one by about 5e-8, which moves the log likelihood by 1.4e-6 (to
-4909.6624245939). The steady-state system is singular in one direction (only the products of A_H and the labour weights are
pinned down), so its solution depends on the solver.
