"""The three ways of dealing with the unknown shocks when computing how likely the data are.

Each takes a solved model (through contract 1 only) and arranged data:

- `kalman`: average the shocks out exactly. Needs a linear model with normal noise (the `linear_form` and `gaussian_noise`
  abilities).
- `read_off`: every series is observed exactly and there are as many shocks as series, so each period's shocks follow from
  that period's data. Works for any shock distribution. The change of variables from shocks to data adds -log|det(Z Q)| per period.
- `sample`: the shocks (and, if needed, the starting state) are unknowns sampled with the parameters; the likelihood is then
  just the measurement errors. Works for any distributions, but every series must have measurement error.

All three return minus infinity when the parameter values give no usable model.
"""

import jax
import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.kalman import kalman_filter
from mutatis.numerics.status import Status
from mutatis.solve.contract import SolvedModel


def _guard(solved: SolvedModel, value):
    ok = (solved.status == Status.OK) & jnp.isfinite(value)
    return jnp.where(ok, value, -jnp.inf)


def kalman(solved: SolvedModel, observed, outside, start_state=None):
    """`start_state`: the state read from the data (known exactly), or None to use the model's starting distribution."""
    form, noise = solved.linear_form(), solved.gaussian_noise()
    n = form.P.shape[0]
    if start_state is None:
        m0, P0 = solved.initial_gaussian()
    else:
        m0, P0 = start_state, jnp.zeros((n, n))
    c = form.kappa + outside @ form.R.T
    d = jnp.broadcast_to(form.d, observed.shape)
    W = form.Q @ noise.shock_cov @ form.Q.T
    H = jnp.diag(noise.error_variance)
    result = kalman_filter(observed, m0, P0, form.P, W, form.Z, H, c, d)
    return _guard(solved, result.log_likelihood)


def read_off(solved: SolvedModel, observed, outside, start_state):
    form = solved.linear_form()
    ZQ = form.Z @ form.Q
    lu = jax.scipy.linalg.lu_factor(ZQ)
    log_det = jnp.sum(jnp.log(jnp.abs(jnp.diagonal(lu[0]))))

    def period(state, inputs):
        y, z = inputs
        predicted = form.kappa + form.P @ state + form.R @ z
        shocks = jax.scipy.linalg.lu_solve(lu, y - form.d - form.Z @ predicted)
        return predicted + form.Q @ shocks, solved.shock_log_prob(shocks)

    _, log_probs = jax.lax.scan(period, start_state, (observed, outside))
    return _guard(solved, jnp.sum(log_probs) - observed.shape[0] * log_det)


def sample(solved: SolvedModel, observed, outside, shocks, start_state):
    def period(state, inputs):
        y, z, e = inputs
        following = solved.step(state, e, z)
        return following, solved.error_log_prob(y, solved.observe(following))

    _, log_probs = jax.lax.scan(period, start_state, (observed, outside, shocks))
    return _guard(solved, jnp.sum(log_probs))
