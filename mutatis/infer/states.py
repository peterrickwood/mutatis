"""Estimates of the hidden state at every date, for given values of the unknowns, by whichever route the problem uses."""

import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.infer import routes
from mutatis.infer.problem import PosteriorProblem, _initial_from_standard
from mutatis.numerics.kalman import kalman_filter, rts_smoother


def hidden_states(problem: PosteriorProblem, z):
    """(periods, states): the smoothed mean (Kalman), the exact states (read_off), or the states the sampled shocks imply (sample)."""
    values, _ = problem.layout.constrain(z)
    params = {k: v for k, v in values.items() if k in problem.system.parameters}
    solved = problem.solve(params)
    if problem.route == "read_off":
        return routes.recover_shocks(solved, problem.observed, problem.outside, problem.start_state)[1]
    if problem.route == "sample":
        start = _initial_from_standard(solved, values["initial"]) if problem.start == "model" else problem.start_state
        states, state = [], start
        for t in range(problem.observed.shape[0]):
            state = solved.step(state, solved.shocks_from_standard(values["shocks"][t]), problem.outside[t])
            states.append(state)
        return jnp.stack(states)
    form, noise = solved.linear_form(), solved.gaussian_noise()
    n = form.P.shape[0]
    m0, P0 = solved.initial_gaussian() if problem.start == "model" else (problem.start_state, jnp.zeros((n, n)))
    filtered = kalman_filter(problem.observed, m0, P0, form.P, form.Q @ noise.shock_cov @ form.Q.T, form.Z, jnp.diag(noise.error_variance),
                             form.kappa + problem.outside @ form.R.T, jnp.broadcast_to(form.d, problem.observed.shape))
    return rts_smoother(filtered, form.P).mean
