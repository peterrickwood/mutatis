"""Estimates of the hidden state at every date, and the model's one-step-ahead predictions of the data, for given values of the
unknowns, by whichever route the problem uses."""

import jax
import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.quantities.recover import recover_shocks
from mutatis.infer.problem import PosteriorProblem, _initial_from_standard
from mutatis.infer.routes import kalman_run
from mutatis.numerics.kalman import rts_smoother
from mutatis.numerics.status import Status


def hidden_states(problem: PosteriorProblem, z):
    """(periods, states): the smoothed mean (Kalman), the exact states (read_off), or the states the sampled shocks imply (sample)."""
    values, _ = problem.layout.constrain(z)
    params = {k: v for k, v in values.items() if k in problem.system.parameters}
    solved = problem.solve(params)
    if problem.route == "read_off":
        return recover_shocks(solved, problem.observed, problem.outside, problem.start_state)[1]
    if problem.route == "sample":
        return _sampled_path(problem, values, solved)[1]
    filtered = kalman_run(solved, problem.observed, problem.outside, problem.start_state)
    return rts_smoother(filtered, solved.linear_form().P).mean


def _sampled_path(problem: PosteriorProblem, values: dict, solved):
    """The starting state and the states (periods, states) that the sampled shocks imply."""
    start = _initial_from_standard(solved, values["initial"]) if problem.start == "model" else problem.start_state
    states, state = [], start
    for t in range(problem.observed.shape[0]):
        state = solved.step(state, solved.shocks_from_standard(values["shocks"][t]), problem.outside[t])
        states.append(state)
    return start, jnp.stack(states)


def one_step_predictions(problem: PosteriorProblem, z):
    """(periods, observed series): what the model predicts each observed series to be, from the state one period earlier, with
    that period's shocks and measurement errors at zero. The earlier state is the route's estimate of it:

    - kalman: the filter's estimate given the data up to that period, so this is the usual one-step-ahead prediction;
    - read_off: the state read exactly from the data up to that period (the same thing, known exactly);
    - sample: the state the sampled shocks imply, which were drawn given all the data.

    NaN everywhere when the parameter values give no usable model.
    """
    values, _ = problem.layout.constrain(z)
    params = {k: v for k, v in values.items() if k in problem.system.parameters}
    solved = problem.solve(params)
    if problem.route == "kalman":
        predicted = jax.vmap(solved.observe)(kalman_run(solved, problem.observed, problem.outside, problem.start_state).predicted_mean)
    else:
        if problem.route == "read_off":
            start, states = problem.start_state, recover_shocks(solved, problem.observed, problem.outside, problem.start_state)[1]
        else:
            start, states = _sampled_path(problem, values, solved)
        previous = jnp.concatenate([start[None], states[:-1]])
        zero = jnp.zeros(len(problem.system.shock_labels))
        predicted = jax.vmap(lambda state, outside: solved.observe(solved.step(state, zero, outside)))(previous, problem.outside)
    return jnp.where(solved.status == Status.OK, predicted, jnp.nan)
