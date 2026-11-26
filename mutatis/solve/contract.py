"""Contract 1: what every solved model offers, whatever produced it.

A solved model is a rule for moving the state forward, plus a rule for what would be observed:

    state(t) = step(state(t-1), shocks(t), outside(t))
    data(t)  = observe(state(t)) + measurement error

together with the probability of any shocks, of any measurement errors, and of the starting state. Estimation code sees solved
models only through this class (see the import rules in PLAN.md), so a new kind of solver plugs in by implementing it.

Optional abilities are methods that return None when the model does not have them. Estimation chooses its method by asking:
`linear_form` (the step and observation are straight lines) and `gaussian_noise` (shocks and measurement errors are normal with
mean zero).
"""

from typing import NamedTuple

import equinox as eqx
import jax
from jaxtyping import Array, Float

from mutatis import _jax_config  # noqa: F401


class LinearForm(NamedTuple):
    """state(t) = kappa + P state(t-1) + Q shocks(t) + R outside(t);  observe(state) = d + Z state."""

    kappa: Float[Array, " n"]
    P: Float[Array, "n n"]
    Q: Float[Array, "n k"]
    R: Float[Array, "n m"]
    d: Float[Array, " o"]
    Z: Float[Array, "o n"]


class GaussianNoise(NamedTuple):
    shock_cov: Float[Array, "k k"]
    error_variance: Float[Array, " o"]   # zero for series observed exactly


class SolvedModel(eqx.Module):
    """The interface. `status` is a `mutatis.numerics.Status` code; anything other than OK means the parameter values give no usable model."""

    status: eqx.AbstractVar[jax.Array]
    exact: eqx.AbstractVar[tuple[bool, ...]]   # per observed series: observed exactly (no measurement error)

    def step(self, state, shocks, outside):
        raise NotImplementedError

    def observe(self, state):
        raise NotImplementedError

    def shock_log_prob(self, shocks) -> jax.Array:
        """Log probability of one period's shocks."""
        raise NotImplementedError

    def sample_shocks(self, key, n_periods: int):
        raise NotImplementedError

    def error_log_prob(self, observed, predicted) -> jax.Array:
        """Log probability of one period's measurement errors, over series that have them and are not missing (NaN)."""
        raise NotImplementedError

    def sample_errors(self, key, n_periods: int):
        """Measurement errors for n periods: (periods, observed series), zero for series observed exactly."""
        raise NotImplementedError

    def initial_log_prob(self, state) -> jax.Array:
        raise NotImplementedError

    def sample_initial(self, key):
        raise NotImplementedError

    def initial_gaussian(self):
        """(mean, covariance) of the starting state when it is normal, else None."""
        return None

    def linear_form(self) -> LinearForm | None:
        return None

    def gaussian_noise(self) -> GaussianNoise | None:
        return None
