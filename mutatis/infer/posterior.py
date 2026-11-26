"""The result of sampling, whatever the sampler: weighted draws of the unconstrained unknowns."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Posterior:
    particles: np.ndarray                  # (draws, unknowns), unconstrained
    weights: np.ndarray                    # sum to one
    log_marginal_likelihood: float | None  # None when it is not defined (an improper prior) or not computed
    method: str
    diagnostics: dict = field(default_factory=dict)

    def resample(self, key, n: int | None = None) -> np.ndarray:
        n = len(self.weights) if n is None else n
        index = jax.random.choice(key, len(self.weights), (n,), p=jnp.asarray(self.weights))
        return self.particles[np.asarray(index)]

    def mean(self) -> np.ndarray:
        return np.average(self.particles, axis=0, weights=self.weights)

    def sd(self) -> np.ndarray:
        return np.sqrt(np.average((self.particles - self.mean()) ** 2, axis=0, weights=self.weights))
