"""Small posterior problems with known answers, implementing contract 2, for testing samplers."""

import jax
import jax.numpy as jnp
import numpy as np


class NormalMean:
    """Unknown means of d independent series, prior N(0, prior_sd^2) on each, data N(mean, 1). Everything is known exactly."""

    def __init__(self, data: np.ndarray, prior_sd: float = 10.0):
        self.y, self.prior_sd = jnp.asarray(data), prior_sd
        self.n, self.size = data.shape
        self.post_var = 1.0 / (1.0 / prior_sd**2 + self.n)
        self.post_mean = np.asarray(self.post_var * data.sum(axis=0))
        cov = np.eye(self.n) + prior_sd**2
        self.log_evidence = float(sum(jax.scipy.stats.multivariate_normal.logpdf(self.y[:, j], jnp.zeros(self.n), cov) for j in range(self.size)))

    def log_prior(self, z):
        return jnp.sum(jax.scipy.stats.norm.logpdf(z, 0.0, self.prior_sd))

    def log_likelihood(self, z):
        return jnp.sum(jax.scipy.stats.norm.logpdf(self.y, z, 1.0))

    def draw_prior(self, key):
        return self.prior_sd * jax.random.normal(key, (self.size,))


class TwoPeaks:
    """One unknown with prior N(0, 5^2) and a likelihood with two narrow peaks at -3 and +3, weighted 0.3 and 0.7.

    The posterior puts weight exactly 0.3 and 0.7 on the two peaks, because the prior density is the same at -3 and +3.
    """

    size = 1
    weights, centres, width = (0.3, 0.7), (-3.0, 3.0), 0.3

    def log_prior(self, z):
        return jnp.sum(jax.scipy.stats.norm.logpdf(z, 0.0, 5.0))

    def log_likelihood(self, z):
        parts = [jnp.log(w) + jax.scipy.stats.norm.logpdf(z[0], c, self.width) for w, c in zip(self.weights, self.centres)]
        return jax.scipy.special.logsumexp(jnp.stack(parts))

    def draw_prior(self, key):
        return 5.0 * jax.random.normal(key, (1,))
