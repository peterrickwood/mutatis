"""Square-root factors of covariance matrices that may be singular."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from mutatis import _jax_config  # noqa: F401

JITTER = 1e-12


def cholesky_with_jitter(cov: Float[Array, "n n"], relative: float = JITTER) -> Float[Array, "n n"]:
    """A lower-triangular L with L L' = cov + j I, where j is `relative` times the largest variance (at least `relative`).

    Covariances of model states are often singular, because some variables are exact functions of others; a plain Cholesky
    factor then does not exist. The jitter changes results by amounts of order `relative`.
    """
    jitter = relative * jnp.maximum(jnp.max(jnp.diagonal(cov)), 1.0)
    return jnp.linalg.cholesky(cov + jitter * jnp.eye(cov.shape[0]))
