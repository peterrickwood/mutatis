"""Domains: the set of values an unknown may take, and the map between that set and unconstrained real numbers.

Samplers work on unconstrained numbers. A domain says how to turn an unconstrained vector `z` into an allowed value `x`
(`to_constrained`), how to go back (`to_unconstrained`), and the log of the Jacobian determinant of the forward map, which a
density needs when it is moved from `x` to `z`.

A domain with no such map (the orthogonal matrices) can still describe what is allowed; unknowns in it cannot be sampled by
gradient methods and are drawn another way.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float
from jaxtyping import Real as RealNumber

from mutatis import _jax_config  # noqa: F401


class Domain(eqx.Module):
    """The set of values an unknown may take."""

    has_transform: eqx.AbstractClassVar[bool]

    def contains(self, x: RealNumber[ArrayLike, "..."]) -> Bool[Array, "..."]:
        raise NotImplementedError

    def unconstrained_size(self, shape: tuple[int, ...]) -> int:
        """Length of the unconstrained vector for a value of this shape."""
        return math.prod(shape)

    def to_constrained(self, z: Float[Array, " k"], shape: tuple[int, ...]) -> Float[Array, "..."]:
        raise NotImplementedError

    def to_unconstrained(self, x: RealNumber[ArrayLike, "..."]) -> Float[Array, " k"]:
        raise NotImplementedError

    def log_det_jacobian(self, z: Float[Array, " k"], shape: tuple[int, ...]) -> Float[Array, ""]:
        """log |det d to_constrained(z) / dz|."""
        raise NotImplementedError


class Real(Domain):
    """Any real number, elementwise."""

    has_transform = True

    def contains(self, x):
        return jnp.isfinite(x)

    def to_constrained(self, z, shape):
        return jnp.reshape(z, shape)

    def to_unconstrained(self, x):
        return jnp.ravel(x)

    def log_det_jacobian(self, z, shape):
        return jnp.zeros(())


class Positive(Domain):
    """Positive numbers, elementwise. x = exp(z)."""

    has_transform = True

    def contains(self, x):
        return x > 0

    def to_constrained(self, z, shape):
        return jnp.reshape(jnp.exp(z), shape)

    def to_unconstrained(self, x):
        return jnp.ravel(jnp.log(x))

    def log_det_jacobian(self, z, shape):
        return jnp.sum(z)


class Interval(Domain):
    """Numbers strictly between `low` and `high`, elementwise. x = low + (high - low) * sigmoid(z)."""

    low: float = 0.0
    high: float = 1.0
    has_transform = True

    def contains(self, x):
        return (x > self.low) & (x < self.high)

    def to_constrained(self, z, shape):
        return jnp.reshape(self.low + (self.high - self.low) * jax.nn.sigmoid(z), shape)

    def to_unconstrained(self, x):
        u = (jnp.ravel(x) - self.low) / (self.high - self.low)
        return jnp.log(u) - jnp.log1p(-u)

    def log_det_jacobian(self, z, shape):
        # d sigmoid / dz = sigmoid(z) * sigmoid(-z), and log sigmoid(z) = -softplus(-z)
        return jnp.sum(jnp.log(self.high - self.low) - jax.nn.softplus(-z) - jax.nn.softplus(z))


def UnitInterval() -> Interval:
    """Numbers strictly between 0 and 1."""
    return Interval(0.0, 1.0)


class PositiveDefinite(Domain):
    """Symmetric positive definite n-by-n matrices.

    x = L L', where L is lower triangular with a positive diagonal. `z` holds the entries of L's lower triangle, row by row,
    with the logarithm of each diagonal entry in place of the entry itself.
    """

    has_transform = True

    def contains(self, x):
        symmetric = jnp.allclose(x, jnp.swapaxes(x, -1, -2))
        return symmetric & jnp.all(jnp.linalg.eigvalsh(x) > 0)

    def unconstrained_size(self, shape):
        n = _square_size(shape)
        return n * (n + 1) // 2

    def _lower(self, z, n):
        rows, cols = jnp.tril_indices(n)
        values = jnp.where(rows == cols, jnp.exp(z), z)
        return jnp.zeros((n, n), dtype=z.dtype).at[rows, cols].set(values)

    def to_constrained(self, z, shape):
        L = self._lower(z, _square_size(shape))
        return L @ L.T

    def to_unconstrained(self, x):
        n = x.shape[-1]
        L = jnp.linalg.cholesky(x)
        rows, cols = jnp.tril_indices(n)
        values = L[rows, cols]
        return jnp.where(rows == cols, jnp.log(values), values)

    def log_det_jacobian(self, z, shape):
        # Two steps. From L to x = L L' the Jacobian determinant is 2^n * prod_i L_ii^(n - i), and from z to L each diagonal
        # entry adds a factor L_ii (counting i from 0). With log L_ii = z_ii this is n log 2 + sum_i (n - i + 1) z_ii.
        n = _square_size(shape)
        rows, cols = jnp.tril_indices(n)
        diagonal = rows == cols
        powers = jnp.where(diagonal, n - rows + 1, 0)
        return n * jnp.log(2.0) + jnp.sum(jnp.where(diagonal, powers * z, 0.0))


class Orthogonal(Domain):
    """Orthogonal n-by-n matrices: x' x = I.

    There is no smooth map from unconstrained numbers onto all of them, so `has_transform` is False and gradient samplers must
    not move them. For storage alongside other unknowns, a matrix is simply flattened (`to_unconstrained`) and reshaped back
    (`to_constrained`); methods that draw such unknowns directly use this.
    """

    tolerance: float = 1e-8
    has_transform = False

    def contains(self, x):
        n = x.shape[-1]
        return jnp.max(jnp.abs(jnp.swapaxes(x, -1, -2) @ x - jnp.eye(n))) < self.tolerance

    def unconstrained_size(self, shape):
        _square_size(shape)
        return shape[0] * shape[1]

    def to_constrained(self, z, shape):
        return jnp.reshape(z, shape)

    def to_unconstrained(self, x):
        return jnp.ravel(x)

    def log_det_jacobian(self, z, shape):
        return jnp.zeros(())


def _square_size(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"expected a square matrix shape, got {shape}")
    return shape[0]
