"""Domains: the map to unconstrained numbers must invert exactly, and its log Jacobian must match automatic differentiation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.dist import Interval, Orthogonal, Positive, PositiveDefinite, Real, UnitInterval

ELEMENTWISE = [Real(), Positive(), UnitInterval(), Interval(-2.0, 3.0)]
SHAPES = [(), (3,), (2, 4)]


def _z(domain, shape, seed):
    return jnp.asarray(np.random.default_rng(seed).normal(size=domain.unconstrained_size(shape)))


def _jacobian_log_det(domain, z, shape):
    """log |det J| of z -> x, computed by automatic differentiation on a square version of the map."""
    if isinstance(domain, PositiveDefinite):
        rows, cols = np.tril_indices(shape[0])
        forward = lambda v: domain.to_constrained(v, shape)[rows, cols]  # the lower triangle determines x
    else:
        forward = lambda v: jnp.ravel(domain.to_constrained(v, shape))
    return jnp.linalg.slogdet(jax.jacfwd(forward)(z))[1]


@pytest.mark.parametrize("domain", ELEMENTWISE, ids=type)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", range(5))
def test_elementwise_round_trip_and_jacobian(domain, shape, seed):
    z = _z(domain, shape, seed)
    x = domain.to_constrained(z, shape)
    assert x.shape == shape
    assert bool(jnp.all(domain.contains(x)))
    np.testing.assert_allclose(domain.to_unconstrained(x), z, rtol=1e-10, atol=1e-12)
    if z.size:
        np.testing.assert_allclose(domain.log_det_jacobian(z, shape), _jacobian_log_det(domain, z, shape), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", range(5))
def test_positive_definite_round_trip_and_jacobian(n, seed):
    domain, shape = PositiveDefinite(), (n, n)
    z = _z(domain, shape, seed)
    x = domain.to_constrained(z, shape)
    assert bool(domain.contains(x))
    np.testing.assert_allclose(x, x.T, atol=0)
    np.testing.assert_allclose(domain.to_unconstrained(x), z, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(domain.log_det_jacobian(z, shape), _jacobian_log_det(domain, z, shape), rtol=1e-10, atol=1e-10)


def test_contains():
    assert not bool(Positive().contains(-1.0))
    assert not bool(Interval(-2.0, 3.0).contains(3.0))
    assert bool(Interval(-2.0, 3.0).contains(2.999))
    assert not bool(PositiveDefinite().contains(jnp.array([[1.0, 2.0], [2.0, 1.0]])))  # eigenvalues 3 and -1
    assert not bool(PositiveDefinite().contains(jnp.array([[2.0, 1.0], [0.0, 2.0]])))  # not symmetric


def test_orthogonal_domain_is_stored_flat_but_has_no_transform():
    q, _ = jnp.linalg.qr(jnp.asarray(np.random.default_rng(0).normal(size=(4, 4))))
    assert bool(Orthogonal().contains(q))
    assert not bool(Orthogonal().contains(q + 1e-3))
    assert not Orthogonal.has_transform  # gradient samplers must not move it
    assert Orthogonal().unconstrained_size((4, 4)) == 16
    np.testing.assert_array_equal(Orthogonal().to_constrained(Orthogonal().to_unconstrained(q), (4, 4)), q)
    with pytest.raises(ValueError):
        Orthogonal().unconstrained_size((4, 3))
