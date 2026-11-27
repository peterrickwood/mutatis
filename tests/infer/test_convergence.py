"""The NumPy convergence diagnostics give BlackJAX's values, on draws of every kind the sampler meets."""

import blackjax.diagnostics as bd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.infer.convergence import bulk_ess, effective_sample_size, split_rhat, tail_ess


def _cases():
    rng = np.random.default_rng(0)
    yield "independent", rng.normal(size=(4, 501, 3))
    ar = np.zeros((4, 400, 3))
    for t in range(1, 400):
        ar[:, t] = 0.9 * ar[:, t - 1] + rng.normal(size=(4, 3))
    yield "autocorrelated", ar
    walk = np.cumsum(rng.normal(size=(2, 40, 2)), axis=1)          # every autocorrelation pair stays positive
    yield "random walk, short", walk
    yield "ties", rng.integers(0, 3, size=(4, 300, 2)).astype(float)
    stuck = rng.normal(size=(4, 300, 2))
    stuck[3] += 4.0
    yield "one chain elsewhere", stuck
    constant = rng.normal(size=(4, 200, 2))
    constant[:, :, 1] = 1.5
    yield "a constant unknown", constant
    yield "eight chains", rng.standard_t(3, size=(8, 1000, 5))


@pytest.mark.parametrize("name, draws", list(_cases()))
def test_matches_blackjax(name, draws):
    x = jnp.asarray(draws)
    expected = {
        "rhat": np.asarray(jax.vmap(bd.rhat, in_axes=2)(x)),
        "bulk": np.asarray(jax.vmap(bd.ess_bulk, in_axes=2)(x)),
        "tail": np.asarray(jax.vmap(bd.ess_tail, in_axes=2)(x)),
        "plain": np.asarray(jax.vmap(bd.effective_sample_size, in_axes=2)(x)),
    }
    got = {"rhat": split_rhat(draws), "bulk": bulk_ess(draws), "tail": tail_ess(draws), "plain": effective_sample_size(draws)}
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], rtol=1e-9, atol=1e-12, err_msg=f"{name}: {key}")
