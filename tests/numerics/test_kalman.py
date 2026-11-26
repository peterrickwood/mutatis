"""The Kalman filter and smoother against the joint normal distribution of all states and observations, written out in full."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
import scipy.stats as ss

from mutatis.numerics.kalman import kalman_filter, rts_smoother, stationary_covariance
from mutatis.numerics.status import Status
from tests.numerics import reference


def random_model(rng, n=3, k=2, t=6, missing=0.0):
    T = rng.normal(size=(n, n))
    T *= 0.9 / np.max(np.abs(np.linalg.eigvals(T)))
    R = rng.normal(size=(n, n))
    m0, P0 = rng.normal(size=n), np.eye(n) + 0.3 * np.ones((n, n))
    Z, H = rng.normal(size=(k, n)), np.diag(rng.uniform(0.1, 0.5, k))
    c, d = rng.normal(size=(t, n)), rng.normal(size=(t, k))
    y = rng.normal(size=(t, k)) * 2.0
    y[rng.uniform(size=(t, k)) < missing] = np.nan
    if missing:
        y[2] = np.nan  # a period with nothing observed
    return dict(y=y, m0=m0, P0=P0, T=T, W=R @ R.T, Z=Z, H=H, c=c, d=d)


def as_jax(model):
    return {name: jnp.asarray(value) for name, value in model.items()}


@pytest.mark.parametrize("missing", [0.0, 0.3])
@pytest.mark.parametrize("seed", range(4))
def test_filter_and_smoother_match_the_joint_distribution(seed, missing):
    rng = np.random.default_rng(seed)
    model = random_model(rng, missing=missing)
    filtered = kalman_filter(**as_jax(model))
    smoothed = rts_smoother(filtered, jnp.asarray(model["T"]))
    joint = reference.stacked_gaussian(*(model[name] for name in ("m0", "P0", "T", "W", "Z", "H", "c", "d")))
    t, k = model["y"].shape
    n = model["T"].shape[0]
    seen = np.flatnonzero(~np.isnan(model["y"].ravel()))
    # The likelihood of the observed entries
    expected = ss.multivariate_normal(joint[1][seen], joint[4][np.ix_(seen, seen)]).logpdf(model["y"].ravel()[seen])
    np.testing.assert_allclose(filtered.log_likelihood, expected, rtol=1e-10)
    # Filtered: each state given the data up to its own date. Smoothed: each state given all the data.
    for s in range(t):
        up_to_now = seen[seen < (s + 1) * k]
        mean, cov = reference.condition(*joint, model["y"], up_to_now)
        np.testing.assert_allclose(filtered.filtered_mean[s], mean[s * n:(s + 1) * n], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(filtered.filtered_cov[s], cov[s * n:(s + 1) * n, s * n:(s + 1) * n], rtol=1e-9, atol=1e-10)
    mean, cov = reference.condition(*joint, model["y"], seen)
    np.testing.assert_allclose(smoothed.mean.ravel(), mean, rtol=1e-9, atol=1e-10)
    for s in range(t):
        np.testing.assert_allclose(smoothed.cov[s], cov[s * n:(s + 1) * n, s * n:(s + 1) * n], rtol=1e-9, atol=1e-10)


def test_smoother_with_a_state_that_moves_without_noise():
    # The second state is a lagged copy of the first, so its noise variance is zero and the predicted covariance is singular.
    rng = np.random.default_rng(11)
    model = random_model(rng, n=2, k=1, t=5)
    model["T"] = np.array([[0.7, 0.0], [1.0, 0.0]])
    model["W"] = np.array([[1.0, 0.0], [0.0, 0.0]])
    model["P0"] = np.array([[1.0, 0.0], [0.0, 0.5]])
    smoothed = rts_smoother(kalman_filter(**as_jax(model)), jnp.asarray(model["T"]))
    joint = reference.stacked_gaussian(*(model[name] for name in ("m0", "P0", "T", "W", "Z", "H", "c", "d")))
    mean, _ = reference.condition(*joint, model["y"], np.arange(5))
    np.testing.assert_allclose(smoothed.mean.ravel(), mean, rtol=1e-8, atol=1e-10)


def test_log_likelihood_gradient_matches_finite_differences():
    rng = np.random.default_rng(21)
    model = as_jax(random_model(rng, missing=0.3))

    def log_lik(T, W, H):
        return kalman_filter(**{**model, "T": T, "W": W, "H": H}).log_likelihood

    gradients = jax.grad(log_lik, argnums=(0, 1, 2))(model["T"], model["W"], model["H"])
    h = 1e-6
    for which, gradient in enumerate(gradients):
        args = [model["T"], model["W"], model["H"]]
        direction = jnp.asarray(rng.normal(size=args[which].shape))
        if which:  # keep covariance matrices symmetric
            direction = direction + direction.T
        plus, minus = list(args), list(args)
        plus[which], minus[which] = args[which] + h * direction, args[which] - h * direction
        numeric = (log_lik(*plus) - log_lik(*minus)) / (2 * h)
        np.testing.assert_allclose(jnp.sum(gradient * direction), numeric, rtol=1e-6)


def test_stationary_covariance():
    rng = np.random.default_rng(31)
    T = rng.normal(size=(4, 4))
    T *= 0.97 / np.max(np.abs(np.linalg.eigvals(T)))
    R = rng.normal(size=(4, 4))
    X, status = stationary_covariance(jnp.asarray(T), jnp.asarray(R @ R.T))
    assert int(status) == Status.OK
    np.testing.assert_allclose(X, scipy.linalg.solve_discrete_lyapunov(T, R @ R.T), rtol=1e-9)
    _, status = stationary_covariance(jnp.eye(2), jnp.eye(2))  # a random walk has no long-run spread
    assert int(status) == Status.NOT_CONVERGED
