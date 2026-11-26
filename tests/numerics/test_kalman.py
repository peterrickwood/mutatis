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


@pytest.mark.parametrize("radius", [0.99, 0.9999, 0.99999])
def test_stationary_covariance_near_the_unit_circle(radius):
    rng = np.random.default_rng(30)
    T = rng.normal(size=(30, 30))
    T *= radius / np.max(np.abs(np.linalg.eigvals(T)))
    R = rng.normal(size=(30, 30))
    X, status = stationary_covariance(jnp.asarray(T), jnp.asarray(R @ R.T))
    assert int(status) == Status.OK
    X, expected = np.asarray(X), scipy.linalg.solve_discrete_lyapunov(T, R @ R.T)
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(X - T @ X @ T.T - R @ R.T)) < 1e-13 * scale  # the residual (a backward error) is at rounding level
    # The answer's sensitivity grows like 1 / (1 - radius^2), so two correct methods may differ by rounding times that.
    assert np.max(np.abs(X - expected)) < 1e3 * np.finfo(float).eps / (1 - radius**2) * scale


def test_gradient_memory_stays_near_one_covariance_per_period():
    # Reverse mode keeps what the backward pass needs from every period. Kept naively that was about 7 matrices of n x n per
    # period per particle (43 GB for 50 states, 150 periods and 2,000 particles); recomputing each period's step in the backward
    # pass keeps only the state mean and covariance carried between periods. Checked on the compiled program, without running it.
    n, k, t, particles = 20, 5, 150, 100
    model = random_model(np.random.default_rng(0), n=n, k=k, t=t)
    log_lik = lambda T: kalman_filter(**{**as_jax(model), "T": T}).log_likelihood
    Ts = jnp.broadcast_to(jnp.asarray(model["T"]), (particles, n, n))
    compiled = jax.jit(jax.vmap(jax.grad(log_lik))).lower(Ts).compile()
    one_covariance_per_period = t * particles * n * n * 8
    assert compiled.memory_analysis().temp_size_in_bytes < 2 * one_covariance_per_period


# At moderate sizes, against statsmodels (medium tier) -------------------------------------------------------------------------

def simulated_model(rng, n, k, t, missing, exact, noise_rank):
    """A stable model with `exact` series observed without error and state noise of rank `noise_rank`, and data simulated from it.
    Also returns R, with W = R R'."""
    T = rng.normal(size=(n, n))
    T *= 0.95 / np.max(np.abs(np.linalg.eigvals(T)))
    R, Z = rng.normal(size=(n, noise_rank)), rng.normal(size=(k, n))
    h = rng.uniform(0.1, 0.5, k)
    h[:exact] = 0.0
    c, d, m0 = 0.1 * rng.normal(size=(t, n)), rng.normal(size=(t, k)), rng.normal(size=n)
    state, y = m0 + rng.normal(size=n), np.zeros((t, k))
    for s in range(t):
        state = c[s] + T @ state + R @ rng.normal(size=noise_rank)
        y[s] = d[s] + Z @ state + np.sqrt(h) * rng.normal(size=k)
    y[rng.uniform(size=(t, k)) < missing] = np.nan
    return dict(y=y, m0=m0, P0=np.eye(n), T=T, W=R @ R.T, Z=Z, H=np.diag(h), c=c, d=d), R


def statsmodels_filter_and_smoother(model):
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

    t, k = model["y"].shape
    n = model["T"].shape[0]
    smoother = KalmanSmoother(k_endog=k, k_states=n, k_posdef=n)
    smoother.bind(model["y"])
    smoother["design"], smoother["obs_cov"], smoother["transition"] = model["Z"], model["H"], model["T"]
    smoother["selection"], smoother["state_cov"], smoother["obs_intercept"] = np.eye(n), model["W"], model["d"].T
    smoother["state_intercept"] = np.column_stack([model["c"][1:].T, np.zeros(n)])  # in statsmodels c(t + 1) moves the state from t to t + 1
    smoother.initialize_known(model["c"][0] + model["T"] @ model["m0"], model["T"] @ model["P0"] @ model["T"].T + model["W"])
    out = smoother.smooth()
    return out.llf_obs.sum(), out.filtered_state.T, out.smoothed_state.T


@pytest.mark.medium
@pytest.mark.parametrize("n, k, missing, exact, noise_rank", [(20, 5, 0.0, 0, 20), (20, 5, 0.2, 2, 8), (50, 10, 0.2, 3, 50), (50, 10, 0.2, 3, 15)])
def test_matches_statsmodels_at_moderate_size(n, k, missing, exact, noise_rank):
    model, _ = simulated_model(np.random.default_rng(n + noise_rank), n, k, 200, missing, exact, noise_rank)
    filtered = kalman_filter(**as_jax(model))
    smoothed = rts_smoother(filtered, jnp.asarray(model["T"]))
    log_likelihood, filtered_means, smoothed_means = statsmodels_filter_and_smoother(model)
    np.testing.assert_allclose(filtered.log_likelihood, log_likelihood, rtol=1e-12)
    np.testing.assert_allclose(filtered.filtered_mean, filtered_means, rtol=0, atol=1e-12 * np.max(np.abs(filtered_means)))
    # The smoother inverts predicted covariances, which are singular when the state noise is: a pseudo-inverse here, another
    # route in statsmodels. They agreed to 7e-13 (relative) in the audit.
    np.testing.assert_allclose(smoothed.mean, smoothed_means, rtol=0, atol=1e-10 * np.max(np.abs(smoothed_means)))


@pytest.mark.medium
def test_log_likelihood_gradient_at_moderate_size():
    rng = np.random.default_rng(3)
    model, R = simulated_model(rng, 20, 5, 200, 0.2, 2, 8)
    base = as_jax(model)

    def log_lik(T, R, h):  # W = R R' stays positive semi-definite, and exact series stay exact, along every direction used
        return kalman_filter(**{**base, "T": T, "W": R @ R.T, "H": jnp.diag(h)}).log_likelihood

    args = [base["T"], jnp.asarray(R), jnp.diagonal(base["H"])]
    gradients = jax.grad(log_lik, argnums=(0, 1, 2))(*args)
    value = jax.jit(log_lik)
    for which, gradient in enumerate(gradients):
        direction = jnp.asarray(rng.normal(size=args[which].shape))
        if which == 2:
            direction = jnp.where(args[2] > 0, direction, 0.0)
        direction = direction / jnp.linalg.norm(direction) * jnp.linalg.norm(args[which])

        def central(step):
            plus, minus = list(args), list(args)
            plus[which], minus[which] = args[which] + step * direction, args[which] - step * direction
            return float(value(*plus) - value(*minus)) / (2 * step)

        fine, coarse = central(5e-6), central(1e-5)
        numeric = (4 * fine - coarse) / 3  # Richardson extrapolation removes the step^2 term
        # Uncertain by about the change between the two steps, plus rounding in the likelihood divided by the step.
        assert abs(float(jnp.sum(gradient * direction)) - numeric) < 10 * abs(fine - coarse) + 1e-7 * abs(numeric)
