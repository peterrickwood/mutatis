"""The Kalman filter and smoother for linear models with normally distributed shocks and measurement errors.

The model is

    s(t) = c(t) + T s(t-1) + w(t),      w(t) ~ N(0, W)      how the hidden state moves
    y(t) = d(t) + Z s(t)   + v(t),      v(t) ~ N(0, H)      what is observed

with s(0) ~ N(m0, P0): the state in the period before the first observation. c and d may change over time, which is how
outside variables enter. Missing observations are NaN in y; each period uses only the entries that are present.

The filter walks through the data one period at a time: predict the state, compare the prediction with that period's data,
update. It gives the log likelihood of the data, with the hidden states and shocks averaged out exactly, and the filtered
state estimates. The smoother then runs backwards to give estimates of the state at every date using all the data.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from mutatis import _jax_config  # noqa: F401
from mutatis.numerics.status import Status

LOG_2PI = jnp.log(2.0 * jnp.pi)


class Filtered(NamedTuple):
    log_likelihood: Float[Array, ""]
    log_likelihood_by_period: Float[Array, " t"]
    predicted_mean: Float[Array, "t n"]
    predicted_cov: Float[Array, "t n n"]
    filtered_mean: Float[Array, "t n"]
    filtered_cov: Float[Array, "t n n"]


class Smoothed(NamedTuple):
    mean: Float[Array, "t n"]
    cov: Float[Array, "t n n"]


def kalman_filter(y: Float[Array, "t k"], m0: Float[Array, " n"], P0: Float[Array, "n n"], T: Float[Array, "n n"], W: Float[Array, "n n"],
                  Z: Float[Array, "k n"], H: Float[Array, "k k"], c: Float[Array, "t n"], d: Float[Array, "t k"]) -> Filtered:
    """Run the filter. NaN entries of y are treated as missing."""
    n = T.shape[0]

    def step(carry, inputs):
        m, P = carry
        y_t, c_t, d_t = inputs
        m_pred = c_t + T @ m
        P_pred = _symmetric(T @ P @ T.T + W)
        seen = ~jnp.isnan(y_t)
        m_new, P_new, log_lik = _update(m_pred, P_pred, jnp.where(seen, y_t, 0.0), seen, Z, H, d_t, n)
        return (m_new, P_new), (log_lik, m_pred, P_pred, m_new, P_new)

    _, (log_lik, m_pred, P_pred, m_filt, P_filt) = jax.lax.scan(step, (m0, P0), (y, c, d))
    return Filtered(jnp.sum(log_lik), log_lik, m_pred, P_pred, m_filt, P_filt)


def _update(m_pred, P_pred, y_t, seen: Bool[Array, " k"], Z, H, d_t, n):
    # Rows of Z and H for missing entries are zeroed, and the forecast-error covariance gets a 1 on the diagonal there, so a
    # missing entry contributes nothing to the update or the likelihood while every array keeps its shape.
    both_seen = seen[:, None] & seen[None, :]
    Z_seen = jnp.where(seen[:, None], Z, 0.0)
    H_seen = jnp.where(both_seen, H, 0.0)
    error = jnp.where(seen, y_t - d_t - Z @ m_pred, 0.0)
    F = _symmetric(Z_seen @ P_pred @ Z_seen.T + H_seen) + jnp.diag(jnp.where(seen, 0.0, 1.0))
    L = jnp.linalg.cholesky(F)
    gain = jax.scipy.linalg.cho_solve((L, True), Z_seen @ P_pred).T  # P Z' F^-1
    m_new = m_pred + gain @ error
    keep = jnp.eye(n) - gain @ Z_seen
    P_new = _symmetric(keep @ P_pred @ keep.T + gain @ H_seen @ gain.T)  # Joseph form: stays positive semi-definite
    whitened = jax.scipy.linalg.solve_triangular(L, error, lower=True)
    log_lik = -0.5 * (jnp.sum(seen) * LOG_2PI + 2.0 * jnp.sum(jnp.log(jnp.diagonal(L))) + whitened @ whitened)
    return m_new, P_new, log_lik


def rts_smoother(filtered: Filtered, T: Float[Array, "n n"]) -> Smoothed:
    """Rauch-Tung-Striebel smoother: the state at each date given all the data.

    A pseudo-inverse is used for the predicted covariance, which is singular whenever part of the state moves without noise
    (a lagged copy of a variable, for example).
    """
    def step(carry, inputs):
        m_next, P_next = carry
        m_f, P_f, m_pred_next, P_pred_next = inputs
        G = P_f @ T.T @ jnp.linalg.pinv(P_pred_next, hermitian=True)
        m = m_f + G @ (m_next - m_pred_next)
        P = _symmetric(P_f + G @ (P_next - P_pred_next) @ G.T)
        return (m, P), (m, P)

    last = (filtered.filtered_mean[-1], filtered.filtered_cov[-1])
    inputs = (filtered.filtered_mean[:-1], filtered.filtered_cov[:-1], filtered.predicted_mean[1:], filtered.predicted_cov[1:])
    _, (means, covs) = jax.lax.scan(step, last, inputs, reverse=True)
    return Smoothed(jnp.concatenate([means, last[0][None]]), jnp.concatenate([covs, last[1][None]]))


def stationary_covariance(T: Float[Array, "n n"], W: Float[Array, "n n"], tolerance: float = 1e-14,
                          max_iterations: int = 64) -> tuple[Float[Array, "n n"], jax.Array]:
    """The covariance X with X = T X T' + W: the long-run spread of a state that does not drift. And the outcome status.

    Uses doubling: X = W + T W T' + T^2 W T^2' + ..., summed in blocks that double in length, so each iteration squares T.
    Only meaningful when every eigenvalue of T is inside the unit circle; otherwise the sum does not settle.
    """
    def step(carry):
        X, A, iteration, _ = carry
        X_next = X + A @ X @ A.T
        change = jnp.max(jnp.abs(X_next - X)) / jnp.maximum(jnp.max(jnp.abs(X_next)), jnp.finfo(X.dtype).tiny)
        return X_next, A @ A, iteration + 1, change

    def keep_going(carry):
        *_, iteration, change = carry
        return (change > tolerance) & (iteration < max_iterations) & jnp.isfinite(change)

    big = jnp.array(jnp.finfo(W.dtype).max, dtype=W.dtype)
    X, _, _, change = jax.lax.while_loop(keep_going, step, (W, T, jnp.array(0), big))
    ok = (change <= tolerance) & jnp.all(jnp.isfinite(X))
    return _symmetric(X), jnp.where(ok, Status.OK, Status.NOT_CONVERGED)


def _symmetric(X):
    return 0.5 * (X + X.T)
