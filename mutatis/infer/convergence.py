"""Convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter and Burkner, 2021) in NumPy: rank-normalised split R-hat, and
bulk and tail effective sample sizes, computed exactly as BlackJAX 1.7 computes them (tested against it).

Why not BlackJAX's own: the sampler's stopping rule calls these after every block of draws, and each new number of draws made
BlackJAX's JAX versions compile afresh, about 11 s a call run eagerly and 7 s under `jax.jit` (measured 2026-11-27), which
dominated the run time on cheap targets. NumPy needs no compilation.

All functions take draws of shape (chains, draws, unknowns) and return one value per unknown.
"""

import numpy as np
import scipy.fft
import scipy.special
import scipy.stats


def _split(x: np.ndarray) -> np.ndarray:
    """Each chain cut into two halves (an odd last draw dropped): (2 x chains, draws // 2, unknowns)."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Average ranks over all chains and draws pooled, mapped to normal scores by Blom's plotting position."""
    m, n = x.shape[:2]
    flat = x.reshape(m * n, -1)
    ranks = scipy.stats.rankdata(flat, axis=0, method="average")
    z = scipy.special.ndtri((ranks - 3.0 / 8.0) / (m * n + 1.0 / 4.0))
    z[:, np.any(np.isnan(flat), axis=0)] = np.nan
    return z.reshape(x.shape)


def _potential_scale_reduction(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    between = n * x.mean(axis=1).var(axis=0, ddof=1)
    within = x.var(axis=1, ddof=1).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):   # an unknown that never moves has no R-hat: nan, as in BlackJAX
        return np.sqrt((between / within + n - 1.0) / n)


def effective_sample_size(x: np.ndarray) -> np.ndarray:
    """The effective sample size of draws (chains, draws, unknowns), by FFT autocovariances and Geyer's initial monotone
    sequence, as BlackJAX's `effective_sample_size`."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape[:2]
    varies = np.any(x != x[:, :1], axis=(0, 1))
    chain_means = x.mean(axis=1, keepdims=True)
    size = scipy.fft.next_fast_len(2 * n)
    spectrum = np.fft.rfft(x - chain_means, n=size, axis=1)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)[:, :n] / n
    mean_autocov = autocov.mean(axis=0)                                   # (n, unknowns)
    var0 = mean_autocov[0] * n / (n - 1.0)
    degenerate = np.isfinite(var0) & (~varies | (var0 <= 0.0))
    weighted = var0 * (n - 1.0) / n
    if m > 1:
        weighted = weighted + chain_means[:, 0].var(axis=0, ddof=1)
    weighted = np.where(degenerate, 1.0, weighted)
    even_n = n - n % 2
    rho = np.concatenate([np.ones_like(var0)[None], 1.0 - (var0 - mean_autocov[1:even_n]) / weighted], axis=0)
    rho_even, rho_odd = rho[0::2], rho[1::2]
    # Geyer's initial positive sequence: pairs are kept while their sums stay positive
    positive = (rho_even + rho_odd) > 0.0
    keep = np.cumprod(positive, axis=0).astype(bool)
    last = np.where(keep.any(axis=0), keep.shape[0] - 1 - np.argmax(keep[::-1], axis=0), 0)   # last kept pair (0 if none)
    rho_odd = np.where(keep, rho_odd, 0.0)
    columns = np.arange(rho_even.shape[1])
    keep_even = keep.copy()
    after = np.minimum(last + 1, rho_even.shape[0] - 1)
    inside = last + 1 < rho_even.shape[0]
    keep_even[after[inside], columns[inside]] = rho_even[after[inside], columns[inside]] > 0
    rho_even = np.where(keep_even, rho_even, 0.0)
    # Geyer's initial monotone sequence
    sums = rho_even + rho_odd
    running = np.minimum.accumulate(sums, axis=0)
    lowered = sums > np.concatenate([sums[:1], running[:-1]], axis=0)
    rho_even = np.where(lowered, running / 2.0, rho_even)
    rho_odd = np.where(lowered, running / 2.0, rho_odd)
    total = m * n
    beyond = rho_even[after, columns]   # as BlackJAX: past the last pair, its index is clamped to the last pair
    tau = -1.0 + 2.0 * np.sum(rho_even + rho_odd, axis=0) - beyond
    tau = np.maximum(tau, 1.0 / np.log10(total))
    ess = np.where(degenerate, 0.0, total / tau)
    return np.where(np.any(np.isnan(x), axis=(0, 1)), np.nan, ess)


def split_rhat(draws: np.ndarray) -> np.ndarray:
    """Rank-normalised split R-hat: the larger of the bulk version and the folded (spread) version."""
    x = _split(np.asarray(draws, dtype=float))
    bulk = _potential_scale_reduction(_rank_normalise(x))
    flat = x.reshape(-1, x.shape[2])
    folded = np.abs(x - np.median(flat, axis=0))
    return np.maximum(bulk, _potential_scale_reduction(_rank_normalise(folded)))


def bulk_ess(draws: np.ndarray) -> np.ndarray:
    return effective_sample_size(_rank_normalise(_split(np.asarray(draws, dtype=float))))


def tail_ess(draws: np.ndarray, prob: float = 0.90) -> np.ndarray:
    """The smaller effective sample size of the indicators of the lower and upper tails (5% and 95% quantiles by default)."""
    x = _split(np.asarray(draws, dtype=float))
    flat = x.reshape(-1, x.shape[2])
    low, high = np.quantile(flat, (1.0 - prob) / 2.0, axis=0), np.quantile(flat, (1.0 + prob) / 2.0, axis=0)
    ess = np.minimum(effective_sample_size((x <= low).astype(float)), effective_sample_size((x >= high).astype(float)))
    return np.where(np.any(np.isnan(x), axis=(0, 1)), np.nan, ess)
