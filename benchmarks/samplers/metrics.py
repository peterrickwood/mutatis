"""Accuracy of a sampler's draws against a target's known answer.

For each checked quantity: the error in the posterior mean, in posterior standard deviations; the ratio of the draws'
standard deviation to the true one; and the Kolmogorov-Smirnov distance between the draws and the true distribution (the
largest gap between the two cumulative distributions), where the true distribution is known. For targets with separate peaks,
the share of draws in each peak against its true weight. Draws carry weights (SMC's do); all summaries use them.

The criteria, applied to the worst quantity of a run, are fixed here and stated in the report: mean error within 0.1 posterior
sd, sd ratio within [0.9, 1.1], each peak's weight within 0.05, log evidence within 0.5.
"""

import numpy as np

MEAN_ERROR_LIMIT = 0.1
SD_RATIO_RANGE = (0.9, 1.1)
PEAK_WEIGHT_LIMIT = 0.05
EVIDENCE_LIMIT = 0.5


def weighted_ks(values: np.ndarray, weights: np.ndarray, cdf) -> float:
    """Largest gap between the weighted empirical cumulative distribution of `values` and `cdf`."""
    order = np.argsort(values)
    x, w = values[order], weights[order] / np.sum(weights)
    upper = np.cumsum(w)
    lower = upper - w
    f = np.asarray(cdf(x), dtype=float)
    return float(max(np.max(np.abs(upper - f)), np.max(np.abs(lower - f))))


def accuracy(target, particles: np.ndarray, weights: np.ndarray) -> dict:
    ref = target.reference
    out = {"n_quantities": len(ref.names)}
    if not np.all(np.isfinite(particles)):
        return out | {"finite": False, "posterior_ok": False, "failed": ["some draws are not finite"]}
    q = target.quantities(particles)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    mean = w @ q
    sd = np.sqrt(np.maximum(w @ (q - mean) ** 2, 0.0))
    mean_error = (mean - ref.mean) / ref.sd
    sd_ratio = sd / ref.sd
    ks = np.array([weighted_ks(q[:, j], w, cdf) if cdf is not None else np.nan for j, cdf in enumerate(ref.cdfs)])
    worst = int(np.argmax(np.abs(mean_error)))
    out |= {"finite": True,
            "max_abs_mean_error": float(np.max(np.abs(mean_error))), "worst_quantity": ref.names[worst],
            "min_sd_ratio": float(np.min(sd_ratio)), "max_sd_ratio": float(np.max(sd_ratio)),
            "max_ks": float(np.nanmax(ks)) if np.any(np.isfinite(ks)) else None,
            "mean_error": _short(mean_error), "sd_ratio": _short(sd_ratio), "ks": _short(ks)}
    failed = []
    if out["max_abs_mean_error"] > MEAN_ERROR_LIMIT:
        failed.append(f"mean error {out['max_abs_mean_error']:.3f} sd in {ref.names[worst]}")
    if out["min_sd_ratio"] < SD_RATIO_RANGE[0] or out["max_sd_ratio"] > SD_RATIO_RANGE[1]:
        failed.append(f"sd ratio {out['min_sd_ratio']:.3f} to {out['max_sd_ratio']:.3f}")
    if ref.peak_weights is not None:
        peaks = target.peak_of(particles)
        shares = [float(np.sum(w[peaks == k])) for k in range(len(ref.peak_weights))]
        error = float(max(abs(s - t) for s, t in zip(shares, ref.peak_weights)))
        out |= {"peak_shares": shares, "peak_weight_error": error}
        if error > PEAK_WEIGHT_LIMIT:
            failed.append(f"peak weights {', '.join(f'{s:.3f}' for s in shares)} against {', '.join(f'{t:g}' for t in ref.peak_weights)}")
    out["failed"] = failed
    out["posterior_ok"] = not failed
    return out


def evidence_accuracy(estimate: float | None, standard_error: float | None, truth: float) -> dict:
    """`ok` is None when the true log evidence is not known (NaN: the Inference Gym's Bayesian models)."""
    if estimate is None or not np.isfinite(estimate):
        return {"log_value": None, "standard_error": standard_error, "error": None, "ok": None if np.isnan(truth) else False}
    if np.isnan(truth):
        return {"log_value": float(estimate), "standard_error": None if standard_error is None else float(standard_error),
                "error": None, "ok": None}
    error = float(estimate - truth)
    return {"log_value": float(estimate), "standard_error": None if standard_error is None else float(standard_error),
            "error": error, "ok": abs(error) <= EVIDENCE_LIMIT}


def _short(values: np.ndarray, limit: int = 400) -> list:
    """Per-quantity values for the results file, as plain floats (None for missing), at most `limit` of them."""
    return [None if not np.isfinite(v) else round(float(v), 6) for v in np.asarray(values)[:limit]]
