"""Interval summaries of posterior draws, defined as in Read (2022), so results can be compared with his directly.

Percentiles use the "hazen" definition, which is what MATLAB's prctile computes.
"""

import numpy as np


def shortest_centred_interval(draws: np.ndarray, level: float = 0.68) -> tuple[float, float]:
    """The interval [c - r, c + r] with the smallest r that holds `level` of the draws, with c searched over the draws themselves."""
    centres = np.sort(draws)
    distance = np.abs(centres[None, :] - draws[:, None])
    radius = np.percentile(distance, 100 * level, axis=0, method="hazen")
    best = int(np.argmin(radius))
    return float(centres[best] - radius[best]), float(centres[best] + radius[best])


def robust_credible_interval(lower: np.ndarray, upper: np.ndarray, level: float = 0.68, grid: int = 1000) -> tuple[float, float]:
    """The shortest interval [c - r, c + r] that contains the whole range [lower_i, upper_i] for `level` of the draws.

    Giacomini and Kitagawa's robust credible region, computed on a grid of centres as in Read (2022).
    """
    centres = np.linspace(lower.min(), upper.max(), grid)
    distance = np.maximum(np.abs(centres[None, :] - lower[:, None]), np.abs(centres[None, :] - upper[:, None]))
    radius = np.percentile(distance, 100 * level, axis=0, method="hazen")
    best = int(np.argmin(radius))
    return float(centres[best] - radius[best]), float(centres[best] + radius[best])
