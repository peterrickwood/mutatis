"""Draws docs/figures/prior_strength.svg: one parameter under priors of decreasing strength.

Schematic only. Each panel is scaled to its own peak, so heights are not comparable between panels.
Run: python docs/figures/prior_strength.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SURFACE, TEXT, TEXT_SECONDARY, SERIES = "#ffffff", "#191919", "#666666", "#257de0"
X_MIN, X_MAX, VALUE = -2.0, 5.0, 1.5

x = np.linspace(X_MIN, X_MAX, 300)
flat = np.full_like(x, 0.45)
bell = lambda sd: norm.pdf(x, VALUE, sd) / norm.pdf(VALUE, VALUE, sd)  # scaled so the peak is 1
panels = [  # (title, subtitle, curve height, or None for the single-value spike)
    ("Fixed", "all weight on one value", None),
    ("Tight prior", "probably close to this value", bell(0.3)),
    ("Loose prior", "somewhere around here", bell(1.1)),
    ("Sign restriction", "any value, as long as it is positive", np.where(x > 0, flat, 0.0)),
    ("No restriction", "any value at all", flat),
]

plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10})
fig, axes = plt.subplots(1, len(panels), figsize=(12, 2.9), sharex=True, facecolor=SURFACE)

for ax, (title, subtitle, height) in zip(axes, panels):
    if height is None:
        ax.plot([VALUE, VALUE], [0, 1], color=SERIES, linewidth=2, solid_capstyle="round")
    else:
        ax.fill_between(x, 0, height, color=SERIES, alpha=0.18, linewidth=0)
        ax.plot(x, height, color=SERIES, linewidth=2)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=22)
    ax.text(0.5, 1.06, subtitle, transform=ax.transAxes, ha="center", va="bottom", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, length=3)
    ax.set_facecolor(SURFACE)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

fig.text(0.5, 0.10, "value of the parameter", ha="center", color=TEXT_SECONDARY, fontsize=9)
fig.text(0.01, 0.02, "←  tighten the assumption", ha="left", color=TEXT, fontsize=10)
fig.text(0.99, 0.02, "relax the assumption  →", ha="right", color=TEXT, fontsize=10)
fig.subplots_adjust(left=0.02, right=0.98, top=0.76, bottom=0.24, wspace=0.12)

out = Path(__file__).with_name("prior_strength.svg")
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
