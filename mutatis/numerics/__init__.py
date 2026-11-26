"""Numerical methods on plain arrays: root finding, the linear forward-looking solve, the Kalman filter and smoother, rotations.

Arrays in, arrays out. Imports nothing else from mutatis except the JAX settings.
"""

from mutatis import _jax_config  # noqa: F401  (64-bit floats before any JAX use)

from mutatis.numerics.kalman import Filtered, Smoothed, kalman_filter, rts_smoother, stationary_covariance
from mutatis.numerics.linear_re import LinearSolution, roots, solve_backward, solve_forward_looking
from mutatis.numerics.rootfind import find_root
from mutatis.numerics.simulate import simulate
from mutatis.numerics.status import Status
