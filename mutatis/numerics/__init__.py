"""Numerical methods on plain arrays: root finding, the linear forward-looking solve, the Kalman filter and smoother, rotations.

Arrays in, arrays out. Imports nothing else from mutatis except the JAX settings.
"""

from mutatis import _jax_config  # noqa: F401  (64-bit floats before any JAX use)
