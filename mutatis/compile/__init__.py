"""Turns a specification into an equation system. Applies the expectation scheme and cuts blocks out for estimation on their own.
"""

from mutatis import _jax_config  # noqa: F401  (64-bit floats before any JAX use)
