"""Turns an equation system and parameter values into a solved model (contract 1).
"""

from mutatis import _jax_config  # noqa: F401  (64-bit floats before any JAX use)

from mutatis.solve.contract import GaussianNoise, LinearForm, SolvedModel
from mutatis.solve.linear import LinearSolved, SolveError, solve_linear
