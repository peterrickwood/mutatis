"""Shock handling, the posterior problem (contract 2), samplers, bounds and checks.

Sees a solved model only through contract 1.
"""

from mutatis import _jax_config  # noqa: F401  (64-bit floats before any JAX use)

from mutatis.infer.problem import PosteriorProblem, ProblemError, build_problem, choose_route
