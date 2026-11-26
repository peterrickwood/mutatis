"""From a specification and data to a posterior problem: compile, arrange the data, and attach the solver.

This is the one place that knows every layer, so the layers below it need not know each other.
"""

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.system import System, compile_model
from mutatis.infer.problem import PosteriorProblem, build_problem
from mutatis.measure.data import Dataset, arrange
from mutatis.solve.linear import solve_linear
from mutatis.spec.model import Model


def solver(system: System):
    """The solver for a compiled system: a function from parameter values to a solved model."""
    return lambda params: solve_linear(system, params)


def prepare(model: Model, dataset: Dataset, route: str = "auto", start: str = "auto") -> PosteriorProblem:
    system = compile_model(model)
    return build_problem(system, solver(system), arrange(system, dataset, model.data), route, start)
