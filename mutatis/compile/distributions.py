"""From distribution text in a specification to distribution objects, whose arguments may be parameters of the model.

`normal(0, sigma_eta)` becomes a `DistributionSpec` that, given parameter values, builds `Normal(0, value of sigma_eta)`.
A scalar family applied to something with a shape (a vector of shocks, a matrix of coefficients) acts on each element.
Arguments may be given by position or by name; gamma, beta and inverse_gamma also accept `mean=` and `sd=`.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from mutatis import _jax_config  # noqa: F401
from mutatis.compile.expand import CompileError, Scope, resolve
from mutatis.compile.symbols import PARAMETER, Ref
from mutatis.dist import (Beta, Flat, Gamma, HalfNormal, InverseGamma, InverseWishart, MultivariateNormal, Normal, StudentT, Uniform,
                          UniformOrthogonal)
from mutatis.dist.distributions import JeffreysCovariance
from mutatis.dist.domains import Interval, Orthogonal, Positive, PositiveDefinite, Real
from mutatis.spec.expr import Call, Expr, Name, Num, parse_expression
from mutatis.spec.vocabulary import NO_PRIOR

DOMAINS = {"real": Real(), "positive": Positive(), "unit_interval": Interval(0.0, 1.0), "positive_definite": PositiveDefinite(), "orthogonal": Orthogonal()}

# family: (positional argument names, the domain of a value drawn from it, whether it describes a whole vector or matrix)
FAMILIES = {
    "normal": (("loc", "scale"), Real(), False),
    "half_normal": (("scale",), Positive(), False),
    "gamma": (("a", "rate"), Positive(), False),
    "beta": (("a", "b"), Interval(0.0, 1.0), False),
    "inverse_gamma": (("a", "scale"), Positive(), False),
    "student_t": (("df", "loc", "scale"), Real(), False),
    "uniform": (("low", "high"), None, False),
    "multivariate_normal": (("mean", "cov"), Real(), True),
    "inverse_wishart": (("df", "scale"), PositiveDefinite(), True),
    "flat": ((), None, False),
    "jeffreys": ((), PositiveDefinite(), True),
}


@dataclass(frozen=True)
class DistributionSpec:
    family: str
    arguments: dict[str, Expr]
    shape: tuple[int, ...]  # shape of what it is a distribution of

    @property
    def proper(self) -> bool:
        return self.family not in ("flat", "jeffreys")

    @property
    def gaussian(self) -> bool:
        """Normal with mean zero, known before any parameter values: what the Kalman filter needs of shocks and measurement errors."""
        if self.family not in ("normal", "multivariate_normal"):
            return False
        location = self.arguments.get("loc", self.arguments.get("mean"))
        return location is None or (isinstance(location, Num) and float(location.value) == 0.0)

    def build(self, params: dict):
        args = {name: _value(expr, params, self.shape) for name, expr in self.arguments.items()}
        f = self.family
        if f == "normal":
            return Normal(args.get("loc", 0.0), args.get("scale", 1.0))
        if f == "half_normal":
            return HalfNormal(args.get("scale", 1.0))
        if f in ("gamma", "beta", "inverse_gamma"):
            cls = {"gamma": Gamma, "beta": Beta, "inverse_gamma": InverseGamma}[f]
            if "mean" in args:
                return cls.from_mean_sd(args["mean"], args["sd"])
            names = FAMILIES[f][0]
            return cls(args[names[0]], args[names[1]])
        if f == "student_t":
            return StudentT(args["df"], args.get("loc", 0.0), args.get("scale", 1.0))
        if f == "uniform":
            if not args:
                if len(self.shape) == 2 and self.shape[0] == self.shape[1]:
                    return UniformOrthogonal(self.shape[0])
                raise CompileError("uniform with no arguments means the even distribution over rotations, which needs a square matrix")
            return Uniform(args["low"], args["high"])
        n = self.shape[-1] if self.shape else 1
        if f == "multivariate_normal":
            mean = args.get("mean", 0.0)
            return MultivariateNormal(jnp.broadcast_to(mean, (n,)), args["cov"])
        if f == "inverse_wishart":
            return InverseWishart(args["df"], args["scale"])
        if f == "jeffreys":
            return JeffreysCovariance(n)
        if f == "flat":
            return Flat()
        raise CompileError(f"unknown distribution {f!r}")

    def covariance(self, params: dict):
        """The covariance of a zero-mean normal, over the flattened elements. Only for `gaussian` distributions."""
        dist = self.build(params)
        if isinstance(dist, MultivariateNormal):
            return dist.cov
        return jnp.diag(jnp.broadcast_to(jnp.asarray(dist.scale) ** 2, self.shape).reshape(-1))


def _value(expr: Expr, params: dict, shape):
    if isinstance(expr, Name) and expr.id == "identity":
        n = shape[-1] if shape else 1
        return jnp.eye(n)
    if isinstance(expr, Num):
        return jnp.asarray(float(expr.value))
    if isinstance(expr, Ref) and expr.kind == PARAMETER:
        return params[expr.key]
    from mutatis.compile.evaluate import Layout, Values, evaluate  # parameters only: no variables involved

    return evaluate(expr, Layout({}, {}, {}, {}, {}), Values(None, None, None, None, None, params))


def parse_distribution(text: str, parameter_keys: dict[str, str], shape: tuple[int, ...]) -> DistributionSpec | None:
    """`parameter_keys` maps the names usable in arguments to global parameter keys. Returns None for "none" (no prior)."""
    expr = parse_expression(text, shifts=False)
    if isinstance(expr, Name):
        if expr.id == NO_PRIOR:
            return None
        expr = Call(expr.id)
    if not isinstance(expr, Call) or expr.func not in FAMILIES:
        raise CompileError(f"unknown distribution {text!r}")
    names = FAMILIES[expr.func][0]
    if len(expr.args) > len(names):
        raise CompileError(f"{expr.func} takes at most {len(names)} positional arguments: {', '.join(names)}")
    arguments = dict(zip(names, expr.args)) | dict(expr.kwargs)

    def link(e: Expr) -> Expr:
        if isinstance(e, Name) and e.id == "identity":
            return e
        if isinstance(e, Name) and e.id not in parameter_keys:
            raise CompileError(f"{text!r}: {e.id!r} is not a parameter here")
        if isinstance(e, Name):
            return Ref(PARAMETER, parameter_keys[e.id])
        return resolve(e, Scope(parameter_keys, {}, set(), set(), {}))  # arguments may refer to parameters only

    return DistributionSpec(expr.func, {k: link(v) for k, v in arguments.items()}, tuple(shape))


def value_domain(spec_domain: str | None, distribution: DistributionSpec | None):
    """The domain of a parameter: the one stated in the specification, otherwise the one its prior implies, otherwise any real number."""
    if spec_domain is not None:
        return DOMAINS[spec_domain]
    if distribution is not None:
        implied = FAMILIES[distribution.family][1]
        if implied is not None:
            return implied
        if distribution.family == "uniform" and "low" in distribution.arguments:
            low, high = distribution.arguments["low"], distribution.arguments["high"]
            if isinstance(low, Num) and isinstance(high, Num):
                return Interval(float(low.value), float(high.value))
    return Real()
