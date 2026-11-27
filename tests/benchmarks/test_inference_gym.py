"""The Inference Gym targets, as wrapped for the benchmark (benchmarks/samplers/inference_gym.py). Needs the benchmarks extra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

pytest.importorskip("inference_gym", reason="the Inference Gym targets need the benchmarks extra")

from benchmarks.samplers.inference_gym import GymDensity  # noqa: E402
from benchmarks.samplers.targets import GYM, build  # noqa: E402  (through the registry, as the benchmark builds them)

# Built once per test session: the larger ones take minutes to build.
FAST = [name for name in GYM if name != "gym_stochastic_volatility"]
FLOAT32_RELATIVE = 1e-6   # the models evaluate in 32-bit floats (relative rounding 6e-8), summed over a few terms


@pytest.fixture(scope="module", params=FAST)
def target(request):
    return build(request.param)


def test_prior_and_likelihood_add_up_to_the_models_own_density(target):
    """At draws from the prior: the log prior is finite; the log likelihood is finite or minus infinity (32-bit overflow at
    extreme prior draws), never NaN; and log prior + log likelihood equals the model's own log density in the unconstrained
    unknowns, to 32-bit rounding."""
    z = jax.vmap(target.draw_prior)(jax.random.split(jax.random.key(0), 16))
    assert np.all(np.isfinite(np.asarray(z)))
    log_prior, log_likelihood = np.asarray(jax.vmap(target.log_prior)(z)), np.asarray(jax.vmap(target.log_likelihood)(z))
    assert np.all(np.isfinite(log_prior)) and not np.any(np.isnan(log_likelihood))
    x, log_jacobian = jax.vmap(target._x)(z)
    joint = np.asarray(jax.vmap(target.flat.unnormalized_log_prob)(x) + log_jacobian, dtype=float)
    ok = np.isfinite(log_likelihood)
    assert ok.sum() >= 8, "most prior draws should give a finite likelihood"
    np.testing.assert_allclose(log_prior[ok] + log_likelihood[ok], joint[ok], rtol=FLOAT32_RELATIVE, atol=1e-4)


def test_gradients_are_finite_where_the_density_is(target):
    z = jax.vmap(target.draw_prior)(jax.random.split(jax.random.key(1), 8))
    value_and_grad = jax.vmap(jax.value_and_grad(lambda u: target.log_prior(u) + target.log_likelihood(u)))
    value, grad = value_and_grad(z)
    finite = np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(grad)[finite]))


def test_ground_truth_is_precise_enough_for_the_accuracy_test(target):
    """The benchmark calls a mean accurate within 0.1 sd. The ground truth's own Monte Carlo error must be well below that:
    at most 0.02 sd, a fifth of the threshold (exact ground truths have no error)."""
    ref = target.reference
    assert ref.mean.shape == ref.sd.shape == (target.size,) and np.all(ref.sd > 0)
    if target.truth_mean_se is not None:
        assert np.max(target.truth_mean_se / target.truth_sd) <= 0.02
    assert target.quantities(np.zeros((3, target.size))).shape == (3, target.size)


def _joined(parts, names) -> np.ndarray:
    """Parts joined in the order of `names` (a dict, or a tuple with field names), or with no names in their own order."""
    if names is not None:
        parts = parts._asdict() if hasattr(parts, "_asdict") else parts
        parts = [parts[name] for name in names]
    return np.concatenate([np.ravel(np.asarray(p, dtype=float)) for p in jax.tree_util.tree_leaves(parts)])


def test_each_quantity_is_compared_with_the_ground_truth_of_the_same_model_value(target):
    """The ground truth describes the Inference Gym's "identity" transformation of a draw. Where the ground truth names its
    parts, join the parts of a draw and of the ground truth in the same order of names, using the model itself rather than
    the wrapper; position i must then be the same model value on both sides. (Keeping the unknowns' own order compared
    Brownian motion's last location with its observation scale.)"""
    identity = target.model.sample_transformations["identity"]
    truth = identity.ground_truth_mean
    names = sorted(truth) if isinstance(truth, dict) else None
    z = jax.random.normal(jax.random.key(3), (2, target.size))
    draws = np.stack([_joined(identity(target.flat.vector_event_to_structured(target._x(u)[0])), names) for u in z])
    np.testing.assert_array_equal(target.quantities(z), draws)
    # the flattened model keeps the ground truth at the model's precision (32 bits for all but the Cox process)
    np.testing.assert_array_equal(target.reference.mean, _joined(truth, names).astype(target.dtype))


def test_banana_density_is_normalised():
    """Its reference says log evidence 0, which holds only if the density integrates to one: check on a grid wide enough
    for the curved tail (x1 sd 10, x2 bent by 0.03 x1^2)."""
    target = build("gym_banana")
    a, b = np.linspace(-80, 80, 1601), np.linspace(-220, 40, 2601)
    grid = np.stack(np.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2)
    density = np.exp(np.asarray(jax.vmap(target.log_density)(jnp.asarray(grid))))
    assert abs(density.sum() * (a[1] - a[0]) * (b[1] - b[0]) - 1.0) < 1e-3


def test_funnel_density_is_neals_normalised_formula():
    """v ~ N(0, 3^2) and each x_i ~ N(0, exp(v / 2)^2): a product of normal densities, so normalised (log evidence 0)."""
    target = build("gym_neals_funnel")
    z = jax.random.normal(jax.random.key(2), (20, target.size)) * jnp.array([3.0] + [1.0] * (target.size - 1))
    formula = jax.vmap(lambda u: norm.logpdf(u[0], 0.0, 3.0) + jnp.sum(norm.logpdf(u[1:], 0.0, jnp.exp(u[0] / 2))))(z)
    np.testing.assert_allclose(np.asarray(jax.vmap(target.log_density)(z)), np.asarray(formula), rtol=FLOAT32_RELATIVE, atol=1e-4)


def test_ill_conditioned_gaussian_is_a_normalised_normal():
    """For a normal density, the Hessian H of the log density is constant, the peak is x + H^-1 grad at any x, and the log
    density at the peak is 0.5 log det(-H) - (d/2) log(2 pi) exactly when the density is normalised."""
    target = build("gym_ill_conditioned_gaussian")
    f = lambda u: target.log_density(u)
    x = jnp.zeros(target.size) + 0.1
    hessian = np.asarray(jax.hessian(f)(x), dtype=float)
    peak = np.asarray(x) - np.linalg.solve(hessian, np.asarray(jax.grad(f)(x)))
    sign, log_det = np.linalg.slogdet(-hessian)
    assert sign > 0
    expected = 0.5 * log_det - 0.5 * target.size * np.log(2 * np.pi)
    assert abs(float(f(jnp.asarray(peak))) - expected) < 1e-3 * max(1.0, abs(expected))


def test_density_targets_say_log_evidence_zero_and_bayesian_ones_unknown(target):
    if isinstance(target, GymDensity):
        assert target.reference.log_evidence == 0.0
    else:
        assert np.isnan(target.reference.log_evidence)
