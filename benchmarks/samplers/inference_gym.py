"""Held-out targets from Google's Inference Gym (Apache-2.0; github.com/tensorflow/probability, spinoffs/inference_gym).

They are in the long suite only, and the sampler's rules are never tuned on them: they check that what works on our own
targets also works on problems we did not choose. Each one is wrapped as contract 2, like the other targets:

- A Bayesian model (prior and likelihood given separately) keeps them separate, so that the ladder can temper the likelihood.
  The unknowns are the Inference Gym's own unconstrained ones: its bijector maps them to the model's values, and the prior
  includes the bijector's log Jacobian.
- A plain density (Banana, IllConditionedGaussian, NealsFunnel) is treated as our density targets are: a normal prior
  covering it, and the likelihood density / prior, so that the posterior is the density. These densities are normalised, so
  the true log evidence is 0; for the Bayesian models it is not known.

The reference is the Inference Gym's ground truth for its "identity" transformation: the mean and standard deviation of each
model value, from exact formulas or long runs (with their own standard errors, checked to be small in tests/benchmarks). The
quantities compared with it are that same transformation of each draw, because its order is not always the unknowns' order:
for a model with named parts it lists the parts by name, alphabetically (Brownian motion with unknown scales: innovation
scale, locations, observation scale; the unknowns: innovation scale, observation scale, locations).

Inference Gym models are written in 32-bit floats (except the log-Gaussian Cox process); the wrapper evaluates them at
32-bit precision and returns 64-bit values. This needs the `benchmarks` extra: `pip install -e ".[benchmarks]"` (TensorFlow
Probability's nightly build, because its last release does not work with JAX 0.12).
"""

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.samplers.targets import GYM, DensityTarget, Reference, Target

GROUP = "Inference Gym (held out)"


def _gym():
    try:
        from inference_gym import using_jax as gym
    except ImportError as error:
        raise ImportError("the Inference Gym targets need the benchmarks extra: pip install -e \".[benchmarks]\"") from error
    return gym


def _raw_key(key):
    """TensorFlow Probability's sampling wants a raw key (two 32-bit words), not a typed JAX key."""
    return jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key


class _Wrapped:
    """The parts shared by both kinds: the flat model, its bijector, its precision and its ground truth."""

    def _wrap(self, name: str):
        gym = _gym()
        class_name, self.description = GYM[name]
        self.name, self.group = name, GROUP
        self.model = getattr(gym.targets, class_name)()
        self.flat = gym.targets.VectorModel(self.model, flatten_sample_transformations=True)
        self.bijector = self.flat.default_event_space_bijector
        self.dtype = self.flat.dtype
        self.size = int(np.prod(self.flat.event_shape))
        truth = self.identity = self.flat.sample_transformations["identity"]
        self.truth_mean = np.asarray(truth.ground_truth_mean, dtype=float)
        self.truth_sd = np.asarray(truth.ground_truth_standard_deviation, dtype=float)
        se = getattr(truth, "ground_truth_mean_standard_error", None)
        self.truth_mean_se = None if se is None else np.asarray(se, dtype=float)

    def _x(self, z):
        """Model values from the unconstrained unknowns, at the model's precision; and the log Jacobian of that map."""
        z = jnp.asarray(z).astype(self.dtype)
        return self.bijector(z), self.bijector.forward_log_det_jacobian(z)

    def quantities(self, z: np.ndarray) -> np.ndarray:
        """The model values the ground truth describes, in its order (see the module's description)."""
        return np.asarray(jax.vmap(lambda u: self.identity(self._x(u)[0]))(jnp.asarray(z)), dtype=float)

    def _reference(self, log_evidence: float, source: str) -> Reference:
        names = tuple(f"x[{i}]" for i in range(self.size))
        return Reference(names, self.truth_mean, self.truth_sd, (None,) * self.size, log_evidence, source)


class GymBayesian(_Wrapped, Target):
    def __init__(self, name: str):
        self._wrap(name)
        self.reference = self._reference(float("nan"), "Inference Gym ground truth (evidence unknown)")

    def log_prior(self, z):
        x, log_jacobian = self._x(z)
        prior = self.model.prior_distribution().log_prob(self.flat.vector_event_to_structured(x))
        return (prior + log_jacobian).astype(jnp.float64)

    def log_likelihood(self, z):
        x, _ = self._x(z)
        return jnp.asarray(self.model.log_likelihood(self.flat.vector_event_to_structured(x))).astype(jnp.float64)

    def draw_prior(self, key):
        x = self.flat.structured_event_to_vector(self.model.prior_distribution().sample(seed=_raw_key(key)))
        return jnp.asarray(self.bijector.inverse(x)).astype(jnp.float64)


class GymDensity(_Wrapped, DensityTarget):
    c = 0.0

    def __init__(self, name: str):
        self._wrap(name)
        self._set_prior(self.truth_mean, self.truth_sd)
        self.reference = self._reference(0.0, "Inference Gym ground truth; normalised density, so log evidence 0")

    def log_density(self, z):
        x, log_jacobian = self._x(z)
        return (self.flat.unnormalized_log_prob(x) + log_jacobian).astype(jnp.float64)


def build(name: str) -> Target:
    gym = _gym()
    model_class = getattr(gym.targets, GYM[name][0])
    return GymBayesian(name) if issubclass(model_class, gym.targets.BayesianModel) else GymDensity(name)
