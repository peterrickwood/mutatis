"""The benchmark's known answers, each checked independently of how it was derived.

- Densities: each target's log density against an independent formula (SciPy), its exact sampler's draws against the stated
  marginal distributions (Kolmogorov-Smirnov at the 0.1% level, and the mean and sd within 5 Monte Carlo standard errors), and
  the evidence device (the prior times the likelihood integrates to exp(c)) by numerical integration in one and two dimensions.
- Conjugate regression: the closed-form evidence against the identity evidence = likelihood x prior / posterior at several
  points; the Jacobian against automatic differentiation; prior and exact posterior draws against the stated distributions.
- Grid integration: on targets with known answers.
- The model targets' references: grid convergence, and the prior integrating to one.

The checks that draw 200,000 samples or integrate on fine grids are in the medium tier; the rest are fast.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate as si
import scipy.stats as ss
from scipy.special import logsumexp

from benchmarks.samplers import targets as T
from benchmarks.samplers.quadrature import grid_posterior

N = 200_000
KS_CRITICAL = 1.95 / np.sqrt(N)   # Kolmogorov-Smirnov critical value at the 0.1% level (asymptotic), for single checks
DENSITY_TARGETS = ["correlated_normal_d10_c100", "correlated_normal_d10_c10000", "scales_d20", "two_peaks_d1", "two_peaks_d5",
                   "two_peaks_d20", "banana_d2_strong", "banana_d10_mild", "funnel_d10", "student_t_d10", "student_t_d50",
                   "correlated_normal_d50_c10000", "correlated_normal_d100_c10000", "two_peaks_d50", "banana_d50_mild", "funnel_d20",
                   "student_t_d100"]


def _independent_log_density(t, z):
    """The target's density written again with SciPy, from its definition."""
    if isinstance(t, T.Gaussian):
        if np.count_nonzero(t.cov - np.diag(np.diag(t.cov))) == 0:   # SciPy's own check calls a 1e10 spread of scales singular
            return np.sum(ss.norm.logpdf(z, t.m, np.sqrt(np.diag(t.cov))), axis=1)
        return ss.multivariate_normal(t.m, t.cov).logpdf(z)
    if isinstance(t, T.StudentT):
        return ss.multivariate_t(t.loc, t.scale, df=t.nu).logpdf(z)
    if isinstance(t, T.TwoPeaks):
        return logsumexp([np.log(w) + ss.multivariate_normal(c, s**2 * np.eye(t.size)).logpdf(z)
                          for w, s, c in zip(t.weights, t.sds, t.centres)], axis=0)
    if isinstance(t, T.Banana):
        x = z.copy()
        x[:, 1] = z[:, 1] - t.b * (z[:, 0] ** 2 - 100.0)
        return ss.norm.logpdf(x[:, 0], 0, 10) + np.sum(ss.norm.logpdf(x[:, 1:]), axis=1)
    if isinstance(t, T.Funnel):
        return ss.norm.logpdf(z[:, 0], 0, 3) + np.sum(ss.norm.logpdf(z[:, 1:], 0, np.exp(z[:, :1] / 2)), axis=1)
    raise TypeError(t)


@pytest.mark.parametrize("name", DENSITY_TARGETS)
def test_density_matches_an_independent_formula(name):
    t = T.build(name)
    z = t.exact_draws(np.random.default_rng(0), 50)
    ours = np.asarray(jax.vmap(t.log_density)(jnp.asarray(z)))
    np.testing.assert_allclose(ours, _independent_log_density(t, z), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("name", DENSITY_TARGETS)
@pytest.mark.medium
def test_exact_draws_follow_the_stated_answer(name):
    t = T.build(name)
    ref = t.reference
    q = t.quantities(t.exact_draws(np.random.default_rng(1), N))
    for j in range(q.shape[1]):
        x = q[:, j]
        m, s = x.mean(), x.std()
        se_mean, se_sd = s / np.sqrt(N), np.std((x - m) ** 2) / np.sqrt(N) / (2 * s)
        assert abs(m - ref.mean[j]) < 5 * se_mean, (ref.names[j], m, ref.mean[j], se_mean)
        assert abs(s - ref.sd[j]) < 5 * se_sd, (ref.names[j], s, ref.sd[j], se_sd)
        if ref.cdfs[j] is not None:   # at the 0.1% level divided by the number of quantities, as in tests/infer/test_calibration.py
            assert ss.kstest(x, ref.cdfs[j]).pvalue > 1e-3 / q.shape[1], ref.names[j]


def test_banana_twisted_coordinate_distribution_by_adaptive_quadrature():
    t = T.build("banana_d2_strong")
    for y in [-12.0, -9.0, -5.0, 0.0, 3.0, 10.0, 40.0, 120.0]:
        inner = lambda x2: ss.norm.pdf(x2) * (2 * ss.norm.cdf(np.sqrt(max((y - x2) / t.b + 100.0, 0.0)) / 10.0) - 1)
        kink = y + 100.0 * t.b
        exact = sum(si.quad(inner, a, c, epsabs=1e-13, limit=200)[0] for a, c in [(-12, min(kink, 12)), (min(kink, 12), 12)] if c > a)
        assert abs(t.reference.cdfs[1](y) - exact) < 2e-5, y


def test_evidence_device_integrates_to_exp_c_in_one_dimension():
    t = T.build("two_peaks_d1")
    f = jax.jit(lambda z: jnp.exp(t.log_prior(z) + t.log_likelihood(z)))
    total = sum(si.quad(lambda x: float(f(jnp.array([x]))), a, b, limit=400, epsabs=1e-14)[0]
                for a, b in [(-60, -8), (-8, -2), (-2, 2), (2, 8), (8, 60)])
    assert abs(np.log(total) - t.reference.log_evidence) < 1e-7


@pytest.mark.medium
def test_evidence_device_integrates_to_exp_c_in_two_dimensions():
    t = T.build("banana_d2_strong")
    x1 = np.linspace(-80, 80, 1601)
    x2 = np.linspace(-80, 1200, 12801)
    f = jax.jit(jax.vmap(lambda z: t.log_prior(z) + t.log_likelihood(z)))
    rows = [logsumexp(np.asarray(f(jnp.column_stack([np.full_like(x2, a), x2])))) for a in x1]
    log_total = logsumexp(rows) + np.log(x1[1] - x1[0]) + np.log(x2[1] - x2[0])
    assert abs(log_total - t.reference.log_evidence) < 1e-6


def test_normal_means_evidence_by_one_dimensional_integration():
    t = T.build("normal_means_d2")
    y = np.asarray(t.inner.y)
    total = 0.0
    for j in range(t.size):
        f = lambda m: np.exp(ss.norm.logpdf(m, 0, 10) + np.sum(ss.norm.logpdf(y[:, j], m, 1)) + 70.0)
        centre = y[:, j].mean()
        total += np.log(si.quad(f, centre - 3, centre + 3, epsabs=0, epsrel=1e-12)[0]) - 70.0
    assert abs(total - t.reference.log_evidence) < 1e-8


# --- conjugate regression ---


@pytest.mark.parametrize("name", ["conjugate_regression_n2_p1", "conjugate_regression_n4_p4", "conjugate_regression_n6_p2"])
def test_regression_evidence_identity(name):
    t = T.build(name)
    for k in jax.random.split(jax.random.key(0), 5):
        B, L = t.unpack(t.draw_posterior(k))
        value = t.log_likelihood_values(B, L) + t.log_prior_values(B, L) - t.log_posterior_values(B, L)
        assert abs(float(value) - t.reference.log_evidence) < 1e-7 * abs(t.reference.log_evidence)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_regression_jacobian(n):
    t = T.ConjugateRegression(n, 1, periods=20)
    z = jnp.asarray(np.random.default_rng(n).normal(size=t.size))
    w_index = slice(t.k * n, t.size)

    def vech_sigma(w):
        _, L = t.unpack(z.at[w_index].set(w))
        return (L @ L.T)[t.rows, t.cols]

    jac = jax.jacfwd(vech_sigma)(z[w_index])
    assert abs(float(jnp.linalg.slogdet(jac)[1]) - float(t.log_jacobian(z))) < 1e-10


def test_regression_pack_and_unpack():
    t = T.build("conjugate_regression_n4_p1")
    B = np.random.default_rng(0).normal(size=(t.k, t.n))
    sigma = np.cov(np.random.default_rng(1).normal(size=(t.n, 30))) + np.eye(t.n)
    Bz, L = t.unpack(jnp.asarray(t.pack(B, sigma)))
    np.testing.assert_allclose(Bz, B, rtol=1e-12)
    np.testing.assert_allclose(L @ L.T, sigma, rtol=1e-12)


@pytest.mark.medium
def test_regression_prior_draws_follow_the_prior():
    """Under the prior, each coefficient is Student-t with nu0 - n + 1 degrees of freedom, each variance inverse-gamma, and each
    covariance symmetric about zero (S0 is diagonal). The prior's variances have no finite variance themselves (nu0 = n + 3),
    so the checks are on distributions, not moments."""
    t = T.build("conjugate_regression_n2_p1")
    q = t.quantities(np.asarray(jax.vmap(t.draw_prior)(jax.random.split(jax.random.key(1), N))))
    dof = t.nu0 - t.n + 1
    for i in range(t.k):
        for j in range(t.n):
            scale = np.sqrt(t.Omega0[i, i] * t.S0[j, j] / dof)
            assert ss.kstest(q[:, i * t.n + j] / scale, "t", args=(dof,)).statistic < KS_CRITICAL, (i, j)
    for m, (r, c) in enumerate(zip(t.rows, t.cols)):
        x = q[:, t.k * t.n + m]
        if r == c:
            assert ss.kstest(x, ss.invgamma((t.nu0 - t.n + 1) / 2, scale=t.S0[r, r] / 2).cdf).statistic < KS_CRITICAL, (r, c)
        else:
            assert abs(np.mean(x > 0) - 0.5) < 5 * 0.5 / np.sqrt(N), (r, c)


@pytest.mark.medium
@pytest.mark.parametrize("name", ["conjugate_regression_n2_p1", "conjugate_regression_n4_p4", "conjugate_regression_n6_p2"])
def test_regression_exact_posterior_draws_follow_the_reference(name):
    t = T.build(name)
    draws = np.asarray(jax.jit(jax.vmap(t.draw_posterior))(jax.random.split(jax.random.key(2), N)))
    q = t.quantities(draws)
    ref = t.reference
    for j in range(q.shape[1]):
        x = q[:, j]
        m, s = x.mean(), x.std()
        assert abs(m - ref.mean[j]) < 5 * s / np.sqrt(N), ref.names[j]
        assert abs(s - ref.sd[j]) < 5 * np.std((x - m) ** 2) / np.sqrt(N) / (2 * s), ref.names[j]
        if ref.cdfs[j] is not None:
            assert ss.kstest(x, ref.cdfs[j]).pvalue > 1e-3 / q.shape[1], ref.names[j]


# --- grid integration ---


@pytest.mark.parametrize("d", [2, 3])
def test_grid_integration_on_a_normal(d):
    rng = np.random.default_rng(d)
    a = rng.normal(size=(d, d))
    t = T.Gaussian("g", "g", "g", rng.normal(size=d), a @ a.T + 0.5 * np.eye(d), -4.2, "exact")
    log_post = lambda z: t.log_prior(z) + t.log_likelihood(z)
    g = grid_posterior(log_post, lambda k, v: v, np.zeros((1, d)), 201 if d == 2 else 81)
    assert abs(g.log_evidence - (-4.2)) < 1e-8
    np.testing.assert_allclose(g.mean, t.m, atol=1e-8 * t.reference.sd.max())
    np.testing.assert_allclose(g.sd, t.reference.sd, rtol=1e-7)
    for k in range(d):
        x = t.m[k] + np.linspace(-3, 3, 13) * t.reference.sd[k]
        np.testing.assert_allclose(g.cdf(k, x), ss.norm.cdf(x, t.m[k], t.reference.sd[k]), atol=2e-3)


@pytest.mark.medium
def test_grid_integration_on_a_banana():
    t = T.build("banana_d2_strong")
    g = grid_posterior(lambda z: t.log_prior(z) + t.log_likelihood(z), lambda k, v: v, np.array([[0.0, 0.0], [5.0, 3.0]]), 801)
    assert abs(g.log_evidence - t.reference.log_evidence) < 1e-4
    assert np.all(np.abs(g.mean - t.reference.mean) < 2e-3 * t.reference.sd)
    np.testing.assert_allclose(g.sd, t.reference.sd, rtol=2e-3)


# --- the model targets' references ---


@pytest.mark.medium
@pytest.mark.parametrize("name, coarse", [("hidden_level_shocks_T50", 201), ("hidden_level_shocks_T200", 201), ("forward_looking", 61)])
def test_model_reference_has_converged(name, coarse):
    fine = T.build(name)
    rough = T.build(name)
    rough.grid_points = coarse
    a, b = fine.reference, rough.reference
    assert abs(a.log_evidence - b.log_evidence) < 1e-4
    assert np.all(np.abs(a.mean - b.mean) < 1e-3 * a.sd)
    np.testing.assert_allclose(a.sd, b.sd, rtol=1e-3)
    for j in range(len(a.names)):
        x = a.mean[j] + np.linspace(-3, 3, 25) * a.sd[j]
        assert np.max(np.abs(a.cdfs[j](x) - b.cdfs[j](x))) < 5e-3


@pytest.mark.medium
@pytest.mark.parametrize("name", ["hidden_level_shocks_T200", "forward_looking"])
def test_model_prior_integrates_to_one(name):
    """The priors of the parameters are independent, so the integral over all of them is a product of one-dimensional
    integrals of slices: prod_k (integral of exp(log prior) along axis k) / exp((d - 1) * log prior at the slices' crossing)."""
    kp = T.build(name).kalman_problem
    centre = np.asarray(kp.draw_prior(jax.random.key(0)))
    log_prior = jax.jit(jax.vmap(kp.log_prior))
    axis = np.linspace(-40.0, 40.0, 40001)
    total = -(kp.size - 1) * float(kp.log_prior(jnp.asarray(centre)))
    for k in range(kp.size):
        z = np.tile(centre, (len(axis), 1))
        z[:, k] = axis
        values = np.asarray(log_prior(jnp.asarray(z)))
        total += logsumexp(values) + np.log(axis[1] - axis[0])
    assert abs(total) < 1e-6
