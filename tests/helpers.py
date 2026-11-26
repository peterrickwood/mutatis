"""Checks shared by tests."""

import numpy as np


def assert_mean_matches(draws, true_mean, standard_errors: float = 5.0):
    """The sample mean is within `standard_errors` Monte Carlo standard errors of the true mean, each estimated from the sample.

    For independent draws. With 5 standard errors a correct implementation fails about once in 1.7 million checks.
    """
    draws = np.asarray(draws)
    standard_error = draws.std(axis=0, ddof=1) / np.sqrt(draws.shape[0])
    np.testing.assert_array_less(np.abs(draws.mean(axis=0) - np.asarray(true_mean)), standard_errors * standard_error)


def assert_covariance_matches(draws, true_mean, true_cov, standard_errors: float = 5.0):
    """The covariance is the mean of products of deviations from the true mean, so it is checked the same way."""
    centred = np.asarray(draws) - np.asarray(true_mean)
    products = (centred[:, :, None] * centred[:, None, :]).reshape(len(centred), -1)
    assert_mean_matches(products, np.asarray(true_cov).ravel(), standard_errors)
