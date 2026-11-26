"""The names a specification may use for distributions, domains and expectation schemes. One list, shared by every layer."""

DISTRIBUTIONS = frozenset({
    "normal", "half_normal", "gamma", "beta", "inverse_gamma", "student_t", "uniform", "multivariate_normal", "inverse_wishart",
    "flat",       # equal weight everywhere: not a probability distribution (README, "Flat" needs care)
    "jeffreys",   # the conventional uninformative prior for a covariance matrix; also not a probability distribution
})
NO_PRIOR = "none"
DOMAINS = frozenset({"real", "positive", "unit_interval", "positive_definite", "orthogonal"})
EXPECTATION_SCHEMES = frozenset({"rational", "naive", "discounted"})
