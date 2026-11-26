"""Independent reference implementations, used only by tests. They use SciPy and NumPy, never mutatis code."""

import numpy as np
import scipy.linalg


def roots_of_quadratic(A, B, C):
    """All 2n roots of det(lambda^2 A + lambda B + C) = 0, from SciPy's generalised eigenvalue routine (LAPACK QZ).

    Uses the first-order form F z(t+1) = G z(t) with z(t) = [x(t-1), x(t)]. Roots at infinity come back as inf.
    """
    A, B, C = map(np.asarray, (A, B, C))
    n = A.shape[0]
    F = np.block([[np.zeros((n, n)), A], [np.eye(n), np.zeros((n, n))]])
    G = np.block([[-C, -B], [np.zeros((n, n)), np.eye(n)]])
    values, vectors = scipy.linalg.eig(G, F)
    return values, vectors


def stable_solution(A, B, C, D):
    """P and Q for A E x(t+1) + B x(t) + C x(t-1) + D e(t) = 0, built from the eigenvectors of the n roots inside the unit circle.

    Needs exactly n such roots and P diagonalisable; raises otherwise.
    """
    A, B, C, D = map(np.asarray, (A, B, C, D))
    n = A.shape[0]
    values, vectors = roots_of_quadratic(A, B, C)
    inside = np.isfinite(values) & (np.abs(values) < 1.0)
    if inside.sum() != n:
        raise ValueError(f"{inside.sum()} roots inside the unit circle, need {n}")
    W, lam = vectors[:n, inside], values[inside]
    P = np.real_if_close(W @ np.diag(lam) @ np.linalg.inv(W), tol=1e6)
    Q = -np.linalg.solve(A @ P + B, D)
    return np.real(P), np.real(Q)


def model_with_known_solution(n, inside, outside, rng, k=2):
    """A random model whose unique non-exploding solution is known.

    Writing the matrix polynomial as (lambda A + M)(lambda I - P) makes its roots the eigenvalues of P together with those of
    S = -A^-1 M. Choosing P with eigenvalues `inside` and S with eigenvalues `outside` fixes which roots lie where.
    """
    def with_eigenvalues(eigenvalues):
        V = rng.normal(size=(n, n))
        return V @ np.diag(eigenvalues) @ np.linalg.inv(V)

    P = with_eigenvalues(inside)
    S = with_eigenvalues(outside)
    A = rng.normal(size=(n, n)) + n * np.eye(n)
    B = -A @ (S + P)
    C = A @ S @ P
    D = rng.normal(size=(n, k))
    return A, B, C, D, P


def stacked_gaussian(m0, P0, T, W, Z, H, c, d):
    """Mean and covariance of all states s(1..t) and observations y(1..t) together, written out as one joint normal distribution.

    The model is s(t) = c(t) + T s(t-1) + w(t), y(t) = d(t) + Z s(t) + v(t), s(0) ~ N(m0, P0). No recursion over data is used, so
    this is independent of the Kalman filter. Returns (mean_s, mean_y, cov_ss, cov_sy, cov_yy), each flattened period by period.
    """
    m0, P0, T, W, Z, H, c, d = map(np.asarray, (m0, P0, T, W, Z, H, c, d))
    t, n = c.shape
    means, variances = [], []
    mean, variance = m0, P0
    for s in range(t):
        mean, variance = c[s] + T @ mean, T @ variance @ T.T + W
        means.append(mean)
        variances.append(variance)
    cov_ss = np.zeros((t * n, t * n))
    for a in range(t):
        for b in range(a + 1):  # Cov(s_a, s_b) = T^(a-b) Var(s_b) for a >= b
            block = np.linalg.matrix_power(T, a - b) @ variances[b]
            cov_ss[a * n:(a + 1) * n, b * n:(b + 1) * n] = block
            cov_ss[b * n:(b + 1) * n, a * n:(a + 1) * n] = block.T
    big_Z = np.kron(np.eye(t), Z)
    cov_sy = cov_ss @ big_Z.T
    cov_yy = big_Z @ cov_ss @ big_Z.T + np.kron(np.eye(t), H)
    mean_s = np.concatenate(means)
    return mean_s, big_Z @ mean_s + d.ravel(), cov_ss, cov_sy, cov_yy


def condition(mean_s, mean_y, cov_ss, cov_sy, cov_yy, y, use):
    """Mean and covariance of the states given the observations y at the flattened positions `use`."""
    y = np.asarray(y).ravel()
    gain = cov_sy[:, use] @ np.linalg.inv(cov_yy[np.ix_(use, use)])
    return mean_s + gain @ (y[use] - mean_y[use]), cov_ss - gain @ cov_sy[:, use].T
