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


def qz_solution(A, B, C, D, margin):
    """P and Q from SciPy's ordered QZ decomposition (Klein's method), and the number of roots inside the unit circle.

    The pencil of the first-order form is reordered so the roots smaller than 1 + `margin` in size come first; the first n
    columns of the right Schur vectors then span the non-exploding solutions, z = [x(t-1), x(t)], so x(t) = Z21 Z11^-1 x(t-1).
    P and Q are None unless exactly n roots are inside.
    """
    A, B, C, D = map(np.asarray, (A, B, C, D))
    n = A.shape[0]
    F = np.block([[np.zeros((n, n)), A], [np.eye(n), np.zeros((n, n))]])
    G = np.block([[-C, -B], [np.zeros((n, n)), np.eye(n)]])
    inside = lambda alpha, beta: np.abs(alpha) < (1 + margin) * np.abs(beta)
    _, _, alpha, beta, _, Z = scipy.linalg.ordqz(G, F, sort=inside, output="complex")
    count = int(np.sum(inside(alpha, beta)))
    if count != n:
        return None, None, count
    P = np.real(Z[n:, :n] @ np.linalg.inv(Z[:n, :n]))
    return P, -np.linalg.solve(A @ P + B, D), count


def structured_model(rng, inside, outside_inverse, k=2):
    """A random model (A, B, C, D) whose non-exploding solution P is known, allowing static equations and unlagged variables.

    The matrix polynomial is built as (lambda A + M)(lambda I - P), so its roots are the eigenvalues of P (`inside`; zeros are
    variables that do not appear lagged) and -1/mu for the eigenvalues mu of M^-1 A (`outside_inverse`; zeros make A singular,
    which puts roots at infinity, as static equations do). Complex values in `inside` must come in adjacent conjugate pairs.
    """
    n = len(inside)

    def with_eigenvalues(values):
        J, i = np.zeros((n, n)), 0
        while i < n:
            if np.imag(values[i]) != 0:  # a conjugate pair becomes a real 2 x 2 rotation-and-scaling block
                a, b = np.real(values[i]), np.imag(values[i])
                J[i:i + 2, i:i + 2] = [[a, -b], [b, a]]
                i += 2
            else:
                J[i, i] = np.real(values[i])
                i += 1
        V = rng.normal(size=(n, n))
        return V @ J @ np.linalg.inv(V)

    P = with_eigenvalues(np.asarray(inside, dtype=complex))
    M = rng.normal(size=(n, n)) + np.sqrt(n) * np.eye(n)
    A = M @ with_eigenvalues(np.asarray(outside_inverse, dtype=complex))
    return A, M - A @ P, -M @ P, rng.normal(size=(n, k)), P


def sylvester_matrix(M, A, P):
    """The n^2 x n^2 matrix of the map X -> M X + A X P, acting on X flattened row by row."""
    n = M.shape[0]
    return np.kron(M, np.eye(n)) + np.kron(A, P.T)


def series_amplification(M, A, P):
    """The sum over j of |F^j| |P^j| with F = -M^-1 A: how much summing X = sum_j F^j (M^-1 R) P^j can magnify rounding errors.

    Frobenius norms; the sum stops once a term is below 1e-12 of the total, which is ample for an order of magnitude.
    """
    F = -np.linalg.solve(M, A)
    total, Fj, Pj = 0.0, np.eye(len(M)), np.eye(len(M))
    while True:
        term = np.linalg.norm(Fj) * np.linalg.norm(Pj)
        total += term
        if term < 1e-12 * total:
            return total
        Fj, Pj = Fj @ F, Pj @ P


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
