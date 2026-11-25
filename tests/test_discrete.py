import numpy as np
from scipy.optimize import linear_sum_assignment

from otlib.discrete import solve_kantorovich


def test_primal_feasibility():
    rng = np.random.default_rng(42)

    n, m = 8, 6
    a = rng.random(n)
    b = rng.random(m)
    a /= a.sum()
    b /= b.sum()

    C = rng.random((n, m))

    P, cost, f, g = solve_kantorovich(a, b, C)

    assert P.shape == (n, m)
    assert np.all(P >= -1e-9)
    assert np.allclose(P.sum(axis=1), a, atol=1e-9)
    assert np.allclose(P.sum(axis=0), b, atol=1e-9)
    assert np.isclose(cost, np.sum(C * P))


def test_extreme_point_sparsity():
    rng = np.random.default_rng(123)

    n, m = 10, 7
    a = rng.random(n)
    b = rng.random(m)
    a /= a.sum()
    b /= b.sum()

    C = rng.random((n, m))

    P, _, _, _ = solve_kantorovich(a, b, C)

    nnz = np.count_nonzero(P > 1e-9)

    assert nnz <= n + m - 1


def test_duality_and_complementary_slackness():
    a = np.array([0.3, 0.7])
    b = np.array([0.4, 0.6])

    C = np.array([
        [0.0, 2.0],
        [1.0, 0.0],
    ])

    P, cost, f, g = solve_kantorovich(a, b, C)

    dual_cost = a @ f + b @ g
    slack = C - f[:, None] - g[None, :]

    # Dual feasibility
    assert np.all(slack >= -1e-9)

    # Strong duality
    assert np.isclose(cost, dual_cost, atol=1e-9)

    # Complementary slackness
    assert np.max(np.abs(P * slack)) < 1e-9


def test_birkhoff_theorem():
    rng = np.random.default_rng(2026)

    n = 6
    a = np.ones(n) / n
    b = np.ones(n) / n
    C = rng.random((n, n))

    P, cost, _, _ = solve_kantorovich(a, b, C)

    rows, cols = linear_sum_assignment(C)
    assignment_cost = C[rows, cols].sum() / n

    assert np.isclose(cost, assignment_cost, atol=1e-9)
