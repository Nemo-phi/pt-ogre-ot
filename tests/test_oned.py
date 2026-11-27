import numpy as np

from otlib.discrete import Wp_discrete, otdiscrete_primal


def test_wasserstein_monotone():
    a = np.array([0.5, 0.5])
    x = np.array([0.0, 1.0])

    b = np.array([0.25, 0.75])
    y = np.array([0.0, 2.0])

    C = (x[:, None] - y[None, :]) ** 2

    P, cost = otdiscrete_primal(a, b, C)

    W2_quantile = Wp_discrete(a, x, b, y, 2)

    assert np.isclose(W2_quantile, np.sqrt(1.5))
    assert np.isclose(W2_quantile, np.sqrt(cost))

    assert np.allclose(P.sum(axis=1), a)
    assert np.allclose(P.sum(axis=0), b)
