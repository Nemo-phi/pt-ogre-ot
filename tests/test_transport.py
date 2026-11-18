import numpy as np

from otlib.transport import solve_dual, solve_transport


def test_known_transport_plan():
    a = [0.6, 0.4]
    b = [0.5, 0.5]
    cost = [[1, 4], [3, 2]]

    plan = solve_transport(a, b, cost)
    f, g = solve_dual(a, b, cost)
    slack = cost - np.add.outer(f, g)

    assert np.allclose(plan, [[0.5, 0.1], [0, 0.4]])
    assert np.isclose(np.sum(plan * cost), 1.7)
    assert np.isclose(np.dot(a, f) + np.dot(b, g), 1.7)
    assert np.all(slack >= -1e-9)
    assert np.allclose(plan * slack, 0)


def test_transport_route_count():
    a = [0.4, 0.6]
    b = [0.2, 0.3, 0.5]
    cost = [[1, 2, 4], [4, 2, 1]]

    plan = solve_transport(a, b, cost)
    assert np.allclose(plan.sum(axis=1), a)
    assert np.allclose(plan.sum(axis=0), b)
    assert np.count_nonzero(plan > 1e-9) <= len(a) + len(b) - 1
