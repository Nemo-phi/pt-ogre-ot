import numpy as np
from scipy.optimize import linprog

from otlib import discrete, sinkhorn, jko, crowd


def test_package_imports():
    assert discrete is not None
    assert sinkhorn is not None
    assert jko is not None
    assert crowd is not None


def test_linear_programming():
    result = linprog(
        c=[1.0, 2.0],
        A_eq=[[1.0, 1.0]],
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
    )

    assert result.success
    assert np.isclose(result.fun, 1.0)
