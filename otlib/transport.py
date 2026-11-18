import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def _constraints(n, m):
    matrix = lil_matrix((n + m, n * m))

    for i in range(n):
        for j in range(m):
            route = i * m + j
            matrix[i, route] = 1
            matrix[n + j, route] = 1

    return matrix.tocsr()


def solve_transport(a, b, cost):
    cost = np.asarray(cost)
    n, m = cost.shape
    totals = np.concatenate((a, b))

    result = linprog(cost.ravel(), A_eq=_constraints(n, m), b_eq=totals,
                     bounds=(0, None))

    if not result.success:
        raise RuntimeError(f"transport LP failed: {result.message}")

    return result.x.reshape(n, m)


def solve_dual(a, b, cost):
    cost = np.asarray(cost)
    n, m = cost.shape
    weights = np.concatenate((a, b), dtype=float)

    result = linprog(-weights, A_ub=_constraints(n, m).T, b_ub=cost.ravel(),
                     bounds=(None, None))

    if not result.success:
        raise RuntimeError(f"dual LP failed: {result.message}")

    return result.x[:n], result.x[n:]
