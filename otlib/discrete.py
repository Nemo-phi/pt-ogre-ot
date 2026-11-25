import numpy as np
from scipy import sparse
from scipy.optimize import linprog

def solve_kantorovich(a, b, C):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Marginals must be vectors.")

    n, m = len(a), len(b)
    if n == 0 or m == 0 or C.shape != (n, m):
        raise ValueError("Invalid input shapes.")

    if not all(np.all(np.isfinite(z)) for z in (a, b, C)):
        raise ValueError("Inputs must be finite.")

    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Marginals must be nonnegative.")

    if a.sum() <= 0 or not np.isclose(a.sum(), b.sum()):
        raise ValueError("Marginals must have equal positive mass.")

    A_row = sparse.kron(
        sparse.eye(n, format="csr"),
        np.ones((1, m)),
        format ="csr",
    )

    A_col = sparse.kron(
        np.ones((1, n)),
        sparse.eye(m, format="csr"),
        format ="csr",
    )

    A_eq = sparse.vstack([A_row, A_col], format="csr")
    b_eq = np.concatenate([a,b])

    result = linprog(
        C.ravel(),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs-ds",
    )

    if not result.success:
        raise RuntimeError(result.message)

    P = result.x.reshape(n, m)
    cost = result.fun

    f = result.eqlin.marginals[:n]
    g = result.eqlin.marginals[n:]

    return P, cost, f, g
