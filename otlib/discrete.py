from scipy.optimize import linprog
import numpy as np


# Numerical solver

def otdiscrete_primal(a:np.array, b:np.array, C:np.array):
    """
    Primal resolution of the Kantorovich formulation using linear programming

    Input:
    - a,b: mass distribution
    - C: cost matrix

    Output:
    - P : optimal transport plan
    - obj: cost of the optimal tranport
    """
    # Initial problem
    # min <C,P> on P >= 0 in R^{n x m} s.t. P.1 = a and P^T.1 = b
    n, m = C.shape
    
    # Reformulation
    # min c^T.p on p in R+^{nm} s.t. Ap = [a^T, b^T]^T
    A = np.zeros((n+m, n*m), dtype=float)
    A[:n,:] = np.kron(np.ones(m), np.identity(n))
    A[n:,:] = np.kron(np.identity(m), np.ones(n))
    c = C.flatten('F') 

    # Solver
    res = linprog(c, A_eq=A, b_eq= np.concatenate([a,b]))
    if not res.success:
        raise RuntimeError(f"transport LP failed: {res.message}")

    p = res.x
    P = p.reshape((n,m), order='F')
    obj = res.fun

    return P, obj


def otdiscrete_dual(a:np.array, b:np.array, C:np.array):
    """
    Dual resolution of the Kantorovich formulation using linear programming
    
    Input:
    - a,b: mast distribution
    - C: cost matrix

    Output:
    - f,g: Kantorovich potentials
    - obj: price of the optimal potentials     
    """
    # Initial problem
    # max <f,a> + <g,b> on f in R^{n}, g in R^{m} such that f.1^T + 1.g^T <= C
    n, m = C.shape
    # Reformulation
    # max (a,b).h on h in R^{n+m} such that A^Th <= c
    obj_func = np.concatenate([a,b])
    A = np.zeros((n+m, n*m), dtype=float)
    A[0:n,:] = np.kron(np.ones(m), np.identity(n))
    A[n:,:] = np.kron(np.identity(m), np.ones(n))
    c = C.flatten('F')

    # Solver
    res = linprog(-obj_func, A_ub=A.T, b_ub=c, bounds=(None, None))
    f = res.x[:n]
    g = res.x[n:]
    obj = res.fun

    return f, g, -obj




# Wasserstein and quantile function

def discrete_quantile(a, x, r):
    cdf = np.cumsum(a)
    i = np.searchsorted(cdf, r, side='left')
    return x[i]


def Wp_discrete(a,x,b,y,p):
    """
    a,b: histograms
    x (resp. y): support of a (resp. b) in order i.e x[i] < x[i+1]
    """
    n = a.size
    m = b.size
    F_a = 0
    F_b = 0
    i = 0
    j = 0
    I = 0
    while (i < n) and (j < m):
        if F_a + a[i] < F_b + b[j]:
            F_a += a[i]
            I += np.abs(x[i]-y[j])**p*np.abs(F_a-F_b)
            i += 1
        elif F_a + a[i] > F_b + b[j]:
            F_b += b[j]
            I += np.abs(x[i]-y[j])**p*np.abs(F_a-F_b)
            j += 1
        else: #equality
            F_a += a[i]
            I += np.abs(x[i]-y[j])**p*np.abs(F_a-F_b)
            i += 1
            F_b += b[j]
            j+=1    
    return I**(1/p)


def Wp_continuous(C_inv_alpha, C_inv_beta, p, N):
    """
    C_inv_(alpha/beta) are the quantile function
    Number of point use in the middle point method (integrale approximation)
    """    
    h = 1 / N
    x = np.linspace(h/2,1-h/2,N)
    I = 0
    for point in x:
        I += (np.abs(C_inv_alpha(point)-C_inv_beta(point))**p)*h
    return I**(1/p)


def Gaussian_W2(m_a, sigma_a, m_b, sigma_b):
    return np.sqrt((m_a-m_b)**2+(sigma_a-sigma_b)**2)

