import numpy as np
from math import dist

np.random.seed(0)

def hist_gaussian(m, sigma, N=1000, b=100):
    """
    Generate histogram (hist_p) of a gaussian of mean m and standart deviation sigma.
    Input: m: mean, sigma: standard deviation, N: number of sample, b: size of the histogram and x
    Output: hist_p, vals: hist_p[i] is the probability of the bin which as for mean vals[i]
    """
    s = np.random.normal(m, sigma, N)
    hist, edge_bin = np.histogram(s, bins=b)
    hist_p = hist / N
    vals = np.array([(edge_bin[i]+edge_bin[i+1])/2 for i in range(b)])
    return hist_p, vals


def gaussian_gen(m_a,sigma_a,m_b,sigma_b, N=1000, n=100):
    # Generation of histogram
    a,x = hist_gaussian(m_a,sigma_a,N,n)
    b,y = hist_gaussian(m_b,sigma_b,N,n)
    # Computation of the cost matrix
    C = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            C[i][j] = np.abs(x[i]-y[j])**2
    return a,x,b,y,C