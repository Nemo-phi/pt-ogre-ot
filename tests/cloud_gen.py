import numpy as np
from scipy.spatial.distance import cdist

def point_clouds(n, m, seed):
    rng = np.random.default_rng(seed)
    sources = rng.normal([-0.8, 0], 0.55, size=(n, 2))
    destinations = rng.normal([0.8, 0.2], 0.55, size=(m, 2))
    return sources, destinations

def dirichlet_cloud_gen(n,m,seed=1701):
    x, y = point_clouds(n,m,seed)
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    cost = cdist(x,y, metric="sqeuclidean")

    return a,x,b,y,cost

def uniform_cloud_gen(n,m,seed=1):
    x, y = point_clouds(n,m,seed)
    a = np.ones(n)/n
    b = np.ones(m)/m
    cost = cdist(x,y,metric="sqeuclidean")

    return a,x,b,y,cost