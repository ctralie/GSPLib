import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from CSMSSMTools import getCSM

def getGreedyPerm(X, M):
    """
    Purpose: Naive O(NM) algorithm to do the greedy permutation
    :param X: Nxd array of Euclidean points
    :param M: Number of points in returned permutation
    :returns: (permutation (N-length array of indices), \
            lambdas (N-length array of insertion radii))
    """
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = getCSM(X[0, :][None, :], X).flatten()
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getCSM(X[idx, :][None, :], X).flatten())
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas}

if __name__ == '__main__':
    t = np.linspace(0, 10*np.pi, 301)[0:300]
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    np.random.seed(420)
    X = X + 0.1*np.random.randn(X.shape[0], 2)
    Y = getGreedyPerm(X, 50)['Y']
    
    plt.scatter(X[:, 0], X[:, 1], 40, edgecolor = 'none')
    plt.plot(Y[:, 0], Y[:, 1], 'rx')
    plt.show()
