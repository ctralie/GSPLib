"""
Reference: 
[1] Edelman, Alan, Tomas A. Arias, and Steven T. Smith. "The geometry of algorithms with orthogonality constraints." SIAM journal on Matrix Analysis and Applications 20.2 (1998)
"""
import numpy as np
import matplotlib.pyplot as plt

def getSubspaceAngle(X, Y):
    """
    Return the subspace angles between subspaces X and Y
    :param X: An orthonormal Nxp basis for the first subspace
    :param Y: An orthonormal Nxq basis for the second subspace
    :returns: Arc length along Grassmanian [1]
    """
    D = (X.T).dot(Y)
    U, S, V = np.linalg.svd(D)
    S[S > 1] = 1
    thetas = np.arccos(S)
    return np.sqrt(np.sum(thetas**2))

def getRandomSubspace(N, d):
    U, S, V = np.linalg.svd(np.random.randn(N, d))
    return U[:, 0:d]

if __name__ == '__main__':
    trials = []
    for i in range(100000):
        X = getRandomSubspace(5, 3)
        Y = getRandomSubspace(5, 3)
        trials.append(getSubspaceAngle(X, Y))
    plt.hist(trials)
    plt.show()
