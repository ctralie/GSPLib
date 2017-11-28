import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from CSMSSMTools import *
import time

def getContinuousRuns(x):
    """
    Given a 1D array, find all of the continuous runs
    of 1s, and count the numbers with a certain length
    :param x: 1D array
    :returns Runs: A frequency distribution dictionary of
        the form {length:counts}
    """
    Runs = {}
    (IN_ZEROS, IN_ONES) = (0, 1)
    state = IN_ZEROS
    count = 0
    for i in range(len(x)):
        if state == IN_ZEROS:
            if x[i] == 1:
                state = IN_ONES
                count = 1
        elif state == IN_ONES:
            if x[i] == 0:
                if not count in Runs:
                    Runs[count] = 0
                Runs[count] += 1
                state = IN_ZEROS
            elif x[i] == 1:
                count += 1
    return Runs

def getRQAVerts(R):
    """
    Return a frequency distribution of the vertical 1s
    in an RQA plot
    :param R: An RQA Plot
    :returns Runs: A frequency distribution dictionary of
        the form {length:counts}
    """
    N = R.shape[0]
    D = np.zeros((N+1, N+1))
    D[0:N, 0:N] = R
    return getContinuousRuns(D.flatten())

def getRQADiags(R):
    """
    Return a frequency distribution of the diagonal 1s
    in an RQA plot
    :param R: An RQA Plot
    :returns Runs: A frequency distribution dictionary of
        the form {length:counts}
    """
    x = []
    for i in range(1, R.shape[1]):
        x += np.diagonal(R, i).tolist() + [0]
    return getContinuousRuns(x)

def zeroDenom(x):
    if x == 0:
        return 1.0
    return x

def getRQAStats(R, dmin, vmin):
    """
    Compute different recurrence quantification statistics.
    See http://www.recurrence-plot.tk/ for more details
    :param R: NxN RQA plot (binary matrix)
    :param dmin: Minimum diagonal length
    :param vmin: Minimum vertical length
    :returns: Dictionary of features
    """
    Verts = getRQAVerts(R)
    Diags = getRQADiags(R)
    N = R.shape[0]
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))

    #Expected value of diagonal lengths above a certain length
    SumDiags = np.sum(np.array([c*Diags[c] for c in Diags if c >= dmin]))
    SumDiags = float(SumDiags)
    #Expected value of vertical lengths above a certain length
    SumVerts = np.sum(np.array([c*Verts[c] for c in Verts if c >= vmin]))
    SumVerts = float(SumVerts)

    #Recurrence rate
    RR = np.sum(R == 1)/float(R.size)

    #Determinism
    DET = SumDiags/zeroDenom(float(np.sum(R[I > J])))

    #Laminarity
    denom = np.sum(np.array([c*Verts[c] for c in Verts]))
    LAM = SumVerts/zeroDenom(denom)

    #Ratio
    denom = (np.sum(np.array([Diags[c] for c in Diags])))**2
    RATIO = float(R.size)*SumDiags/zeroDenom(denom)

    #Average diag length
    NDiags = np.sum(np.array([Diags[c] for c in Diags if c >= dmin]))
    NDiags = zeroDenom(float(NDiags))
    L = SumDiags / NDiags

    #Trapping Time (Average vertical length)
    NVerts = np.sum(np.array([Verts[c] for c in Verts if c >= vmin]))
    NVerts = zeroDenom(float(NVerts))
    TT = SumVerts / NVerts

    #Entropy
    ps = np.array([Diags[c]/NDiags for c in Diags if c >= dmin])
    ENTR = -np.sum(ps*np.log(ps))

    #Longest diag line
    Lmax = 0
    if len(Diags) > 0:
        Lmax = np.max(np.array([c for c in Diags]))

    #Longest vert line
    Vmax = 0
    if len(Verts) > 0:
        Vmax = np.max(np.array([c for c in Verts]))

    return {'RR':RR, 'DET':DET, 'LAM':LAM, 'RATIO':RATIO, 'L':L, 'TT':TT, 'ENTR':ENTR, 'Lmax':Lmax, 'Vmax':Vmax}

if __name__ == '__main__':
    t = np.linspace(0, 2*np.pi, 200)
    X = np.cos(4*t)
    D = np.abs(X[:, None] - X[None, :])
    R = CSMToBinaryMutual(D, 0.2)
    stats = getRQAStats(R, 5, 5)
    for s in stats:
        print("%s: %.3g"%(s, stats[s]))
    plt.imshow(R)
    plt.show()
