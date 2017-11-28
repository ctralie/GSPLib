"""
Author: Chris Tralie
Description: Contains methods to plot and compare persistence diagrams
              Comparison algorithms include grabbing/sorting, persistence landscapes,
              and the "multiscale heat kernel" (CVPR 2015)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc #Used for downsampling rasterized images avoiding aliasing
import time #For timing kernel comparison
import sklearn.metrics.pairwise

##############################################################################
##########                  Plotting Functions                      ##########
##############################################################################

def plotDGM(dgm, color = 'b', sz = 20, label = 'dgm', axcolor = np.array([0.0, 0.0, 0.0]), marker = None):
    if dgm.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin
    a = max(axMin - axRange/5, 0)
    b = axMax+axRange/5
    # plot line
    plt.plot([a, b], [a, b], c = axcolor, label = 'none')
    plt.hold(True)
    # plot points
    if marker:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, marker, label=label, edgecolor = 'none')
    else:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, label=label, edgecolor = 'none')
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')
    return H

def plotWassersteinMatching(I1, I2, matchidx, color1 = 'r', color2 = 'b', marker1 = 'o', marker2 = 'x'):
    plotDGM(I1, color = color1, marker = marker1, sz=50)
    plt.hold(True)
    plotDGM(I2, color = color2, marker = marker2)
    cp = np.cos(np.pi/4)
    sp = np.sin(np.pi/4)
    R = np.array([[cp, -sp], [sp, cp]])
    if I1.size == 0:
        I1 = np.array([[0, 0]])
    if I2.size == 0:
        I2 = np.array([[0, 0]])
    I1Rot = I1.dot(R)
    I2Rot = I2.dot(R)
    for index in matchidx:
        (i, j) = index
        if i >= I1.shape[0] and j >= I2.shape[0]:
            continue
        if i >= I1.shape[0]:
            diagElem = np.array([I2Rot[j, 0], 0])
            diagElem = diagElem.dot(R.T)
            plt.plot([I2[j, 0], diagElem[0]], [I2[j, 1], diagElem[1]], 'g')
        elif j >= I2.shape[0]:
            diagElem = np.array([I1Rot[i, 0], 0])
            diagElem = diagElem.dot(R.T)
            plt.plot([I1[i, 0], diagElem[0]], [I1[i, 1], diagElem[1]], 'g')
        else:
            plt.plot([I1[i, 0], I2[j, 0]], [I1[i, 1], I2[j, 1]], 'g')


##############################################################################
##########            Diagram Comparison Functions                  ##########
##############################################################################

def getWassersteinDist(S, T):
    """
    Perform the Wasserstein distance matching between persistence diagrams.
    Assumes first two columns of S and T are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching)
    :param S: Mx(>=2) array of birth/death pairs for PD 1
    :param T: Nx(>=2) array of birth/death paris for PD 2
    :returns (tuples of matched indices, total cost, (N+M)x(N+M) cross-similarity)
    """
    import hungarian.hungarian as hungarian #Requires having compiled the library
    N = S.shape[0]
    M = T.shape[0]
    #Handle the cases where there are no points in the diagrams
    if N == 0:
        S = np.array([[0, 0]])
        N = 1
    if M == 0:
        T = np.array([[0, 0]])
        M = 1
    DUL = sklearn.metrics.pairwise.pairwise_distances(S, T)

    #Put diagonal elements into the matrix
    #Rotate the diagrams to make it easy to find the straight line
    #distance to the diagonal
    cp = np.cos(np.pi/4)
    sp = np.sin(np.pi/4)
    R = np.array([[cp, -sp], [sp, cp]])
    S = S[:, 0:2].dot(R)
    T = T[:, 0:2].dot(R)
    D = np.zeros((N+M, N+M))
    D[0:N, 0:M] = DUL
    UR = np.max(D)*np.ones((N, N))
    np.fill_diagonal(UR, S[:, 1])
    D[0:N, M:M+N] = UR
    UL = np.max(D)*np.ones((M, M))
    np.fill_diagonal(UL, T[:, 1])
    D[N:M+N, 0:M] = UL
    D = D.tolist()

    #Run the hungarian algorithm
    matchidx = hungarian.lap(D)[0]
    matchidx = [(i, matchidx[i]) for i in range(len(matchidx))]
    matchdist = 0
    for pair in matchidx:
        (i, j) = pair
        matchdist += D[i][j]

    return (matchidx, matchdist, D)

def sortAndGrab(dgm, NBars = 10, BirthTimes = False):
    """
    Do sorting and grabbing with the option to include birth times
    Zeropadding is also taken into consideration
    """
    dgmNP = np.array(dgm)
    if dgmNP.size == 0:
        if BirthTimes:
            ret = np.zeros(NBars*2)
        else:
            ret = np.zeros(NBars)
        return ret
    #Indices for reverse sort
    idx = np.argsort(-(dgmNP[:, 1] - dgmNP[:, 0])).flatten()
    ret = dgmNP[idx, 1] - dgmNP[idx, 0]
    ret = ret[0:min(NBars, len(ret))].flatten()
    if len(ret) < NBars:
        ret = np.append(ret, np.zeros(NBars - len(ret)))
    if BirthTimes:
        bt = dgmNP[idx, 0].flatten()
        bt = bt[0:min(NBars, len(bt))].flatten()
        if len(bt) < NBars:
            bt = np.append(bt, np.zeros(NBars - len(bt)))
        ret = np.append(ret, bt)
    return ret


def getHeatRasterized(dgm, sigma, xrange, yrange, UpFac = 10):
    """
    Get a discretized verison of the solution of the heat flow equation
    described in the CVPR 2015 paper
    """
    I = np.array(dgm)
    if I.size == 0:
        return np.zeros((yrange.size, xrange.size))
    NX = xrange.size
    NY = yrange.size
    #Rasterize on a finer grid and downsample
    NXFine = UpFac*NX
    NYFine = UpFac*NY
    xrangeup = np.linspace(xrange[0], xrange[-1], NXFine)
    yrangeup = np.linspace(yrange[0], yrange[-1], NYFine)
    X, Y = np.meshgrid(xrangeup, yrangeup)
    u = np.zeros(X.shape)
    for ii in range(I.shape[0]):
        u = u + np.exp(-( (X - I[ii, 0])**2 + (Y - I[ii, 1])**2 )/(4*sigma))
        #Now subtract mirror diagonal
        u = u - np.exp(-( (X - I[ii, 1])**2 + (Y - I[ii, 0])**2 )/(4*sigma))
    u = (1.0/(4*np.pi*sigma))*u
    u = scipy.misc.imresize(u, (NY, NX))
    return u


def evalHeatKernel(dgm1, dgm2, sigma):
    """
    Evaluate the continuous heat-based kernel between dgm1 and dgm2 (more correct
    than L2 on the discretized verison above but may be slower because can't exploit
    fast matrix multiplication when evaluating many, many kernels)
    """
    kSigma = 0
    I1 = np.array(dgm1)
    I2 = np.array(dgm2)
    for i in range(I1.shape[0]):
        p = I1[i, 0:2]
        for j in range(I2.shape[0]):
            q = I2[j, 0:2]
            qc = I2[j, 1::-1]
            kSigma += np.exp(-(np.sum((p-q)**2))/(8*sigma)) - np.exp(-(np.sum((p-qc)**2))/(8*sigma))
    return kSigma / (8*np.pi*sigma)

def evalHeatDistance(dgm1, dgm2, sigma):
    """
    Return the pseudo-metric between two diagrams based on the continuous
    heat kernel
    """
    return np.sqrt(evalHeatKernel(dgm1, dgm1, sigma) + evalHeatKernel(dgm2, dgm2, sigma) - 2*evalHeatKernel(dgm1, dgm2, sigma))
