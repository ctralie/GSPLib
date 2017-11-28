"""
Programmer: Chris Tralie
Purpose: To do a simple classification experiment of different
sampled curve families using RQA statistics and subspace angles
between low rank Laplacians
"""
from Laplacian import *
from RQA import *
from SubspaceAngles import *
from SyntheticCurves import *
from SimilarityFusion import *
from mpl_toolkits.mplot3d import Axes3D

def getPrecisionRecall(D, NPerClass):
    """
    Get average precision recall graph, assuming an equal number of 
    classes and that elements in each class are in a contiguous chunk
    :param D: Matrix of all similarities between objects
    :param NPerClass: Number of objects per class
    :returns: An array of precisions for NPerClass-1 objects recalled
    """
    PR = np.zeros(NPerClass-1)
    DI = np.argsort(D, 1)
    for i in range(DI.shape[0]):
        pr = np.zeros(NPerClass-1)
        recall = 0
        for j in range(1, DI.shape[1]): #Skip the first point (don't compare to itself)
            if DI[i, j]/NPerClass == i/NPerClass:
                pr[recall] = float(recall+1)/j
                recall += 1
        PR += pr
    return PR/float(DI.shape[0])

def doCurveFamilyTest(NPerClass, SamplesPerCurve, NEigs, numPlotExamples = 0):
    """
    Do a classification experiment with different curve families, \
        using RQA statistics and subspace angle
    :param NPerClass: Number of curves per class
    :param SamplesPerCurve: Number of points in each sampled curve
    :param NEigs: Number of eigenvectors to use in low rank\
        Laplacian approximation
    :param numPlotExamples: Number of sampled examples to plot for each family
    """
    #Step 1: Create a library of families of curves
    Curves = {}
    Curves['VFig8'] = lambda t: getVivianiFigure8(0.5, t)
    Curves['TSCubic'] = lambda t: getTschirnhausenCubic(1, t)
    Curves['TK23'] = lambda t: getTorusKnot(2, 3, t)
    Curves['TK35'] = lambda t: getTorusKnot(3, 5, t)
    Curves['PCircle'] = lambda t: getPinchedCircle(t)
    Curves['LJ32'] = lambda t: getLissajousCurve(1, 1, 3, 2, 0, t)
    Curves['LJ54'] = lambda t: getLissajousCurve(1, 1, 5, 4, 0, t)
    Curves['Helix'] = lambda t: getConeHelix(1, 16, t)
    Curves['Epi13'] = lambda t: getEpicycloid(1.5, 0.5, t)
    
    
    #Parameters for bump distortions
    Kappa = 0.1
    NRelMag = 2
    NBumps = 3
    
    Xs = []
    RQAStats = [] #Will hold an array of all RQA Stats
    LapVecs = [] #Will hold an array of Laplacian eigenvectors
    t = np.linspace(0, 1, SamplesPerCurve)
    plt.figure(figsize=(18, 6))
    for c in Curves:
        for i in range(NPerClass):
            print("%s Sample %i"%(c, i))
            X = Curves[c](t)
            (X, Bumps) = addRandomBumps(X, Kappa, NRelMag, NBumps)
            X = X + 0.1*np.random.randn(X.shape[0], X.shape[1])
            Xs.append(X)
            #Get recurrence plot with 20% nearest neighbors
            R = CSMToBinaryMutual(getSSM(X), 0.2)
            stats = getRQAStats(R, 5, 5)
            stats = [stats[s] for s in stats]
            RQAStats.append(stats)
            A = np.array(R) #Adjacency matrix
            np.fill_diagonal(A, 0)
            (w, V, L) = getLaplacianEigsDense(A, NEigs)
            LapVecs.append(V)
            if i < numPlotExamples:
                plt.clf()
                #Plot two examples for each class
                plt.subplot(131)
                if X.shape[1] == 3:
                    ax = plt.subplot(131, projection = '3d')
                    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
                else:
                    plt.subplot(131)
                    plt.scatter(X[:, 0], X[:, 1], 20, np.arange(X.shape[0]), cmap = 'Spectral')
                plt.title("Point Cloud")
                plt.subplot(132)
                plt.imshow(R, cmap = 'gray', interpolation = 'nearest')
                plt.title("Cross-Recurrence Plot")
                plt.subplot(133)
                plt.imshow(V, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
                plt.xlabel("Laplacian Eigenvector Num")
                plt.title("Laplacian Eigenvectors")
                plt.savefig("%s_%i.svg"%(c, i), bbox_inches = 'tight')
    
    plt.figure(figsize=(24, 6))
    x = np.arange(NPerClass/2, NPerClass*len(Curves), NPerClass)
    
    #Step 2: Compare RQA stats
    RQAStats = np.array(RQAStats)
    #Divide each column by standard deviation to normalize features
    RQAStats = RQAStats/np.std(RQAStats, 0)[None, :]
    DRQA = getSSM(RQAStats)
    
    plt.subplot(141)
    plt.imshow(DRQA, cmap = 'afmhot', interpolation = 'nearest')
    plt.xticks(x, [c for c in Curves], rotation='vertical')
    plt.yticks(x, [c for c in Curves], rotation='horizontal')
    plt.title("RQA Stats")
    
    #Step 3: Compute subspace angles
    DSubspace = 0*DRQA
    for i in range(DSubspace.shape[0]):
        for j in range(DSubspace.shape[1]):
            DSubspace[i, j] = getSubspaceAngle(LapVecs[i], LapVecs[j])['chordal']
    plt.subplot(142)
    plt.imshow(DSubspace, cmap = 'afmhot', interpolation = 'nearest')
    plt.xticks(x, [c for c in Curves], rotation='vertical')
    plt.yticks(x, [c for c in Curves], rotation='horizontal')
    plt.title("Rank %i Laplacian\nSubspace Angles"%NEigs)
    
    #Step 4: Fuse them together
    DFused = doSimilarityFusion([DRQA, DSubspace])
    plt.subplot(143)
    plt.imshow(DFused, cmap = 'afmhot', interpolation = 'nearest')
    plt.xticks(x, [c for c in Curves], rotation='vertical')
    plt.yticks(x, [c for c in Curves], rotation='horizontal')
    plt.title("Fused Similarity")
    DFused = np.exp(-DFused)
    
    #Step 4: Make Precision Recall Graphs
    PR1 = getPrecisionRecall(DRQA, NPerClass)
    PR2 = getPrecisionRecall(DSubspace, NPerClass)
    PR3 = getPrecisionRecall(DFused, NPerClass)
    recall = np.linspace(1.0/len(PR1), 1, len(PR1))
    plt.subplot(144)
    plt.plot(recall, PR1, 'b')
    plt.plot(recall, PR2, 'r')
    plt.plot(recall, PR3, 'k')
    plt.legend(['RQA (%.3g)'%np.mean(PR1), 'Subspace (%.3g)'%np.mean(PR2), 'Fused (%.3g)'%np.mean(PR3)], bbox_to_anchor=(0.7, 0.3))
    plt.title("Precision Recall")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.savefig("Stats.svg", bbox_inches = 'tight')

if __name__ == '__main__':
    np.random.seed(100)
    NPerClass = 50
    SamplesPerCurve = 150
    NEigs = 10
    doCurveFamilyTest(NPerClass, SamplesPerCurve, NEigs, 2)
