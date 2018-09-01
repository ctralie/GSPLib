"""
Programmer: Chris Tralie
Purpose: To implement curvature/torsion scale space on sampled curves and to 
         create animations of the algorithm in action

Reference:
[1] Mokhtarian, Farzin, and Alan Mackworth. "Scale-based description and recognition of planar curves and two-dimensional shapes." IEEE transactions on pattern analysis and machine intelligence 1 (1986): 34-43.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gf1d
import scipy.io as sio
import scipy.misc
import matplotlib.animation as animation

def getCurvVectors(X, MaxOrder, sigma, loop = False, m = 'nearest'):
    """
    Get smoothed curvature vectors up to a particular order
    Parameters
    ----------
    X: ndarray(N, d)
        An N x d matrix of points in R^d
    MaxOrder: int
        The maximum order of torsion to compute (e.g. 3 for position, velocity, and curvature, and torsion)
    sigma: float
        The smoothing amount
    loop: boolean
        Whether to treat this trajectory as a topological loop (i.e. add an edge between first and last point)
    
    Returns
    -------
    Curvs: A list of (N, 3) arrays, starting with the smoothed curve, then followed
           by the smoothed velocity, curvature, torsion, etc. up to the MaxOrder
    """
    if loop:
        m = 'wrap'
    XSmooth = gf1d(X, sigma, axis=0, order = 0, mode = m)
    Vel = gf1d(X, sigma, axis=0, order = 1, mode = m)
    VelNorm = np.sqrt(np.sum(Vel**2, 1))
    VelNorm[VelNorm == 0] = 1
    Curvs = [XSmooth, Vel]
    for order in range(2, MaxOrder+1):
        Tors = gf1d(X, sigma, axis=0, order=order, mode = m)
        for j in range(1, order):
            #Project away other components
            NormsDenom = np.sum(Curvs[j]**2, 1)
            NormsDenom[NormsDenom == 0] = 1
            Norms = np.sum(Tors*Curvs[j], 1)/NormsDenom
            Tors = Tors - Curvs[j]*Norms[:, None]
        Tors = Tors/(VelNorm[:, None]**order)
        Curvs.append(Tors)
    return Curvs

def getZeroCrossings(Curvs):
    """
    Get zero crossings estimates from all curvature/torsion
    measurements by using the dot product
    Parameters
    ----------
    Curvs: list
    A list of (N, 3) arrays, starting with the smoothed curve, then followed
           by the smoothed velocity, curvature, torsion, etc. up to the MaxOrder
   
    Returns
    -------
    Crossings: list
        List of crossing arrays for each curvature order.  In each array, there
        is a 1 if a sign crossing occurred, zero if otherwise
    """
    Crossings = []
    for C in Curvs:
        dots = np.sum(C[0:-1, :]*C[1::, :], 1)
        cross = np.arange(len(dots))
        cross = cross[dots < 0]
        Crossings.append(cross)
    return Crossings

def getScaleSpaceImages(X, MaxOrder, sigmas, loop):
    """
    Return the curvature scale space images for a sampled spacecurve
    Parameters
    ----------
    X: ndarray(N, d)
        An N x d matrix of points in R^d
    MaxOrder: int
        The maximum order of torsion to compute (e.g. 3 for position, velocity, and curvature, and torsion)
    sigmas: ndarray(M)
        A list of smoothing amounts at which to estimate curvature/torsion/etc
    loop: boolean
        Whether to treat this trajectory as a topological loop (i.e. add an edge between first and last point)
    
    Returns
    -------
    SSImages: list of ndarray(M, N)
        A list of scale space images for each curvature order
    """
    NSigmas = len(sigmas)
    SSImages = []
    for i in range(MaxOrder):
        SSImages.append(np.zeros((NSigmas, X.shape[0])))
    for s in range(len(sigmas)):
        Curvs = getCurvVectors(X, MaxOrder, sigmas[s], loop)
        Crossings = getZeroCrossings(Curvs[1::])
        for i in range(MaxOrder):
            if len(Crossings[i]) > 0:
                SSImages[i][s, Crossings[i]] = 1.0
    return SSImages

class CSSAnimator(animation.FuncAnimation):
    """
    A class for doing animation of curvature scale space images
    for 2D/3D curves
    """
    def __init__(self, fig, X, sigmas, filename, ImageRes = 400, loop = False):
        self.fig = fig
        self.X = X
        self.sigmas = sigmas
        self.ImageRes = ImageRes
        self.SSImage = np.zeros((len(sigmas), X.shape[0]))
        self.loop = loop

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3 = plt.gca()
        [self.ax1, self.ax2, self.ax3] = [ax1, ax2, ax3]

        #Original curve
        self.origCurve, = ax1.plot(X[:, 0], X[:, 1])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Original Curve')

        #Smoothed Curve
        self.smoothCurve = ax2.scatter([0, 0], [0, 0], [1, 1])
        self.smoothCurveInfl = ax2.scatter([], [], 20, 'r')
        ax2.set_xlim([np.min(X[:, 0]), np.max(X[:, 0])])
        ax2.set_ylim([np.min(X[:, 1]), np.max(X[:, 1])])
        plt.title("Smoothed Curve")

        #Scale space image plot
        #I have to hack this so the colormap is scaled properly
        initial = np.zeros((ImageRes, ImageRes))
        initial[0] = 1
        self.ssImagePlot = ax3.imshow(initial, extent = (0, 1, self.sigmas[0], self.sigmas[-1]), interpolation = 'none', aspect = 'auto', cmap=plt.get_cmap('gray'))
        plt.xlabel("t")
        plt.ylabel("Sigma")
        plt.title("Scale Space Image (Zero Crossings)")
        #Setup animation thread
        animation.FuncAnimation.__init__(self, fig, func = self._draw_frame, frames = len(sigmas), interval = 50)

        #Write movie
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata, bitrate = 30000)
        self.save("out.mp4", writer = writer)

    def _draw_frame(self, i):
        print i

        Curvs = getCurvVectors(self.X, 2, self.sigmas[i], loop = self.loop)
        Crossings = getZeroCrossings(Curvs)
        XSmooth = Curvs[0]
        Curv = Curvs[2]
        CurvMag = np.sqrt(np.sum(Curv**2, 1)).flatten()
        
        #Draw smoothed curve with inflection points
        self.smoothCurve.set_offsets(XSmooth[:, 0:2])
        self.smoothCurve.set_array(CurvMag)
        self.smoothCurve.__sizes = 20*np.ones(XSmooth.shape[0])
        XInflection = XSmooth[Crossings[2], :]
        self.smoothCurveInfl.set_offsets(XInflection[:, 0:2])
        self.ax2.set_title("Sigma = %.3g"%self.sigmas[i])
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        
        self.SSImage[-i-1, Crossings[2]] = 1
        #Do some rudimentary anti-aliasing
        SSImageRes = scipy.misc.imresize(self.SSImage, [self.ImageRes, self.ImageRes])
        SSImageRes[SSImageRes > 0] = 1
        SSImageRes = 1-SSImageRes
        self.ssImagePlot.set_array(SSImageRes)

if __name__ == '__main__':
    X = np.loadtxt('Airplane.txt')
    Y = np.array(X, dtype=np.float32)
    sigmas = np.linspace(10, 160, 160)

    fig = plt.figure(figsize=(8, 6))
    ani = CSSAnimator(fig, Y, sigmas, "out.mp4", loop = True)
