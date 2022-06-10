# Additional classes and functions

import itertools
from functools import reduce
from math import floor, ceil, log

import numpy as np
from numpy.lib.stride_tricks import as_strided

from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as mpcm
import matplotlib.colors as mpcolors
from matplotlib.patches import Polygon



# -----------------------------------------------------------------------------
# Power-law fitting procedure
#
# function alpha, xmin, L = plfit(x, varargin)
# PLFIT fits a power-law distributional model to data.
#    Source: http://www.santafe.edu/~aaronc/powerlaws/
#            aaron.clauset@colorado.edu
#
#    PLFIT(x) estimates x_min and alpha according to the goodness-of-fit
#    based method described in Clauset, Shalizi, Newman (2007). x is a
#    vector of observations of some quantity to which we wish to fit the
#    power-law distribution p(x) ~  x^-alpha for x >= xmin.
#    PLFIT automatically detects whether x is composed of real or integer
#    values, and applies the appropriate method. For discrete data, if
#    min(x) > 1000, PLFIT uses the continuous approximation, which is
#    a reliable in this regime.
#
#    The fitting procedure works as follows:
#    1) For each possible choice of x_min, we estimate alpha via the
#       method of maximum likelihood, and calculate the Kolmogorov-Smirnov
#       goodness-of-fit statistic D.
#    2) We then select as our estimate of x_min, the value that gives the
#       minimum value D over all values of x_min.
#
#    Note that this procedure gives no estimate of the uncertainty of the
#    fitted parameters, nor of the validity of the fit.
#
#    Example:
#       x = [500,150,90,81,75,75,70,65,60,58,49,47,40]
#       [alpha, xmin, L] = plfit(x)
#   or  a = plfit(x)
#
#    The output 'alpha' is the maximum likelihood estimate of the scaling
#    exponent, 'xmin' is the estimate of the lower bound of the power-law
#    behavior, and L is the log-likelihood of the data x>=xmin under the
#    fitted power law.
#
#    For more information, try 'type plfit'
#
#    See also PLVAR, PLPVA

# Version 1.0.10 (2010 January)
# Copyright (C) 2008-2011 Aaron Clauset (Santa Fe Institute)

# Ported to Python by Joel Ornstein (2011 July)
# (joel_ornstein@hmc.edu)

# Distributed under GPL 2.0
# http://www.gnu.org/copyleft/gpl.html
# PLFIT comes with ABSOLUTELY NO WARRANTY
#
#
# The 'zeta' helper function is modified from the open-source library 'mpmath'
#   mpmath: a Python library for arbitrary-precision floating-point arithmetic
#   http://code.google.com/p/mpmath/
#   version 0.17 (February 2011) by Fredrik Johansson and others
#

# Notes:
#
# 1. In order to implement the integer-based methods in Matlab, the numeric
#    maximization of the log-likelihood function was used. This requires
#    that we specify the range of scaling parameters considered. We set
#    this range to be 1.50 to 3.50 at 0.01 intervals by default.
#    This range can be set by the user like so,
#
#       a = plfit(x,'range',[1.50,3.50,0.01])
#
# 2. PLFIT can be told to limit the range of values considered as estimates
#    for xmin in three ways. First, it can be instructed to sample these
#    possible values like so,
#
#       a = plfit(x,'sample',100)
#
#    which uses 100 uniformly distributed values on the sorted list of
#    unique values in the data set. Second, it can simply omit all
#    candidates above a hard limit, like so
#
#       a = plfit(x,'limit',3.4)
#
#    Finally, it can be forced to use a fixed value, like so
#
#       a = plfit(x,'xmin',3.4)
#
#    In the case of discrete data, it rounds the limit to the nearest
#    integer.
#
# 3. When the input sample size is small (e.g., < 100), the continuous
#    estimator is slightly biased (toward larger values of alpha). To
#    explicitly use an experimental finite-size correction, call PLFIT like
#    so
#
#       a = plfit(x,'finite')
#
#    which does a small-size correction to alpha.
#
# 4. For continuous data, PLFIT can return erroneously large estimates of
#    alpha when xmin is so large that the number of obs x >= xmin is very
#    small. To prevent this, we can truncate the search over xmin values
#    before the finite-size bias becomes significant by calling PLFIT as
#
#       a = plfit(x,'nosmall')
#
#    which skips values xmin with finite size bias > 0.1.

def plfit(x, *varargin):
    vec     = []
    sample  = []
    xminx   = []
    limit   = []
    finite  = False
    nosmall = False
    nowarn  = False

    # parse command-line parameters trap for bad input
    i=0
    while i<len(varargin):
        argok = 1
        if type(varargin[i])==str:
            if varargin[i]=='range':
                Range = varargin[i+1]
                if Range[1]>Range[0]:
                    argok=0
                    vec=[]
                try:
                    vec=list(map(lambda X:X*float(Range[2])+Range[0], range(int((Range[1]-Range[0])/Range[2]))))
                except:
                    argok=0
                    vec=[]

                if Range[0]>=Range[1]:
                    argok=0
                    vec=[]
                    i-=1
                i+=1
            elif varargin[i]== 'sample':
                sample  = varargin[i+1]
                i = i + 1
            elif varargin[i]==  'limit':
                limit   = varargin[i+1]
                i = i + 1
            elif varargin[i]==  'xmin':
                xminx   = varargin[i+1]
                i = i + 1
            elif varargin[i]==  'finite':
                finite  = True
            elif varargin[i]==  'nowarn':
                nowarn  = True
            elif varargin[i]==  'nosmall':
                nosmall = True
            else:
                argok=0

        if not argok:
            print( '(PLFIT) Ignoring invalid argument #',i+1 )

        i = i+1

    if vec!=[] and (type(vec)!=list or min(vec)<=1):
        print( '(PLFIT) Error: ''range'' argument must contain a vector or minimum <= 1. using default.\n' )
        vec = []

    if sample!=[] and sample<2:
        print('(PLFIT) Error: ''sample'' argument must be a positive integer > 1. using default.\n' )
        sample = []

    if limit!=[] and limit<min(x):
        print('(PLFIT) Error: ''limit'' argument must be a positive value >= 1. using default.\n' )
        limit = []

    if xminx!=[] and xminx>=max(x):
        print('(PLFIT) Error: ''xmin'' argument must be a positive value < max(x). using default behavior.\n' )
        xminx = []

    # select method (discrete or continuous) for fitting
    if reduce(lambda X,Y:X==True and floor(Y)==float(Y),x,True):
        f_dattype = 'INTS'
    elif reduce(lambda X,Y:X==True and (type(Y)==int or type(Y)==float or type(Y)==int),x,True):
        f_dattype = 'REAL'
    else:
        f_dattype = 'UNKN'

    if f_dattype=='INTS' and min(x) > 1000 and len(x)>100:
        f_dattype = 'REAL'

    # estimate xmin and alpha, accordingly
    if f_dattype== 'REAL':
        xmins = list(set(x)) # unique(x)
        xmins.sort()
        xmins = xmins[0:-1]
        if xminx!=[]:
            xmins = [min(filter(lambda X: X>=xminx,xmins))]
        if limit!=[]:
            xmins=list(filter(lambda X: X<=limit,xmins))
            if xmins==[]: xmins = [min(x)]
        if sample!=[]:
            step = float(len(xmins))/(sample-1)
            index_curr=0
            new_xmins=[]
            for i in range (0,sample):
                if round(index_curr)==len(xmins): index_curr-=1
                new_xmins.append(xmins[int(round(index_curr))])
                index_curr+=step
            xmins = list(set(new_xmins)) #unique(new_xmins)
            xmins.sort()
        dat = []
        z = sorted(x)
        for xm in range(0,len(xmins)):
            xmin = xmins[xm]
            z = list(filter(lambda X:X>=xmin,z))
            n = len(z)
            # estimate alpha using direct MLE
            a    = float(n) / sum(list(map(lambda X: log(float(X)/xmin),z)))
            if nosmall:
                if (a-1)/sqrt(n) > 0.1 and dat!=[]:
                    xm = len(xmins)+1
                    break
            # compute KS statistic
            cf = list(map(lambda X:1-pow((float(xmin)/X),a),z))
            dat.append( max( list(map(lambda X: abs(cf[X]-float(X)/n),range(0,n))) ) )
        D = min(dat)
        xmin  = xmins[dat.index(D)]
        z = list(filter(lambda X:X>=xmin,x))
        z.sort()
        n = len(z)
        alpha = 1 + n / sum(list(map(lambda X: log(float(X)/xmin),z)))
        if finite: alpha = alpha*float(n-1)/n+1./n  # finite-size correction
        if n < 50 and not finite and not nowarn:
            print( '(PLFIT) Warning: finite-size bias may be present.\n')
        L = n*log((alpha-1)/xmin) - alpha*sum(list(map(lambda X: log(float(X)/xmin),z)))

    elif f_dattype== 'INTS':
        x=list(map(int,x))
        if vec==[]:
            for X in range(150,351):
                vec.append(X/100.)    # covers range of most practical
        # scaling parameters
        zvec = list(map(zeta, vec))
        xmins = list(set(x))
        xmins.sort()
        xmins = xmins[0:-1]
        if xminx!=[]:
            xmins = [min(filter(lambda X: X>=xminx,xmins))]

        if limit!=[]:
            limit = round(limit)
            xmins=list(filter(lambda X: X<=limit,xmins))
            if xmins==[]: xmins = [min(x)]

        if sample!=[]:
            step = float(len(xmins))/(sample-1)
            index_curr=0
            new_xmins=[]
            for i in range (0,sample):
                if round(index_curr)==len(xmins): index_curr-=1
                new_xmins.append(xmins[int(round(index_curr))])
                index_curr+=step
            xmins = list(set(new_xmins))
            xmins.sort()

        if xmins==[]:
            print( '(PLFIT) Error: x must contain at least two unique values.\n')
            alpha = 'Not a Number'
            xmin = x[0]
            D = 'Not a Number'
            return [alpha,xmin,D]

        xmax   = max(x)
        z      = x
        z.sort()
        datA=[]
        datB=[]

        for xm in range(0,len(xmins)):
            xmin = xmins[xm]
            z    = list(filter(lambda X:X>=xmin,z))
            n    = len(z)
            # estimate alpha via direct maximization of likelihood function
            # force iterative calculation
            L       = []
            slogz   = sum(list(map(log,z)))
            xminvec = list(map(float,range(1,xmin)))
            for k in range(0,len(vec)):
                L.append(-vec[k]*float(slogz) - float(n)*log(float(zvec[k]) - sum(list(map(lambda X:pow(float(X),-vec[k]),xminvec)))))
            I = L.index(max(L))
            # compute KS statistic
            fit = reduce(lambda X,Y: X+[Y+X[-1]], (list(map(lambda X: pow(X,-vec[I])/(float(zvec[I])-sum(list(map(lambda X: pow(X,-vec[I]),map(float,range(1,xmin)))))),range(xmin,xmax+1)))),[0])[1:]
            cdi=[]
            for XM in range(xmin,xmax+1):
                cdi.append(len(list(filter(lambda X: floor(X)<=XM,z)))/float(n))
            datA.append(max( list(map(lambda X: abs(fit[X] - cdi[X]),range(0,xmax-xmin+1)))))
            datB.append(vec[I])
        # select the index for the minimum value of D
        I = datA.index(min(datA))
        xmin  = xmins[I]
        z     = list(filter(lambda X:X>=xmin,x))
        n     = len(z)
        alpha = datB[I]
        if finite: alpha = alpha*(n-1.)/n+1./n  # finite-size correction
        if n < 50 and not finite and not nowarn:
            print( '(PLFIT) Warning: finite-size bias may be present.\n')

        L     = -alpha*sum(list(map(log,z))) - n*log(zvec[vec.index(max(list(filter(lambda X:X<=alpha,vec))))] - sum(list(map(lambda X: pow(X,-alpha),range(1,xmin)))))
    else:
        print( '(PLFIT) Error: x must contain only reals or only integers.\n')
        alpha = []
        xmin  = []
        L     = []

    return alpha,xmin,L

# helper function
def _polyval(coeffs, x):
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + x*p
    return p

_zeta_int = [\
-0.5,
0.0,
1.6449340668482264365,1.2020569031595942854,1.0823232337111381915,
1.0369277551433699263,1.0173430619844491397,1.0083492773819228268,
1.0040773561979443394,1.0020083928260822144,1.0009945751278180853,
1.0004941886041194646,1.0002460865533080483,1.0001227133475784891,
1.0000612481350587048,1.0000305882363070205,1.0000152822594086519,
1.0000076371976378998,1.0000038172932649998,1.0000019082127165539,
1.0000009539620338728,1.0000004769329867878,1.0000002384505027277,
1.0000001192199259653,1.0000000596081890513,1.0000000298035035147,
1.0000000149015548284]

_zeta_P = [-3.50000000087575873, -0.701274355654678147,
-0.0672313458590012612, -0.00398731457954257841,
-0.000160948723019303141, -4.67633010038383371e-6,
-1.02078104417700585e-7, -1.68030037095896287e-9,
-1.85231868742346722e-11][::-1]

_zeta_Q = [1.00000000000000000, -0.936552848762465319,
-0.0588835413263763741, -0.00441498861482948666,
-0.000143416758067432622, -5.10691659585090782e-6,
-9.58813053268913799e-8, -1.72963791443181972e-9,
-1.83527919681474132e-11][::-1]

_zeta_1 = [3.03768838606128127e-10, -1.21924525236601262e-8,
2.01201845887608893e-7, -1.53917240683468381e-6,
-5.09890411005967954e-7, 0.000122464707271619326,
-0.000905721539353130232, -0.00239315326074843037,
0.084239750013159168, 0.418938517907442414, 0.500000001921884009]

_zeta_0 = [-3.46092485016748794e-10, -6.42610089468292485e-9,
1.76409071536679773e-7, -1.47141263991560698e-6, -6.38880222546167613e-7,
0.000122641099800668209, -0.000905894913516772796, -0.00239303348507992713,
0.0842396947501199816, 0.418938533204660256, 0.500000000000000052]

def zeta(s):
    """
    Riemann zeta function, real argument
    """
    if not isinstance(s, (float, int)):
        try:
            s = float(s)
        except (ValueError, TypeError):
            try:
                s = complex(s)
                if not s.imag:
                    return complex(zeta(s.real))
            except (ValueError, TypeError):
                pass
            raise NotImplementedError
    if s == 1:
        raise ValueError("zeta(1) pole")
    if s >= 27:
        return 1.0 + 2.0**(-s) + 3.0**(-s)
    n = int(s)
    if n == s:
        if n >= 0:
            return _zeta_int[n]
        if not (n % 2):
            return 0.0
    if s <= 0.0:
        return 0
    if s <= 2.0:
        if s <= 1.0:
            return _polyval(_zeta_0,s)/(s-1)
        return _polyval(_zeta_1,s)/(s-1)
    z = _polyval(_zeta_P,s) / _polyval(_zeta_Q,s)
    return 1.0 + 2.0**(-s) + 3.0**(-s) + 4.0**(-s)*z


# ------------------------------------------------
# Additional classes and functions

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def firingrate( start, end, spiketrains, bin_size=10 ):
    """
    Population rate
    as in https://neuronaldynamics.epfl.ch/online/Ch7.S2.html
    """
    if len(spiketrains)==0:
        return NaN
    # create bin edges based on start and end of slices and bin size
    bin_edges = np.arange( start, end, bin_size )
    # print("bin_edges",bin_edges.shape)
    # binning total time, and counting the number of spike times in each bin
    hist = np.zeros( bin_edges.shape[0]-1 )
    for spike_times in spiketrains:
        hist = hist + np.histogram( spike_times, bin_edges )[0]
    return ((hist / len(spiketrains) ) / bin_size ) # average over population


def firinghist_one( spiketrain, bin_edges ):
    return np.zeros( bin_edges.shape[0]-1 ) + np.histogram( spiketrain, bin_edges )[0]


from itertools import islice
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


from functools import reduce
def factors(n):
    return reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))

# Finds baseline in the firing rate
# Params:
#   l for smoothness (λ)
#   p for asymmetry
# Both have to be tuned to the data at hand.
# We found that generally is a good choice (for a signal with positive peaks):
#   10^2 ≤ l ≤ 10^9
#   0.001 ≤ p ≤ 0.1
# but exceptions may occur.
# In any case one should vary l on a grid that is approximately linear for log l
from scipy import sparse
from scipy.sparse.linalg import spsolve
def baseline(y, l, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = l * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


# Cross-correlation with maxlag
# from: https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy
def crosscorrelation(x, y, maxlag, mode='corr'):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag), strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    if mode == 'dot':       # get lagged dot product
        return T.dot(px)
    elif mode == 'corr':    # gets Pearson correlation
        return (T.dot(px)/px.size - (T.mean(axis=1)*px.mean())) / (np.std(T, axis=1) * np.std(px))


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.
    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]
    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
    y = max(ly, ry) + dh
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)
    plt.plot(barx, bary, c='black')
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    # plt.text(*mid, text, **kwargs)


# crop thecenter of an numpy image, given the size of resulting cropped
# it assumes grayscale
# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


# fit line
# https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
# use:
#  a, b = best_fit(X, Y)
#  yfit = [a + b * xi for xi in X]
#  plt.plot(X, yfit)
def linear_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    return a, b

# remove based on median, more robust than mean
def mask_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

# power law
def powerlaw(x, a, c):
	return c * x**(-a)


# http://phrogz.net/css/distinct-colors.html -- 76 colors
custom_palette = [
    '#ffa8a8', '#b37676', '#664343', '#ff8c8c', '#b26262', '#ff7070', '#b24f4f',
    '#662d2d', '#ff5454', '#b23b3b', '#ff3838', '#b22727', '#661616', '#ff0000',
    '#b20000', '#660000', '#ffffa8', '#b3b376', '#ffff8c', '#b2b362', '#666638',
    '#ffff54', '#b2b33b', '#666622', '#ffff00', '#b2b300', '#666600', '#a8ffa8',
    '#76b376', '#8cff8c', '#62b362', '#386638', '#70ff70', '#54ff54', '#3bb33b',
    '#226622', '#00ff00', '#00b300', '#006600', '#8cffff', '#62b3b3', '#386666',
    '#00ffff', '#00b3b3', '#006666', '#a8a8ff', '#7676b3', '#434366', '#8c8cff',
    '#6262b3', '#383866', '#7070ff', '#4f4fb3', '#2d2d66', '#5454ff', '#3b3bb3',
    '#222266', '#3838ff', '#2727b3', '#161666', '#0000ff', '#0000b3', '#000066',
    '#ffa8ff', '#b376b3', '#664366', '#ff8cff', '#b362b3', '#663866', '#ff70ff',
    '#ff54ff', '#b33bb3', '#662266', '#ff00ff', '#b300b3', '#660066'
]

# http://phrogz.net/css/distinct-colors.html -- 128 colors
large_palette = [
    '#d90000', '#990000', '#590000', '#ff4040', '#bf3030', '#7f2020', '#591616',
    '#994d4d', '#331a1a', '#e6acac', '#8c6969', '#594343', '#332626', '#999600',
    '#4c4b00', '#fffc40', '#bfbd30', '#7f7e20', '#66651a', '#403f10', '#e5e473',
    '#bfbe60', '#99984d', '#737239', '#59592d', '#403f20', '#e6e5ac', '#bfbe8f',
    '#8c8c69', '#66664d', '#4d4c39', '#333326', '#06bf00', '#059900', '#023300',
    '#46ff40', '#238020', '#195916', '#124010', '#77e673', '#63bf60', '#4f994d',
    '#3b7339', '#2e592d', '#1a331a', '#c1ffbf', '#9bcc99', '#7ea67c', '#618060',
    '#4d664d', '#304030', '#008c85', '#005955', '#003330', '#40fff5', '#2db3ac',
    '#73e6e0', '#40807c', '#336663', '#20403e', '#bffffc', '#99ccc9', '#7ca6a4',
    '#567371', '#394d4c', '#263332', '#000fe6', '#000dbf', '#000a99', '#000873',
    '#000659', '#000440', '#3d49f2', '#333dcc', '#262e99', '#1d2273', '#13174d',
    '#0d0f33', '#8088ff', '#6066bf', '#4d5299', '#333666', '#26294d', '#1a1b33',
    '#bfc4ff', '#8f93bf', '#737599', '#565873', '#434459', '#262733', '#ea00ff',
    '#c700d9', '#8c0099', '#690073', '#520059', '#2f0033', '#a72db3', '#48134d',
    '#f480ff', '#c466cc', '#9f53a6', '#7a4080', '#623366', '#49264d', '#311a33',
    '#e1ace6', '#bb8fbf', '#967399', '#705673', '#4b394d', '#322633', '#ff001a',
    '#bf0013', '#8c000e', '#73000b', '#4c0008', '#330005', '#ff4053', '#d93646',
    '#992632', '#731d25', '#401015', '#ff808c', '#cc6670', '#73393f', '#4c262a',
    '#bf8f94', '#735659'
]

# ------------------------------------------------------------------------------
# Hack to get viridis colormap in the version of matplotlib currently in use
import matplotlib.colors as mpcolors

_viridis_data = [[0.267004, 0.004874, 0.329415],
                 [0.268510, 0.009605, 0.335427],
                 [0.269944, 0.014625, 0.341379],
                 [0.271305, 0.019942, 0.347269],
                 [0.272594, 0.025563, 0.353093],
                 [0.273809, 0.031497, 0.358853],
                 [0.274952, 0.037752, 0.364543],
                 [0.276022, 0.044167, 0.370164],
                 [0.277018, 0.050344, 0.375715],
                 [0.277941, 0.056324, 0.381191],
                 [0.278791, 0.062145, 0.386592],
                 [0.279566, 0.067836, 0.391917],
                 [0.280267, 0.073417, 0.397163],
                 [0.280894, 0.078907, 0.402329],
                 [0.281446, 0.084320, 0.407414],
                 [0.281924, 0.089666, 0.412415],
                 [0.282327, 0.094955, 0.417331],
                 [0.282656, 0.100196, 0.422160],
                 [0.282910, 0.105393, 0.426902],
                 [0.283091, 0.110553, 0.431554],
                 [0.283197, 0.115680, 0.436115],
                 [0.283229, 0.120777, 0.440584],
                 [0.283187, 0.125848, 0.444960],
                 [0.283072, 0.130895, 0.449241],
                 [0.282884, 0.135920, 0.453427],
                 [0.282623, 0.140926, 0.457517],
                 [0.282290, 0.145912, 0.461510],
                 [0.281887, 0.150881, 0.465405],
                 [0.281412, 0.155834, 0.469201],
                 [0.280868, 0.160771, 0.472899],
                 [0.280255, 0.165693, 0.476498],
                 [0.279574, 0.170599, 0.479997],
                 [0.278826, 0.175490, 0.483397],
                 [0.278012, 0.180367, 0.486697],
                 [0.277134, 0.185228, 0.489898],
                 [0.276194, 0.190074, 0.493001],
                 [0.275191, 0.194905, 0.496005],
                 [0.274128, 0.199721, 0.498911],
                 [0.273006, 0.204520, 0.501721],
                 [0.271828, 0.209303, 0.504434],
                 [0.270595, 0.214069, 0.507052],
                 [0.269308, 0.218818, 0.509577],
                 [0.267968, 0.223549, 0.512008],
                 [0.266580, 0.228262, 0.514349],
                 [0.265145, 0.232956, 0.516599],
                 [0.263663, 0.237631, 0.518762],
                 [0.262138, 0.242286, 0.520837],
                 [0.260571, 0.246922, 0.522828],
                 [0.258965, 0.251537, 0.524736],
                 [0.257322, 0.256130, 0.526563],
                 [0.255645, 0.260703, 0.528312],
                 [0.253935, 0.265254, 0.529983],
                 [0.252194, 0.269783, 0.531579],
                 [0.250425, 0.274290, 0.533103],
                 [0.248629, 0.278775, 0.534556],
                 [0.246811, 0.283237, 0.535941],
                 [0.244972, 0.287675, 0.537260],
                 [0.243113, 0.292092, 0.538516],
                 [0.241237, 0.296485, 0.539709],
                 [0.239346, 0.300855, 0.540844],
                 [0.237441, 0.305202, 0.541921],
                 [0.235526, 0.309527, 0.542944],
                 [0.233603, 0.313828, 0.543914],
                 [0.231674, 0.318106, 0.544834],
                 [0.229739, 0.322361, 0.545706],
                 [0.227802, 0.326594, 0.546532],
                 [0.225863, 0.330805, 0.547314],
                 [0.223925, 0.334994, 0.548053],
                 [0.221989, 0.339161, 0.548752],
                 [0.220057, 0.343307, 0.549413],
                 [0.218130, 0.347432, 0.550038],
                 [0.216210, 0.351535, 0.550627],
                 [0.214298, 0.355619, 0.551184],
                 [0.212395, 0.359683, 0.551710],
                 [0.210503, 0.363727, 0.552206],
                 [0.208623, 0.367752, 0.552675],
                 [0.206756, 0.371758, 0.553117],
                 [0.204903, 0.375746, 0.553533],
                 [0.203063, 0.379716, 0.553925],
                 [0.201239, 0.383670, 0.554294],
                 [0.199430, 0.387607, 0.554642],
                 [0.197636, 0.391528, 0.554969],
                 [0.195860, 0.395433, 0.555276],
                 [0.194100, 0.399323, 0.555565],
                 [0.192357, 0.403199, 0.555836],
                 [0.190631, 0.407061, 0.556089],
                 [0.188923, 0.410910, 0.556326],
                 [0.187231, 0.414746, 0.556547],
                 [0.185556, 0.418570, 0.556753],
                 [0.183898, 0.422383, 0.556944],
                 [0.182256, 0.426184, 0.557120],
                 [0.180629, 0.429975, 0.557282],
                 [0.179019, 0.433756, 0.557430],
                 [0.177423, 0.437527, 0.557565],
                 [0.175841, 0.441290, 0.557685],
                 [0.174274, 0.445044, 0.557792],
                 [0.172719, 0.448791, 0.557885],
                 [0.171176, 0.452530, 0.557965],
                 [0.169646, 0.456262, 0.558030],
                 [0.168126, 0.459988, 0.558082],
                 [0.166617, 0.463708, 0.558119],
                 [0.165117, 0.467423, 0.558141],
                 [0.163625, 0.471133, 0.558148],
                 [0.162142, 0.474838, 0.558140],
                 [0.160665, 0.478540, 0.558115],
                 [0.159194, 0.482237, 0.558073],
                 [0.157729, 0.485932, 0.558013],
                 [0.156270, 0.489624, 0.557936],
                 [0.154815, 0.493313, 0.557840],
                 [0.153364, 0.497000, 0.557724],
                 [0.151918, 0.500685, 0.557587],
                 [0.150476, 0.504369, 0.557430],
                 [0.149039, 0.508051, 0.557250],
                 [0.147607, 0.511733, 0.557049],
                 [0.146180, 0.515413, 0.556823],
                 [0.144759, 0.519093, 0.556572],
                 [0.143343, 0.522773, 0.556295],
                 [0.141935, 0.526453, 0.555991],
                 [0.140536, 0.530132, 0.555659],
                 [0.139147, 0.533812, 0.555298],
                 [0.137770, 0.537492, 0.554906],
                 [0.136408, 0.541173, 0.554483],
                 [0.135066, 0.544853, 0.554029],
                 [0.133743, 0.548535, 0.553541],
                 [0.132444, 0.552216, 0.553018],
                 [0.131172, 0.555899, 0.552459],
                 [0.129933, 0.559582, 0.551864],
                 [0.128729, 0.563265, 0.551229],
                 [0.127568, 0.566949, 0.550556],
                 [0.126453, 0.570633, 0.549841],
                 [0.125394, 0.574318, 0.549086],
                 [0.124395, 0.578002, 0.548287],
                 [0.123463, 0.581687, 0.547445],
                 [0.122606, 0.585371, 0.546557],
                 [0.121831, 0.589055, 0.545623],
                 [0.121148, 0.592739, 0.544641],
                 [0.120565, 0.596422, 0.543611],
                 [0.120092, 0.600104, 0.542530],
                 [0.119738, 0.603785, 0.541400],
                 [0.119512, 0.607464, 0.540218],
                 [0.119423, 0.611141, 0.538982],
                 [0.119483, 0.614817, 0.537692],
                 [0.119699, 0.618490, 0.536347],
                 [0.120081, 0.622161, 0.534946],
                 [0.120638, 0.625828, 0.533488],
                 [0.121380, 0.629492, 0.531973],
                 [0.122312, 0.633153, 0.530398],
                 [0.123444, 0.636809, 0.528763],
                 [0.124780, 0.640461, 0.527068],
                 [0.126326, 0.644107, 0.525311],
                 [0.128087, 0.647749, 0.523491],
                 [0.130067, 0.651384, 0.521608],
                 [0.132268, 0.655014, 0.519661],
                 [0.134692, 0.658636, 0.517649],
                 [0.137339, 0.662252, 0.515571],
                 [0.140210, 0.665859, 0.513427],
                 [0.143303, 0.669459, 0.511215],
                 [0.146616, 0.673050, 0.508936],
                 [0.150148, 0.676631, 0.506589],
                 [0.153894, 0.680203, 0.504172],
                 [0.157851, 0.683765, 0.501686],
                 [0.162016, 0.687316, 0.499129],
                 [0.166383, 0.690856, 0.496502],
                 [0.170948, 0.694384, 0.493803],
                 [0.175707, 0.697900, 0.491033],
                 [0.180653, 0.701402, 0.488189],
                 [0.185783, 0.704891, 0.485273],
                 [0.191090, 0.708366, 0.482284],
                 [0.196571, 0.711827, 0.479221],
                 [0.202219, 0.715272, 0.476084],
                 [0.208030, 0.718701, 0.472873],
                 [0.214000, 0.722114, 0.469588],
                 [0.220124, 0.725509, 0.466226],
                 [0.226397, 0.728888, 0.462789],
                 [0.232815, 0.732247, 0.459277],
                 [0.239374, 0.735588, 0.455688],
                 [0.246070, 0.738910, 0.452024],
                 [0.252899, 0.742211, 0.448284],
                 [0.259857, 0.745492, 0.444467],
                 [0.266941, 0.748751, 0.440573],
                 [0.274149, 0.751988, 0.436601],
                 [0.281477, 0.755203, 0.432552],
                 [0.288921, 0.758394, 0.428426],
                 [0.296479, 0.761561, 0.424223],
                 [0.304148, 0.764704, 0.419943],
                 [0.311925, 0.767822, 0.415586],
                 [0.319809, 0.770914, 0.411152],
                 [0.327796, 0.773980, 0.406640],
                 [0.335885, 0.777018, 0.402049],
                 [0.344074, 0.780029, 0.397381],
                 [0.352360, 0.783011, 0.392636],
                 [0.360741, 0.785964, 0.387814],
                 [0.369214, 0.788888, 0.382914],
                 [0.377779, 0.791781, 0.377939],
                 [0.386433, 0.794644, 0.372886],
                 [0.395174, 0.797475, 0.367757],
                 [0.404001, 0.800275, 0.362552],
                 [0.412913, 0.803041, 0.357269],
                 [0.421908, 0.805774, 0.351910],
                 [0.430983, 0.808473, 0.346476],
                 [0.440137, 0.811138, 0.340967],
                 [0.449368, 0.813768, 0.335384],
                 [0.458674, 0.816363, 0.329727],
                 [0.468053, 0.818921, 0.323998],
                 [0.477504, 0.821444, 0.318195],
                 [0.487026, 0.823929, 0.312321],
                 [0.496615, 0.826376, 0.306377],
                 [0.506271, 0.828786, 0.300362],
                 [0.515992, 0.831158, 0.294279],
                 [0.525776, 0.833491, 0.288127],
                 [0.535621, 0.835785, 0.281908],
                 [0.545524, 0.838039, 0.275626],
                 [0.555484, 0.840254, 0.269281],
                 [0.565498, 0.842430, 0.262877],
                 [0.575563, 0.844566, 0.256415],
                 [0.585678, 0.846661, 0.249897],
                 [0.595839, 0.848717, 0.243329],
                 [0.606045, 0.850733, 0.236712],
                 [0.616293, 0.852709, 0.230052],
                 [0.626579, 0.854645, 0.223353],
                 [0.636902, 0.856542, 0.216620],
                 [0.647257, 0.858400, 0.209861],
                 [0.657642, 0.860219, 0.203082],
                 [0.668054, 0.861999, 0.196293],
                 [0.678489, 0.863742, 0.189503],
                 [0.688944, 0.865448, 0.182725],
                 [0.699415, 0.867117, 0.175971],
                 [0.709898, 0.868751, 0.169257],
                 [0.720391, 0.870350, 0.162603],
                 [0.730889, 0.871916, 0.156029],
                 [0.741388, 0.873449, 0.149561],
                 [0.751884, 0.874951, 0.143228],
                 [0.762373, 0.876424, 0.137064],
                 [0.772852, 0.877868, 0.131109],
                 [0.783315, 0.879285, 0.125405],
                 [0.793760, 0.880678, 0.120005],
                 [0.804182, 0.882046, 0.114965],
                 [0.814576, 0.883393, 0.110347],
                 [0.824940, 0.884720, 0.106217],
                 [0.835270, 0.886029, 0.102646],
                 [0.845561, 0.887322, 0.099702],
                 [0.855810, 0.888601, 0.097452],
                 [0.866013, 0.889868, 0.095953],
                 [0.876168, 0.891125, 0.095250],
                 [0.886271, 0.892374, 0.095374],
                 [0.896320, 0.893616, 0.096335],
                 [0.906311, 0.894855, 0.098125],
                 [0.916242, 0.896091, 0.100717],
                 [0.926106, 0.897330, 0.104071],
                 [0.935904, 0.898570, 0.108131],
                 [0.945636, 0.899815, 0.112838],
                 [0.955300, 0.901065, 0.118128],
                 [0.964894, 0.902323, 0.123941],
                 [0.974417, 0.903590, 0.130215],
                 [0.983868, 0.904867, 0.136897],
                 [0.993248, 0.906157, 0.143936]]
viridis = mpcolors.ListedColormap(_viridis_data, name='viridis')
plt.register_cmap(name='viridis', cmap=viridis)
mpl.rc('image', cmap='viridis')
