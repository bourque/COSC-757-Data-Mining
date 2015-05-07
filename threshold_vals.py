'''
'''
from astropy.io import fits
import math
import numpy as np


def get_thresh(filename):
    '''
    Get the initial thresholds for marking images
    '''

    chip2 = fits.getdata(filename, 1) * 1.5 / 900.0
    chip1 = fits.getdata(filename, 4) * 1.5 / 900.0

    a = chip1[19:2070, 25:2072]
    b = chip1[19:2070, 2130:4178]
    c = chip2[19:2070, 25:2072]
    d = chip1[19:2070, 2130:4178]

    meda = np.median(a)
    astdev = np.std(a)
    medb = np.median(b)
    bstdev = np.std(b)
    medc = np.median(c)
    cstdev = np.std(c)
    medd = np.median(d)
    dstdev = np.std(d)

    threshdict = {'dark_current': [meda, medb, medc, medd],
                  'stdev': [astdev, bstdev, cstdev, dstdev]}

    return threshdict


__iterMax = 25
__delta = 5.0e-7
__epsilon = 1.0e-20


def resistant_mean(inputData, Cut=3.0):
    """
    Robust estimator of the mean of a data set.  Based on the
    resistant_mean function from the AstroIDL User's Library.

    .. seealso::
    :func:`lsl.misc.mathutil.robustmean`

    Ported by Dave Borncamp

    """

    data = inputData.ravel()
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()

    data0 = np.median(data)
    maxAbsDev = np.median(np.abs(data - data0)) / 0.6745
    if maxAbsDev < __epsilon:
        maxAbsDev = (np.abs(data - data0)).mean() / 0.8000

    cutOff = Cut * maxAbsDev
    good = np.where(np.abs(data - data0) <= cutOff)
    good = good[0]
    dataMean = data[good].mean()
    dataSigma = math.sqrt(((data[good] - dataMean) ** 2.0).sum() / len(good))

    if Cut > 1.0:
        sigmaCut = Cut
    else:
        sigmaCut = 1.0
    if sigmaCut <= 4.5:
        dataSigma = dataSigma / (-0.15405 + 0.90723 * sigmaCut - 0.23584 * sigmaCut ** 2.0 + 0.020142 * sigmaCut ** 3.0)

    cutOff = Cut * dataSigma
    good = np.where(np.abs(data - data0) <= cutOff)
    good = good[0]
    dataMean = data[good].mean()
    if len(good) > 3:
        dataSigma = math.sqrt(((data[good] - dataMean) ** 2.0).sum() / len(good))

    if Cut > 1.0:
        sigmaCut = Cut
    else:
        sigmaCut = 1.0
    if sigmaCut <= 4.5:
        dataSigma = dataSigma / (-0.15405 + 0.90723 * sigmaCut - 0.23584 * sigmaCut ** 2.0 + 0.020142 * sigmaCut ** 3.0)

    dataSigma = dataSigma / math.sqrt(len(good) - 1)

    return dataMean
