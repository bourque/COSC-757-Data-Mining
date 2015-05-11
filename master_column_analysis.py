#! /usr/bin/env python

"""Perform analysis on 'master column' images.

Authors:
    Matthew Bourque, May 2015
    David Borncamp, May 2015
    Arielle Leone, May 2015
    James Miller, May 2015

Use:
    This program is intended to be executed via the command line as
    such:
        >>> python master_column_analysis.py

Outputs:
    (1) results.dat - A text file containing the classification
        results.  The file contains four columns: (1) The row/pixel
        number, (2) the pixel class, (3) The classificaiton date (in
        MJD) and (4) The index of the classification date.
    (2) histogram_<row_num>.png - A histogram showing the distribution
        of pixel values for the given row/pixel and where the values
        fall within the outlier/warm/hot regimes.
    (3) scatter_<row_num>.png - A scatter plot showing the pixel values
        over time for the given row/pixel and where the values fall
        within the warm/hot regimes.
    (4) An output to the STDOUT showing the number of pixel for each
        pixel class.
"""

# Python 3 imports
from __future__ import print_function

# Built in imports
from itertools import groupby
from operator import itemgetter
import os

# Third party imports
from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from threshold_vals import get_thresh

# -----------------------------------------------------------------------------

def classify_pixel(pixel, row_num, dark_current, stdev, hot_thresh, sig, var):
    """
    Perform the algorithm to classify the given pixel.  There are 4
    classes of pixels:

    0 = Good
    1 = Warm & Stable
    2 = Hot & Stable
    3 = Unstable

    Good pixels are defined as pixels whose values remain stable and
    are not deemed a consistent outlier throughout their lifetimes.
    Warm pixels are those whose values are outliers but remain below
    the hot pixel threshold throughout their lifetimes.  Hot pixels are
    those whose values consistently exceed the hot pixel threshold
    throughout their lifetimes.  Stable pixels are those whose values
    remain relatively constant throughout their lifetimes, regardless
    if they fall within the "good", "warm", or "hot" regime.  Unstable
    pixels are those whose values vary beyond the variance threshold
    for a significant portion of their lifetimes.

    Parameters:
        pixel : numpy array
            A 1D array corresponding to a row in the master column
            image.
        row_num : int
            The row number in the column image associatied with the
            given pixel.
        dark_current : float
            The background dark current associated with the master
            column image.
        stdev : float
            The standard deviation of the background associated with
            the master column image.
        hot_thresh : float
            The threshold the defines a hot pixel (i.e. if a pixel's
            value exceeds this, it is deemed as "hot")
        sig : float
            The number of standard deviations beyond which a pixel
            shall be deemed an outlier
        var : float
            The number of standard deviations beyond which a pixel
            shall be deemed unstable.

    Returns:
        pixel_class : int
            The class that the pixel belongs to as determined by the
            algorithm. 0 = Good, 1 = Warm & Stable, 2 = Hot & Stable,
            4 = Unstable.
        starting_point : int
            The index of the pixel array in which the pixel became
            classified
    """

    # Define thresholds
    high_threshold = dark_current + (stdev * sig)
    low_threshold = dark_current - (stdev * sig)
    warm_thresh = high_threshold
    consecutive_threshold = 10
    consecutive_flag = False

    # Initialize variables
    starting_point = 0
    variance = 0
    pixel_class = 0

    # Find indices of pixels that are outliers
    outliers = np.where((pixel > high_threshold) | (pixel < low_threshold))[0]

    # If there are no outliers, the pixel is good
    if len(outliers) == 0:
        pixel_class = 0

    # If there are outliers, determine if they occur consecutively
    else:
        for k, g, in groupby(enumerate(outliers), lambda (i,x):i-x):
            consecutive_list = map(itemgetter(1), g)
            # If there are consecutive outliers that exceed the
            # consectuve_threshold, then analyze the remaining pixels
            # for stability
            if len(consecutive_list) >= consecutive_threshold:

                # Indicate that there was a consecutive outlier and determine when it happened
                consecutive_flag = True
                starting_point = consecutive_list[0]

                # Calculate variance based on first bad pixel to the end
                variance = np.std(pixel[starting_point:-1])

                # If variance exceeds variance threshold, it is unstable
                if variance > var * stdev:
                    pixel_class = 3

                break

        # If there were not enough consecutive outliers, then the pixel is good
        if consecutive_flag == False:
            pixel_class = 0

        # If the pixel class is still 0, but has consecutive outliers, then it must be an outlier that is stable
        # Thus, determine if it is warm or hot by comparing its 'beyond outlier' median to the thresholds
        if pixel_class == 0 and consecutive_flag == True:
            median = np.median(pixel[starting_point:-1])
            if median > hot_thresh:
                pixel_class = 2
            else:
                pixel_class = 1

    # Make plots pixel for visual purposes
    make_histogram(pixel, row_num, low_threshold, high_threshold, hot_thresh, pixel_class)
    make_scatter(pixel, row_num, high_threshold, hot_thresh, pixel_class)

    return pixel_class, starting_point

# -----------------------------------------------------------------------------

def get_amp(row, col):
    """
    Return the number of the amp associated with the given row and
    column number.

    Parameters:
        row : int
            The row number.
        col : int
            The column number.

    Returns:
        0, 1, 2, or 3
            The index of the threshold_dict that is associated with the
            amp.
    """

    if row >= 2070 and col <= 2103:
        return 0
    elif row >= 2070 and col > 2103:
        return 1
    elif row < 2070 and col <= 2103:
        return 2
    elif row < 2070 and col > 2103:
        return 3

# -----------------------------------------------------------------------------

def make_histogram(pixel, row_num, low_threshold, high_threshold, hot_thresh, pixel_class):
    """
    Create a histogram of the pixel for visual purposes.

    Parameters:
        pixel : numpy array
            A 1D array corresponding to a row in the master column
            image.
        row_num : int
            The row number in the column image associatied with the
            given pixel.
        low_threshold : float
            The value below which data points are considered outliers.
        high_threshold : float
            The value above which data points are considered outliers.
        hot_thresh : float
            The threshold the defines a hot pixel (i.e. if a pixel's
            value exceeds this, it is deemed as "hot")
        pixel_class : int
            The class of the pixel.

    Outputs:
        scatter_<row_num>.png - A scatter plot showing the pixel values
            over time for the given row/pixel and where the values fall
            within the warm/hot regimes.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvspan(xmin=low_threshold, xmax=high_threshold, facecolor='0.5', alpha=0.3)
    ax.axvspan(xmin=high_threshold, xmax=9999, facecolor='DarkOrange', alpha=0.3)
    ax.axvspan(xmin=hot_thresh, xmax=9999, facecolor='red', alpha=0.3)
    ax.hist(pixel, bins=100, range=(4.0,4.5), histtype='stepfilled', color='green', edgecolor='none')
    ax.set_xlim((4.0,4.5))
    ax.set_xlabel('Pixel Value (e-/s)')
    ax.set_title('Row {}: Pixel Class = {}'.format(row_num, pixel_class))
    plt.savefig('/Users/bourque/Desktop/data_mining/Project/plots/histogram_{}.png'.format(row_num))
    plt.close()

# -----------------------------------------------------------------------------

def make_scatter(pixel, row_num, high_threshold, hot_thresh, pixel_class, savename=''):
    """
    Create a scatter plot of the pixel for visual purposes.

    Parameters:
        pixel : numpy array
            A 1D array corresponding to a row in the master column
            image.
        row_num : int
            The row number in the column image associatied with the
            given pixel.
        high_threshold : float
            The value above which data points are considered outliers.
        hot_thresh : float
            The threshold the defines a hot pixel (i.e. if a pixel's
            value exceeds this, it is deemed as "hot")
        pixel_class : int
            The class of the pixel.
        savename : string
            The path to which to save the plot.

    Outputs:
        histogram_<row_num>.png - A histogram showing the distribution
            of pixel values for the given row/pixel and where the
            values fall within the outlier/warm/hot regimes.
    """

    # to conform to matt's previous saving naming scheme
    if savename == '':
        savename = '/Users/bourque/Desktop/data_mining/Project/plots/scatter_{}.png'.format(row_num)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(0,len(pixel)),pixel, marker='+', c='k', s=1.0)
    ax.set_title('Row {}: Pixel Class = {}'.format(row_num, pixel_class))
    ax.set_ylabel('Pixel value (e-/s)')
    ax.set_xlabel('Time')
    ax.axhline(high_threshold, color='orange', linewidth=2)
    ax.axhline(hot_thresh, color='r', linewidth=2)
    ax.set_xlim((0,len(pixel)))
    ax.set_ylim((4.1,4.7))
    plt.savefig(savename)
    plt.close()

# -----------------------------------------------------------------------------

def preprocess_data(data):
    """
    Convert the data from DN to e-/s and remove postflash level.

    The data in its raw form is in units of DN (data number) which is
    essentially a measure of the number of counts the pixel received
    during the observation.  However, it is easier to consider the
    pixels in units of electrons per second.

    Some observations are taken with "postflash", which is an
    additional uniform background signal (of usually 12 e-) that is
    introduced to the pixels at the time of observation.  In order to
    compare pixel values across postflash and non-postflash
    observations, the postflash signal is removed from those columns
    that correspond to observations that have postflash.  The existance
    of postflash is indicated by the FLASHLVL metadata keyword, which
    stores the number of e- introduced by the postflash.  If the
    FLASHLVL is 0, then no postflash occured.

    Parameters:
        data : numpy array
            The data column to process.

    Returns:
        data : numpy array
            The input array, only with its units converted to e-/s
            and postflash signal removed (if necessary)
    """

    # Convert data from DN to e-/s
    print('\tConverting units from DN to e-/s')
    data = (data * 1.5) / 900.0

    # Remove postflash
    print('\tRemoving postflash')
    for col in xrange(data.shape[1]):
        flashlvl = metadata['FLASHLVL'][col]
        if flashlvl > 0:
            data[:,col] = data[:,col] - (flashlvl / 900.0)

    return data

# -----------------------------------------------------------------------------

def print_summary(results_dict):
    """
    Print the results to the screen

    Parameters:
        results_dict : dict
            A dictionary whose keys are the row numbers and whose
            values are lists containing the pixel class, classification
            date, and classification date index.

    Outputs:
        Prints the number of pixels that fall under each class to the
        STDOUT.
    """

    classes = [item[0] for item in results_dict.values()]
    print('\nResults:\n')
    print('\tGood: {}'.format(len([item for item in classes if item == 0])))
    print('\tWarm & Stable: {}'.format(len([item for item in classes if item == 1])))
    print('\tHot & Stable: {}'.format(len([item for item in classes if item == 2])))
    print('\tUnstable: {}'.format(len([item for item in classes if item == 3])))

# -----------------------------------------------------------------------------

def write_results(results_dict):
    """
    Write results out to a text file.

    Parameters:
        results_dict : dict
            A dictionary whose keys are the row numbers and whose
            values are lists containing the pixel class, classification
            date, and classification date index.

    Outputs:
        results.dat - A text file containing the classification
            results.  The file contains four columns: (1) The row/pixel
            number, (2) the pixel class, (3) The classificaiton date
            (in MJD) and (4) The index of the classification date.
    """

    results_file = '/Users/bourque/Desktop/data_mining/Project/results.dat'
    with open(results_file, 'w') as results:
        results.write('# row class class_date\n')
        for item in results_dict.iteritems():
            results.write('{} {} {} {}\n'.format(item[0], item[1][0], item[1][1], item[1][2]))
    print('\nResults written to {}'.format(results_file))

# -----------------------------------------------------------------------------
# For command line execution
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Open master column image and read in data
    print('\tReading in master column image')
    master_column_image = '/Users/bourque/Desktop/data_mining/Project/master_column_test.fits'
    with fits.open(master_column_image) as hdulist:
        data = hdulist[0].data

    # Read in the metadata
    print('\tReading in metadata')
    metadata_file = '/Users/bourque/Desktop/data_mining/Project/metadata.dat'
    metadata = ascii.read(metadata_file, guess=False, delimiter=',')

    # Get dark current and stdev of first image to be used as baseline background
    first_image = metadata['path'][0]
    threshold_dict = get_thresh(first_image)

    # Convert data to e-/s and remove postflash
    data = preprocess_data(data)

    # Initialize dict to hold results
    results_dict = {}

    num_rows = data.shape[0]
    for row_num in xrange(num_rows):

        print('\tProcessing row {} of {}'.format(row_num + 1, num_rows))

        row_data = data[row_num,:]
        amp = get_amp(row_num, 500)
        dark_current = threshold_dict['dark_current'][amp]
        stdev = threshold_dict['stdev'][amp]

        # Define thresholds
        hot_thresh = dark_current + 0.1 * stdev
        sig = 0.05
        var = 1.0

        # Classify the row/pixel
        pixel_class, class_date = classify_pixel(row_data, row_num+1, dark_current, stdev, hot_thresh, sig, var)

        # Store results in results_dict
        class_expstart = metadata['EXPSTART'][class_date]
        results_dict[row_num+1] = [pixel_class, class_expstart, class_date]

    # Print summary of results
    print_summary(results_dict)

    # Write results to text file
    write_results(results_dict)
