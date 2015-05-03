#! /usr/bin/env python

"""Perform analysis on 'master column' images.

Authors:
    Matthew Bourque, May 2015
    David Borncamp, May 2015

Use:

Outputs:

"""

from __future__ import print_function

from itertools import groupby
from operator import itemgetter
import os

from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

from threshold_vals import get_thresh

# -----------------------------------------------------------------------------

def classify_pixel(pixel, dark_current, stdev, warm_pixel_threshold, hot_pixel_threshold, thresh_sig=3, thresh_var=0.5):
    """
    Perform the algorithm to classify the given pixel.  There are 4
    classes of pixels:

    0 = Good
    1 = Warm & Stable
    2 = Hot & Stable
    3 = Unstable

    Good pixels are defined as pixels whose values remain below the
    warm and hot pixel thresholds throughout their lifetime.  Warm
    pixels are defined as those that have values that fall above the
    warm pixel threshold and fall below the hot pixel threshold.  Hot
    pixels are defined as those that have values that exceed the hot
    pixel threshold limit.

    Parameters:
        pixel : numpy array
            A 1D array corresponding to a row in the master column
            image.
        dark_current : float
            The background dark current associated with the master
            column image.
        stdev : float
            The standard deviation of the background associated with
            the master column image.
        warm_pixel_threshold : float
            The threshold the defines a warm pixel (i.e. if a pixel's
            value exceeds this, it is deemed as "warm")
        hot_pixel_threshold : float
            The threshold the defines a hot pixel (i.e. if a pixel's
            value exceeds this, it is deemed as "hot")
        thresh_sig : float
            The number of standard deviations beyond which a pixel
            shall be deemed an outlier
        thresh_var : float
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
    high_threshold = dark_current + (stdev * thresh_sig)
    low_threshold = dark_current - (stdev * thresh_sig)
    consecutive_threshold = 10

    # Initialize variables
    starting_point = 0
    varience = 0
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

                # Calculate varience based on first bad pixel to the end
                starting_point = consecutive_list[0]
                varience = np.var(pixel[starting_point:-1])
                assert varience != 0, 'Varience is 0'

                # If varience exceeds varience threshold, it is unstable
                if varience > thresh_var * stdev:
                    pixel_class = 3

                break

        # If pixel is not varying, determine if hot or warm
        median = np.median(pixel[starting_point:-1])
        if varience == 0:
            if median < warm_pixel_threshold:
                assert starting_point == 0, 'Starting point for good pixel is not 0'
                pixel_class = 0
            elif median >= warm_pixel_threshold and median < hot_pixel_threshold:
                pixel_class = 1
            else:
                pixel_class = 2
        elif pixel_class != 3:
            if median >= warm_pixel_threshold and median < hot_pixel_threshold:
                pixel_class = 1
            elif median >= hot_pixel_threshold:
                pixel_class = 2
            else:
                print("Noooooooo")

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
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Open master column image and read in data
    print('\tReading in master column image')
    master_column_image = '/Users/bourque/Desktop/data_mining/Project/master_column_500.fits'
    with fits.open(master_column_image) as hdulist:
        data = hdulist[0].data

    # Read in the metadata
    print('\tReading in metadata')
    metadata_file = '/Users/bourque/Desktop/data_mining/Project/metadata.dat'
    metadata = ascii.read(metadata_file, guess=False, delimiter=',')

    # Get dark current and stdev of first image to be used as baseline background
    first_image = metadata['path'][0]
    threshold_dict = get_thresh(first_image)

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
        thresh_sig = 0.2
        thresh_var = 1.0
        warm_pixel_threshold = 2510
        hot_pixel_threshold = 2520

        # Classify the row/pixel
        pixel_class, class_date = classify_pixel(
            row_data,
            dark_current, stdev,
            warm_pixel_threshold,
            hot_pixel_threshold,
            thresh_sig,
            thresh_var)

        # Store results in results_dict
        class_date = metadata['EXPSTART'][class_date]
        results_dict[row_num] = [pixel_class, class_date]

    # Write results to text file
    #with open('/Users/bourque/Desktop/data_mining/Project/results.dat') as results_file:
    for item in results_dict.iteritems():
        print(item)
