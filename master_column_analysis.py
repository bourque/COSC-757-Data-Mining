#! /usr/bin/env python

"""Perform analysis on 'master column' images.

Authors:
    Matthew Bourque, April, 2015

Use:

Outputs:

"""

from __future__ import print_function

import os

from astropy.io import fits
from astropy.io import ascii
import numpy as np

from threshold_vals import get_thresh

# -----------------------------------------------------------------------------

def classify_pixel(pixel, treshold_dict):
    """
    0 = Good
    1 = Warm & Stable
    2 = Warm & Unstable
    3 = Hot & Stable
    4 = Hot & Unstable
    """

    if pixel > threshold_dict['dark_current'] + (3 * stdev):
        results_array[row,:] = 1
    elif pixel < threshold_dict['dark_current'] - (3 * stdev):
        results_array[row,:] = 2

    return pixel_class

# -----------------------------------------------------------------------------

def get_amp(row, col):
    """
    Return the number of the amp associated with the given row and
    column number
    """

    if row >= 2070 and col <= 2103:
        return 0
    elif row >= 2070 and col > 2103:
        return 1
    elif row < 2070 and col <= 2103:
        return 2
    elif row < 2070 and col > 2103:
        return 3

    return amp

# -----------------------------------------------------------------------------
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

    # Set dark current and thresholds
    first_image = metadata['path'][0]
    threshold_dict = get_thresh(first_image)

    # Initialize results array that will hold classes
    results_array = np.zeros((data.shape[0], 1))

    # Iterate over columns and compute statistics
    number_columns = data.shape[1]
    for col in xrange(number_columns):

        print('Processing column {} of {}'.format(col+1, number_columns))
        column = data[:,col]

        for row in xrange(len(column)):
            pixel = column[row]
            amp_num = get_amp(row, 500)

            pixel_class = classify_pixel(pixel, threshold_dict)
            result_array[row,:] = pixel_class
