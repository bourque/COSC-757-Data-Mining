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

    # Subset the data for testing purposes
    data = data[0:2069,3000:3500]

    # Save the test data to new image for visualization purposes
    print('\tWriting out test image')
    test_file = '/Users/bourque/Desktop/data_mining/Project/master_column_test.fits'
    if not os.path.exists(test_file):
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(test_file)

    # Initialize results array
    results_array = np.zeros((2070, 1))

    # Iterate over columns and compute statistics
    number_columns = data.shape[1]
    for col in xrange(number_columns):

        print('Processing column {} of {}'.format(col+1, number_columns))
        column = data[:,col]
        median = np.median(column)
        stdev = np.std(column)

        for row in xrange(len(column)):
            pixel = column[row]

            # If pixel is above 3 sigma, flag as hot
            if pixel > median + (3 * stdev):
                results_array[row,:] = 1

    print(results_array)