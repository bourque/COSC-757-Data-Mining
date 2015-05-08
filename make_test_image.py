#! /usr/bin/env python

"""Create test image that is a subset of a master_column image.

Authors:
    Matthew Bourque, April, 2015

Use:

Outputs:

"""

from __future__ import print_function
import os
from astropy.io import fits


if __name__ == '__main__':

    # Open master column image and read in data
    print('\tReading in master column image')
    master_column_image = '/Users/bourque/Desktop/data_mining/Project/master_column_500.fits'
    with fits.open(master_column_image) as hdulist:
        data = hdulist[0].data

    # Subset the data
    data = data[0:2069,:]

    # Save the test data to new image for visualization purposes
    print('\tWriting out test image')
    test_file = '/Users/bourque/Desktop/data_mining/Project/master_column_test.fits'
    if not os.path.exists(test_file):
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(test_file)