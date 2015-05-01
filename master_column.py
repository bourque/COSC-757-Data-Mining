#! /usr/bin/env python

"""Creates 'master column' FITS images from extracted columns of WFC3/UVIS or
WFC3/IR data.

'Master column' images are FITS images wherein each column is an extracted
column from a WFC3/UVIS or WFC3/IR image for a specific column.  For example,
a 'master column' image of the 500th column of WFC3/UVIS would contain every
500th column of the UVIS dataset over time.  This script allows the user to
provide a Quicklook database query to specify the dataset they wish to create
a master column image for (see Use).

Authors:
    Matthew Bourque, Janurary 2015

Use:
    The user must provide four command line arguements with the option
    of three additional arguments:

    >>> python master_column.py [-d|--detector] [-f|--fitstype]
        [-q|--query_file] [-m|--metadata_columns] [-o|--ouput_path]
        [-c|--column] [-n|--num_cores]

    -d --detector (required) - The detector from which to make the
        master column image.  Must be 'ir' or 'uvis'.

    -f --fitstype (required) - The FITS type of the images to process.
        Must be 'raw', 'flt', or 'ima'.  Note that 'ima' is only a
        valid option of detector is 'ir'.

    -i --image_file (required) - The path to a file containing either
        a list of image paths or a Quicklook database query that will
        return a list of image paths.  If the QL database is to be
        used, this file must be called "query.dat" (see NOTES).

The path to a text file containing either a list' + \
        ' of image paths or a Quicklook database query that will return a' + \
        ' list of image paths.  If the QL database is to be used, this ' + \
        ' file must be called "query.dat".'

    -m --metadata_columns (optional) - The path to a file containing
        Quicklook database header column values to extract (see Notes).
        If not provided, no metadata will be extracted.

    -o --output_path (optional) - The path to which the output products
        will be written.  The default value is the current working
        directory.

    -c --column (optional) - The specific column to extract.  The
        default value is "all", in which master column images for all
        columns will be made.

    -n --num_cores (optional) - The number of cores to use during
        multiprocessing.  The default value is 1.

Outputs:
    (1) metadata.dat - A tab-separated file containing metadata
            describing each column in the master column image (if
            the metadata_columns argument is used).
    (2) master_column_<col>.fits - The master column FITS image, where
            <col> is the particular UVIS or IR column.

Notes:
    The user must provide the following input file:

        A text file containing either a list of image paths (each on a
        separate line) or a query to the quicklook database which will
        be used to determine the dataset.  For the case of the query,
        the entire query must reside on the first line of the text file.
        The query must SELECT only the Master.dir and Master.filename
        columns, as well as ORDER BY the expstart in order to ensure
        proper time series analysis.  For example:

        SELECT MASTER.dir, MASTER.filename
        FROM MASTER
        JOIN IR_FLT_0 ON MASTER.id = IR_FLT_0.id
        WHERE IR_FLT_0.SUBTYPE = 'FULLIMAG'
        AND IR_FLT_0.FILTER = 'Blank'
        AND IR_FLT_0.SAMP_SEQ = 'SPARS200'
        AND IR_FLT_0.NSAMP = '16'
        ORDER BY IR_FLT_0.EXPSTART

    The user may also provide an optional input file:

        A text file containing header values to extract and place into
        the output metadata file.  Each header value must be on its own
        line in the file and must be a valid header keyword.  For
        example:

        targname
        detector
        exptime
        expstart
        aperture
"""

import argparse
import glob
import itertools
import multiprocessing
import os

from astropy.io import ascii
from astropy.io import fits
import numpy as np
import sqlite3

# -----------------------------------------------------------------------------

def get_data(hdulist, column, detector):
    """Return the FITS SCI data from the hdulist for the specific column.

    Parameters:
        hdulist : astropy HDUList object
            The hdulist of the image.
        column : int
            The column to extract.
        detector : string
            The detector of the image.  Can be 'uvis' or 'ir'.

    Returns:
        data : numpy array
            The data for the specific column.

    Outputs:
        nothing

    Notes:
        Since UVIS data is spread over two extensions (1 and 4), UVIS
        data is concatenated into a single array before it is returned.
    """

    # If the data is UVIS, combine extensions 1 and 4
    if detector == 'uvis':
        ext1_data = hdulist[1].data[:, column:column + 1]
        ext4_data = hdulist[4].data[:, column:column + 1]
        data = np.concatenate((ext1_data, ext4_data), axis=0)
    elif detector == 'ir':
        data = hdulist[1].data[:, column:column + 1]

    return data

# -----------------------------------------------------------------------------

def initialize_master_column_array(detector, fitstype):
    """Initialize an empty master column array with a y-axis size
    determined by the detector and fitstype.

    Parameters:
        detector : string
            The detector of the image.  Can be 'uvis' or 'ir'.
        fitstype : string
            The FITS type of the image.  Can be 'raw', 'flt', or 'ima'.

    Returns:
        master_column : numpy array
            A 0 x <yaxis> numpy array to be used as the master column
            array.

    Outputs:
        nothing
    """

    if detector == 'uvis':
        if fitstype == 'raw':
            yaxis = 4140
        elif fitstype == 'flt':
            yaxis = 4102
    elif detector == 'ir':
        if fitstype == 'raw':
            yaxis = 1024
        elif fitstype == 'flt':
            yaxis = 1014
        if fitstype == 'ima':
            yaxis = 1024

    master_column = np.empty((yaxis, 0))

    return master_column

# -----------------------------------------------------------------------------

def query_for_files(query, fitstype):
    """Return a list of file paths meeting the criteria of the given
    query.

    Parameters:
        query : string
            A Quicklook database query that selects filenames and paths
            of either UVIS or IR data to form a dataset from which to
            create a master column image.
        fitstype : string
            The FITS type of the image.  Can be 'raw', 'flt', or 'ima'.

    Returns:
        image_paths : list
            A list of paths to files from which to create a master
            column image.

    Outputs:
        nothing

    Notes:
        The user must SELECT only MASTER.dir and MASTER.filename from
        the QL database and use a condition that selects only data
        from a particular detector (i.e. WHERE MASTER.detector = "IR").
        The user is also encouraged to ORDER BY expstart so that the
        dataset is sorted by time (See module Notes).
    """

    print '\nQuerying for files using the following query:\n'
    print query

    # Open database connection
    conn = sqlite3.connect('/grp/hst/wfc3a/Database/ql.db')
    conn.text_factory = str
    db_cursor = conn.cursor()

    # Execute query
    db_cursor.execute(query)

    # Parse results
    results = db_cursor.fetchall()
    assert len(results) > 0, 'Query did not yield any resuts.'
    image_paths = []
    for result in results:
        path = result[0]
        filename = result[1].replace('flt.fits', '{}.fits'.format(fitstype))
        image_path = os.path.join(path, filename)
        image_paths.append(image_path)
    print '{} files found.'.format(len(image_paths))

    # Close database connection
    conn.close()

    return image_paths

# -----------------------------------------------------------------------------

def process(args):
    """Create a WFC3/UVIS master column image.

    Parameters:
        args : tuple-like
            The function arguments provided by mp_args (multiprocessing
            args.  The 0th element is the IR column to process, the 1st
            element is a list of the image paths, the 2nd element
            is the output path, the 3rd element is the metadata columns,
            the 4th element is the detector, and the fifth element is
            the FITS type.

    Returns:
        nothing

    Outputs:
        (1) metadata.dat - A tab-separated file containing metadata
                describing each column in the master column image.
        (2) master_column_<col>.fits - The master column FITS image,
                where <col> is the particular column.
    """

    # Parse arguments
    column = args[0]
    image_paths = args[1]
    output_path = args[2]
    metadata_cols = args[3]
    detector = args[4]
    fitstype = args[5]

    # Initializations
    metadata_dicts = []
    column_number = 0
    num_images = len(image_paths)
    metadata_file = os.path.join(output_path, 'metadata.dat')
    master_column = initialize_master_column_array(detector, fitstype)

    # For each image, extract the column and place it in the master column image
    print '\nExtracting column {}.'.format(column)
    for image in image_paths:

        print '\tProcessing {}: {}/{}'.format(image, column_number + 1, num_images)

        # Get data and append it to the master_column array
        hdulist = fits.open(image, 'readonly')
        data = get_data(hdulist, column, detector)
        master_column = np.append(master_column, data, axis=1)

        # Get header keywords to place in the metadata file if necessary
        if metadata_cols != None:
            if not os.path.exists(metadata_file):
                metadata_dict = {}
                metadata_dict['path'] = image
                metadata_dict['column'] = column_number + 1
                for metadata_col in metadata_cols:
                    try:
                        metadata_dict[metadata_col] = hdulist[0].header[metadata_col]
                    except:
                        metadata_dict[metadata_col] = 'NULL'
                metadata_dicts.append(metadata_dict)

        column_number += 1

    # Write data out to master column image
    print 'Writing output image.'
    write_image(master_column, os.path.join(output_path, 'master_column_{}.fits'.format(str(column))))

    # Write metadata out to metadata file if necessary
    if metadata_cols != None:
        if not os.path.exists(metadata_file):
            write_metadata(metadata_dicts, metadata_file)

# -----------------------------------------------------------------------------

def remove_columns(output_path, col_range):
    """Remove columns from the col_range that already have an existing
    output product.

    Parameters:
        output_path : string
            The path to the directory in which output products exist.
        col_range : list
            A list of integers specifying the columns to process.

    Returns:
        new_col_range : list
            A list of integers specifying the columns to process, with
            only columns that do not already have a corresponding
            output product.

    Outputs:
        nothing
    """

    # Get list of columns that already have output products
    output_products = glob.glob(os.path.join(output_path, 'master_column_*.fits'))
    output_products = [os.path.basename(item) for item in output_products]
    existing_cols = [int(item.split('.fits')[0].split('_')[-1]) for item in output_products]
    print len(existing_cols)

    new_col_range = [col for col in col_range if col not in existing_cols]

    return new_col_range

# -----------------------------------------------------------------------------

def write_image(image, output_loc):
    """Write out the master column image to a given output location.

    Parameters:
        image : numpy array
            The master column array to be written out.
        output_loc : string
            The path to which the master column image will be written.

    Returns:
        nothing

    Outputs:
        master_column_<col>.fits - The master column image.
    """

    # Overwrite if it already exists
    if os.path.exists(output_loc):
        os.remove(output_loc)

    hdu = fits.PrimaryHDU(image)
    hdu.writeto(output_loc)

    print '\tOutput image written to {}'.format(output_loc)

# -----------------------------------------------------------------------------

def write_metadata(metadata, metadata_file):
    """Write out a tab-separated file containing metadata of the master
    column image.

    Parameters:
        metadata : list
            A list in which each item is a dictionary whose keys are
            column headers and whose values are metadata values.
        metadata_file : string
            The path to the output location of the metadata file.

    Returns:
        nothing

    Outputs:
        metadata.dat - The tab-separated file containing metadata.
    """

    if not os.path.exists(metadata_file):
        print 'Creating metadata output file.'
        ascii.write(metadata, metadata_file, delimiter=',')
        print '\tOutput file written to {}'.format(metadata_file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments. Returns args object.

    Parameters:
        nothing

    Returns:
        args : argparse.Namespace object
            An argparse object containing all of the added arguments.

    Outputs:
        nothing
    """

    # Create help strings
    detector_help = 'The DETECTOR of the images to process. Can be "IR" ' + \
        'or "UVIS".'
    fitstype_help = 'The FITS type of the images to process. Can be "flt",' + \
        ' "raw", or "ima".'
    image_file_help = 'The path to a text file containing either a list' + \
        ' of image paths or a Quicklook database query that will return a' + \
        ' list of image paths.  If the QL database is to be used, this ' + \
        ' file must be called "query.dat".'
    metadata_columns_help = 'The path to the text file containing the ' + \
        'header keywords to extract for the metadata file.  If not ' + \
        'provided, no output metadata will be written.'
    output_path_help = 'The path in which the output products will be ' + \
        'written.  If not provided, the output path will be the current ' + \
        'working directory.'
    column_help = 'The column to extract.  If not provided, master ' + \
        'column images will be made for every column.'
    num_cores_help = 'The number of cores to use during processing.  If ' + \
        'not provided, only one core will be used.'

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector', type=str, help=detector_help,
        action='store', required=True)
    parser.add_argument('-f', '--fitstype', type=str, help=fitstype_help,
        action='store', required=True)
    parser.add_argument('-i', '--image_file', type=str, help=image_file_help,
        action='store', required=True)
    parser.add_argument('-m', '--metadata_columns', type=str,
        help=metadata_columns_help, action='store', required=False,
        default='none')
    parser.add_argument('-o', '--output_path', type=str,
        help=output_path_help, action='store', required=False,
        default=os.getcwd())
    parser.add_argument('-c', '--column', type=str, help=column_help,
        action='store', required=False, default='all')
    parser.add_argument('-n', '--num_cores', type=int, help=num_cores_help,
        action='store', required=False, default=1)

    # Parse args
    args = parser.parse_args()

    return args

# -----------------------------------------------------------------------------

def test_args(args):
    """Ensures valid command line arguments.

    Paramters:
        args : argparse.Namespace object
            An argparse object containing all of the added arguments.

    Returns:
        nothing

    Outputs:
        nothing
    """

    # Ensure the detector is either UVIS or IR
    assert args.detector.lower() in ['uvis', 'ir'], \
        'Detector argument must be "uvis" or "ir".'

    # Ensure that the fitstype is valid
    assert args.fitstype.lower() in ['raw', 'flt', 'ima'], \
        'FITS type argument must be "raw", "flt", or "ima".'

    # Ensure that the detector/fitstype combination is valid.
    valid_combos = ['uvis/raw', 'uvis/flt', 'ir/raw', 'ir/flt', 'ir/ima']
    combo = '{}/{}'.format(args.detector.lower(), args.fitstype.lower())
    assert combo in valid_combos, \
        '{}/{} is not a valid detector/FITS type combination'.format(
            args.detector, args.fitstype)

    # Ensure the image file exists
    assert os.path.exists(args.image_file), \
        '{} does not exist.'.format(args.image_file)

    # Ensure the metadata columns file exists
    if args.metadata_columns != 'none':
        assert os.path.exists(args.metadata_columns), \
            '{} does not exist.'.format(args.metadata_columns)

    # Ensure the output path exists
    assert os.path.exists(args.output_path), \
        '{} does not exist.'.format(args.output_path)

# -----------------------------------------------------------------------------

def master_column_main(detector, fitstype, image_file, metadata_columns, output_path, column, num_cores):
    """The main function of the master_column module.

    Parameters:
        detector : string
            The detector. Must be "ir" or "uvis".
        fitstype : string
            The FITS type of the image.  Can be 'raw', 'flt', or 'ima'.
        image_file : string
            The path to the file containing the dataset or database
            query.
        metadata_columns : string
            The path to the file containing the QL database columns to
            extract.
        output_path : string
            The path to the directory in which output products will be
            written.
        column : string
            The specific column to extract.  Can also be "all" (to
            extract all columns).
        num_cores : int
            The number of cores used for multiprocessing.

    Returns:
        nothing

    Outputs:
        (1) metadata.dat - A tab-separated file containing metadata
                describing each column in the master column image.
        (2) master_column_<col>.fits - The master column FITS image,
                where <col> is the particular column.
    """

    # Determine column range
    if column == 'all':
        if detector == 'uvis' and fitstype == 'raw':
            col_range = range(4207)
        elif detector == 'uvis' and fitstype == 'flt':
            col_range = range(4097)
        elif detector == 'ir' and fitstype == 'raw':
            col_range = range(1024)
        elif detector == 'ir' and fitstype == 'flt':
            col_range = range(1014)
        elif detector == 'ir' and fitstype == 'ima':
            col_range = range(1024)

        # Remove columns that already have output products
        col_range = remove_columns(output_path, col_range)

    else:
        col_range = int(column)

    # Read in the columns to extract from the metadata_columns file
    if metadata_columns == 'none':
        metadata_cols = None
    else:
        with open(metadata_columns) as mfile:
            metadata_cols = mfile.readlines()
            metadata_cols = [item.strip().upper() for item in metadata_cols]

    # Determine the image paths from the image_file
    if os.path.basename(image_file) == 'query.dat':
        with open(image_file) as qfile:
            query = qfile.readlines()[0].strip()
        image_paths = query_for_files(query, fitstype)
    else:
        with open(image_file) as dfile:
            dataset = dfile.readlines()
        image_paths = [item.strip() for item in dataset]

    # Perform multiprocessing if necessary
    if type(col_range) is list:
        pool = multiprocessing.Pool(processes=num_cores)
        mp_args = itertools.izip(
            col_range,
            itertools.repeat(image_paths),
            itertools.repeat(output_path),
            itertools.repeat(metadata_cols),
            itertools.repeat(detector),
            itertools.repeat(fitstype))
        pool.map(process, mp_args)
        pool.close()
        pool.join()
    elif type(col_range) is int:
        process((col_range, image_paths, output_path, metadata_cols, detector, fitstype))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    args = parse_args()
    test_args(args)

    master_column_main(args.detector.lower(), args.fitstype.lower(),
        args.image_file, args.metadata_columns, args.output_path,
        args.column, args.num_cores)
