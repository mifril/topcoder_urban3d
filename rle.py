import numpy as np
import glob
import os
import cv2
import scipy.misc as scm
from dask import delayed, threaded, compute

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
MAIN_DIR = 'C://data//'
TRAIN_DIR = MAIN_DIR + 'training/'
TEST_DIR = MAIN_DIR + 'testing/'
PRED_DIR = OUTPUT_DIR + 'predictions/'

IMG_H = 2048
IMG_W = 2048

BACKGROUND_LABEL = 2

N_TRAIN_ALL = 174
N_FOLDS = 6
N_TRAIN = 145
N_VAL = 29

# def rle_encode(mask_image):
#     pixels = mask_image.flatten()
#     # We avoid issues with '1' at the start or end (at the corners of 
#     # the original image) by setting those pixels to '0' explicitly.
#     # We do not expect these to be non-zero for an accurate mask, 
#     # so this should not harm the score.
#     pixels[0] = 0
#     pixels[-1] = 0
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#     return runs

# def rle_to_string(runs):
#     return ' '.join(str(x) for x in runs)

# def rle(img):
#     img = cv2.resize(img.astype(np.uint8).reshape(img.shape[0], img.shape[1]), (IMG_W, IMG_H))
#     return rle_to_string(rle_encode(img))

def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]

    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values


def rldecode(starts, lengths, values, minlength=None):
    """
    Decode a run-length encoding of a 1D array.

    Parameters
    ----------
    starts, lengths, values : 1D array_like
        The run-length encoding.
    minlength : int, optional
        Minimum length of the output array.

    Returns
    -------
    1D array. Missing data will be filled with NaNs.

    """
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    ends = starts + lengths
    n = ends[-1]
    if minlength is not None:
        n = max(minlength, n)
    x = np.full(n, np.nan)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x


def rle_to_string(rle):
    (starts, lengths, values) = rle
    items = []
    for i in range(len(starts)):
        items.append(str(values[i]))
        items.append(str(lengths[i]))
    return ",".join(items)
