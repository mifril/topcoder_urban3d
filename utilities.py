import numpy as np
import pandas as pd
import glob
import os
import cv2
from skimage import io
import scipy.misc as scm
from sklearn.model_selection import KFold

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
N_TRAIN = 5 * int(N_TRAIN_ALL / N_FOLDS)
N_VAL = int(N_TRAIN_ALL / N_FOLDS)
N_TEST = 62
EPS = 1e-12

RS = 17

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def center_crop(x, mask, center_crop_size=1024):
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size // 2, center_crop_size // 2
    return x[centerw - halfw : centerw + halfw, centerh - halfh : centerh + halfh, :], mask[centerw - halfw : centerw + halfw, centerh - halfh : centerh + halfh]

def get_val_crops(img, mask, tile_size=1024, start=512):
    tiles = []
    masks = []
    for i in range(start, IMG_W-start, tile_size):
        for j in range(start, IMG_W-start, tile_size):
            tiles.append(img[i : i + tile_size,\
                             j : j + tile_size, :])
            if mask is not None:
                masks.append(mask[i : i + tile_size,\
                                  j : j + tile_size])
    return np.array(tiles), np.array(masks)

def random_crop(x, mask, random_crop_size=1024):
    w, h = x.shape[0], x.shape[1]
    # print (x.shape, w ,h)
    rangew = (w - random_crop_size)
    rangeh = (h - random_crop_size)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[offsetw : offsetw + random_crop_size, offseth : offseth + random_crop_size, :], mask[offsetw : offsetw + random_crop_size, offseth : offseth + random_crop_size]

def prepare_folds():
    names = glob.glob(os.path.join(TRAIN_DIR, '*_RGB.tif'))
    names = np.array([name.split('_RGB')[0].split('\\')[1] for name in names])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)
    folds = []
    for _, test_index in kf.split(names):
        folds.append(names[test_index])
    return folds

def get_tiles(img, mask=None, tile_size=1024):
    shift = int(tile_size / 2)
    tiles = []
    masks = []
    for i in range(int(IMG_W / shift - 1)):
        for j in range(int(IMG_W / shift - 1)):
            tiles.append(img[i*shift : i*shift + tile_size, j*shift : j*shift + tile_size, :])
            if mask is not None:
                masks.append(mask[i*shift : i*shift + tile_size, j*shift : j*shift + tile_size, :])
    if mask is not None:
        return np.array(tiles), np.array(masks)
    else:
        return np.array(tiles)

def get_mean_std(cur_val_fold=N_FOLDS - 1):
    mean_std_df = pd.read_csv(INPUT_DIR + 'mean_std.csv');
    img_mean = mean_std_df['mean_no_fold_{}'.format(cur_val_fold)].values
    img_std = mean_std_df['std_no_fold_{}'.format(cur_val_fold)].values
    return img_mean, img_std

def dump_imgs(img, mask, imgname, dirname='tmp', tile_id=''):
    path = os.path.join(OUTPUT_DIR, dirname)
    if not os.path.exists(path):
        os.makedirs(path)
    f = os.path.join(path, '{}{}.png'.format(imgname, tile_id))
    cv2.imwrite(f, img)
    f = os.path.join(path, '{}{}_mask.png'.format(imgname, tile_id))
    cv2.imwrite(f, mask * 255)
