import numpy as np
import pandas as pd
import glob
import os
import cv2
from skimage import io
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
N_TRAIN = 116
N_VAL = 58
N_TEST = 62
N_FOLDS = 3
EPS = 1e-12

def center_crop(x, mask, center_crop_size=1024):
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size // 2, center_crop_size // 2
    return x[centerw - halfw : centerw + halfw, centerh - halfh : centerh + halfh, :], mask[centerw - halfw : centerw + halfw, centerh - halfh : centerh + halfh]

def random_crop(x, mask, random_crop_size=1024):
    w, h = x.shape[0], x.shape[1]
    # print (x.shape, w ,h)
    rangew = (w - random_crop_size) // 2
    rangeh = (h - random_crop_size) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[offsetw : offsetw + random_crop_size, offseth : offseth + random_crop_size, :], mask[offsetw : offsetw + random_crop_size, offseth : offseth + random_crop_size]

def prepare_folds():
    names = glob.glob(os.path.join(TRAIN_DIR, '*_RGB.tif'))
    names = [name.split('_RGB')[0].split('\\')[1] for name in names]
    folds = [names[i:i + N_VAL] for i in range(N_FOLDS)]
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

def get_tiles_4(img, mask, tile_size=1024):
    tiles = []
    masks = []
    for i in range(int(IMG_W / tile_size)):
        for j in range(int(IMG_W / tile_size)):
            tiles.append(img[i*tile_size : (i+1) * tile_size, j*tile_size : (j+1)*tile_size, :])
            if mask is not None:
                masks.append(mask[i*tile_size : (i+1) * tile_size, j*tile_size : (j+1)*tile_size, :])
    return np.array(tiles), np.array(masks)

def get_mean_std(cur_val_fold=N_FOLDS - 1):
    mean_std_df = pd.read_csv(INPUT_DIR + 'mean_std.csv');
    img_mean = mean_std_df['mean_no_fold_{}'.format(cur_val_fold)].values
    img_std = mean_std_df['std_no_fold_{}'.format(cur_val_fold)].values
    return img_mean, img_std

def train_generator(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1, validate=True):
    if validate:
        folds = prepare_folds()
        train_names = [name for fold in folds[0:cur_val_fold] + folds[cur_val_fold + 1:] for name in fold]
    else:
        train_names = [name for fold in folds for name in fold]
    img_mean, img_std = get_mean_std(cur_val_fold=cur_val_fold)
    while True:
        for start in range(0, len(train_names), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_names))
            names_train_batch = train_names[start:end]
            for name in names_train_batch:
                img = cv2.imread(os.path.join(TRAIN_DIR, '{}_RGB.tif'.format(name)))
                # img = (img - img_mean) / (img_std + EPS)
                mask = io.imread(os.path.join(TRAIN_DIR, '{}_GTI.tif'.format(name)))
                mask = np.array(mask != 0, np.uint8)
               
                mask_crop = [0]
                while (np.sum(mask_crop) == 0):
                    img_crop, mask_crop = random_crop(img, mask, img_h)

                # if not os.path.exists('../output/train_imgs'):
                #     os.makedirs('../output/train_imgs')
                # f = os.path.join('../output/train_imgs', '{}.png'.format(name))
                # cv2.imwrite(f, img_crop)
                # f = os.path.join('../output/train_imgs', '{}_mask.png'.format(name))
                # cv2.imwrite(f, mask_crop * 255)

                mask_crop = np.expand_dims(mask_crop, axis=2)
                x_batch.append(img_crop)
                y_batch.append(mask_crop)

            x_batch = np.array(x_batch, np.float32) / 255.0
            y_batch = np.array(y_batch, np.float32)
            
            yield x_batch, y_batch

def val_generator(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1):
    folds = prepare_folds()
    val_names = [name for name in folds[cur_val_fold]]
    img_mean, img_std = get_mean_std(cur_val_fold=cur_val_fold)
    while True:
        for start in range(0, len(val_names), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(val_names))
            names_valid_batch = val_names[start:end]
            for name in names_valid_batch:
                img = cv2.imread(os.path.join(TRAIN_DIR, '{}_RGB.tif'.format(name)))
                # img = (img - img_mean) / (img_std + EPS)
                mask = io.imread(os.path.join(TRAIN_DIR, '{}_GTI.tif'.format(name)))
                mask = np.array(mask != 0, np.uint8)

                mask_crop = [0]
                while (np.sum(mask_crop) == 0):
                    img_crop, mask_crop = random_crop(img, mask, img_h)

                # if not os.path.exists('../output/val_imgs'):
                #     os.makedirs('../output/val_imgs')
                # f = os.path.join('../output/val_imgs', '{}.png'.format(name))
                # cv2.imwrite(f, img_crop)
                # f = os.path.join('../output/val_imgs', '{}_mask.png'.format(name))
                # cv2.imwrite(f, mask_crop * 255)

                mask_crop = np.expand_dims(mask_crop, axis=2)
                x_batch.append(img_crop)
                y_batch.append(mask_crop)


            x_batch = np.array(x_batch, np.float32) / 255.0
            y_batch = np.array(y_batch, np.float32)
            
            yield x_batch, y_batch

def test_generator(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1):
    imgs = glob.glob(os.path.join(TEST_DIR, '*_RGB.tif'))
    imgs = [img.split('_RGB')[0].split('\\')[1] for img in imgs]
    # print (len(imgs))
    img_mean, img_std = get_mean_std(cur_val_fold=cur_val_fold)
    batch_names = []
    img_n = 0
    for i, name in enumerate(imgs):
        # print (i, name)
        img = cv2.imread(os.path.join(TEST_DIR, '{}_RGB.tif'.format(name)))
        img = (img - img_mean) / (img_std + EPS)
        tiles = get_tiles(img, mask=None, tile_size=img_h)
        batch_names.append(name)

        for j in range(0, tiles.shape[0], batch_size):
            x_batch = tiles[j:j+batch_size]
            x_batch = np.array(x_batch, np.float32)
            yield img_n, x_batch, batch_names
            batch_names = []
        img_n += 1


# def train_generator_resize(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1):
#     folds = prepare_folds()
#     train_names = [name for fold in folds[0:cur_val_fold] + folds[cur_val_fold + 1:] for name in fold]
#     while True:
#         for start in range(0, len(train_names), batch_size):
#             x_batch = []
#             y_batch = []
#             end = min(start + batch_size, len(train_names))
#             names_train_batch = train_names[start:end]
#             for name in names_train_batch:
#                 img = cv2.imread(os.path.join(TRAIN_DIR, '{}_RGB.tif'.format(name)))
#                 img = cv2.resize(img, (img_w, img_h))

#                 mask = io.imread(os.path.join(TRAIN_DIR, '{}_GTI.tif'.format(name)))
#                 mask = cv2.resize(mask, (img_w, img_h))
#                 mask = np.array(mask != 0, np.uint8)

#                 img, mask = randomShiftScaleRotate(img, mask,
#                                                    shift_limit=(-0.0625, 0.0625),
#                                                    scale_limit=(-0.1, 0.1),
#                                                    rotate_limit=(-0, 0))
#                 img, mask = randomHorizontalFlip(img, mask)
#                 mask = np.expand_dims(mask, axis=2)
#                 x_batch.append(img)
#                 y_batch.append(mask)

#             x_batch = np.array(x_batch, np.float32) / 255.0
#             y_batch = np.array(y_batch, np.float32)
            
#             yield x_batch, y_batch


# def val_generator_resize(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1):
#     folds = prepare_folds()
#     val_names = [name for name in folds[cur_val_fold]]
#     while True:
#         for start in range(0, len(val_names), batch_size):
#             x_batch = []
#             y_batch = []
#             end = min(start + batch_size, len(val_names))
#             names_valid_batch = val_names[start:end]
#             for name in names_valid_batch:
#                 img = cv2.imread(os.path.join(TRAIN_DIR, '{}_RGB.tif'.format(name)))
#                 img = cv2.resize(img, (img_w, img_h))
#                 mask = cv2.imread(os.path.join(TRAIN_DIR, '{}_GTL.tif'.format(name)), cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (img_w, img_h))
#                 mask = np.array(mask > BACKGROUND_LABEL, np.uint8)
#                 mask = np.expand_dims(mask, axis=2)
#                 x_batch.append(img)
#                 y_batch.append(mask)
#             x_batch = np.array(x_batch, np.float32) / 255.0
#             y_batch = np.array(y_batch, np.float32)
#             yield x_batch, y_batch

# def test_generator_resize(batch_size, img_h, img_w):
#     names = glob.glob(os.path.join(TEST_DIR, '*_RGB.tif'))
#     names = [name.split('_RGB')[0].split('\\')[1] for name in names]
#     x_batch = np.zeros((batch_size, img_h, img_w, 3))
#     batch_names = []
#     batch_n = 0
#     for i, name in enumerate(names):
#         img = cv2.imread(os.path.join(TEST_DIR, '{}_RGB.tif'.format(name)))
#         x_batch[i % batch_size] = cv2.resize(img, (img_w, img_h))
#         batch_names.append(name)
#         if batch_size == 1 or i != 0 and (i - 1) % batch_size == 0:
#             yield batch_n, np.array(x_batch, np.float32) / 255, batch_names
#             batch_names = []
#             batch_n += 1
