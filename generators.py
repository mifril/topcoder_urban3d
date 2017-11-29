import numpy as np
import pandas as pd
import glob
import os
import cv2
from skimage import io
import scipy.misc as scm
from sklearn.model_selection import KFold

from utilities import *

def train_generator(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1, validate=True):
    folds = prepare_folds()
    if validate:
        train_names = [name for fold in folds[0:cur_val_fold] + folds[cur_val_fold + 1:] for name in fold]
    else:
        train_names = [name for fold in folds for name in fold]
    print('CVF: ', cur_val_fold, ', n_train: ', len(train_names))     
    print(np.array(train_names))
    # img_mean, img_std = get_mean_std(cur_val_fold=cur_val_fold)
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
                while (np.sum(mask_crop) < 10000):
                    img_crop, mask_crop = random_crop(img, mask, img_h)

                img_crop, mask_crop = randomHorizontalFlip(img_crop, mask_crop, u=0.5)

                dump_imgs(img_crop, mask_crop, name, dirname='train_imgs', tile_id='')
                
                mask_crop = np.expand_dims(mask_crop, axis=2)
                x_batch.append(img_crop)
                y_batch.append(mask_crop)

            x_batch = np.array(x_batch, np.float32) / 255.0
            # x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch

def val_generator(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1):
    folds = prepare_folds()
    val_names = [name for name in folds[cur_val_fold]]
    print('CVF: ', cur_val_fold)
    print(np.array(val_names))
    # img_mean, img_std = get_mean_std(cur_val_fold=cur_val_fold)
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

                img_crops, mask_crops = get_val_crops(img, mask, tile_size=img_w)

                for tile_id, zipped in enumerate(zip(img_crops, mask_crops)):
                    cur_img, cur_mask = zipped
                    cur_mask = np.expand_dims(cur_mask, axis=2)
                    x_batch.append(cur_img)
                    y_batch.append(cur_mask)

                    dump_imgs(cur_img, cur_mask, name, dirname='val_imgs', tile_id=tile_id)

                    if len(x_batch) >= batch_size:
                        x_batch = np.array(x_batch, np.float32) / 255.0
                        # x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.float32)
                        
                        # print('!!!!!!!!!!!!! YIELD: ', x_batch.shape, ' - ',  y_batch.shape)
                        yield x_batch, y_batch
                        
                        x_batch = []
                        y_batch = []

            if len(x_batch) > 0:
                x_batch = np.array(x_batch, np.float32) / 255.0
                # x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.float32)

                yield x_batch, y_batch

def test_generator(batch_size, img_h, img_w, cur_val_fold=N_FOLDS - 1):
    imgs = glob.glob(os.path.join(TEST_DIR, '*_RGB.tif'))
    imgs = [img.split('_RGB')[0].split('\\')[1] for img in imgs]
    # print (len(imgs))
    # img_mean, img_std = get_mean_std(cur_val_fold=cur_val_fold)
    batch_names = []
    img_n = 0
    for i, name in enumerate(imgs):
        # print (i, name)
        img = cv2.imread(os.path.join(TEST_DIR, '{}_RGB.tif'.format(name)))
        # img = (img - img_mean) / (img_std + EPS)
        tiles = get_tiles(img, mask=None, tile_size=img_h)
        batch_names.append(name)

        for j in range(0, tiles.shape[0], batch_size):
            x_batch = tiles[j:j+batch_size]
            x_batch = np.array(x_batch, np.float32) / 255.0
            # x_batch = np.array(x_batch, np.float32)
            yield img_n, x_batch, batch_names
            batch_names = []

        img_n += 1
