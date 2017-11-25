import argparse
import numpy as np
import pandas as pd
import glob
import os
import cv2
from scipy import ndimage
from tqdm import tqdm

from generators import *
from metric import *
from models import *

from keras.models import Model
import h5py
import gc

from sklearn.model_selection import train_test_split

RS = 17

def threshold_small(mask_img, labeled_array, num_features, pixel_threshold=100):
    unique, counts = np.unique(labeled_array, return_counts=True)
    for (k,v) in dict(zip(unique, counts)).items():
        if v < pixel_threshold:
            mask_img[labeled_array == k] = 0
    labeled_array, num_features = ndimage.label(mask_img)
    return mask_img, labeled_array, num_features

def construct_img_from_tiles(tiles, tile_size=1024):
    result = np.zeros((IMG_W, IMG_H, 1))
    shift = int(tile_size / 2)
    N = int(IMG_W / shift - 1)
    for i in range(N):
        for j in range(N):
            result[i*shift : i*shift + tile_size, j*shift : j*shift + tile_size] = tiles[i * N + j]
    return result

def search_best_threshold(model, model_name, img_h, img_w, load_best_weights, cur_val_fold=0, batch_size=1, start_thr=0.3, end_thr=0.71, delta=0.01, wdir=None):
    load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True, wdir=wdir)

    y_val = []
    y_pred = []
    cur_preds = []
    cur_vals = []
    cur_batch = 0
    if img_h == 512:
        tiles_in_img = 49
    if img_h == 1024:
        tiles_in_img = 9

    for val_batch in tqdm(val_generator(batch_size, img_h, img_w, cur_val_fold)):
        if cur_batch % tiles_in_img == 0 and cur_batch != 0:
            y_val.append(construct_img_from_tiles(np.concatenate(cur_vals), img_h))
            y_pred.append(construct_img_from_tiles(np.concatenate(cur_preds), img_h))
            cur_preds = []
            cur_vals = []
                
        X_batch, y_batch = val_batch
        
        cur_preds.append(model.predict_on_batch(X_batch))
        cur_vals.append(y_batch)

        cur_batch += 1
        if cur_batch > N_VAL * tiles_in_img:
            break

    y_val, y_pred = np.array(y_val), np.array(y_pred)
    print (y_pred.shape, y_val.shape)

    best_dice = -1
    best_thr = -1

    for cur_thr in tqdm(np.arange(start_thr, end_thr, delta)):
        cur_dice = get_score(y_val, y_pred, cur_thr)
        if cur_dice > best_dice:
            print('thr: {}, val dice: {:.5}'.format(cur_thr, cur_dice))
            best_dice = cur_dice
            best_thr = cur_thr
    print ('Best thr: ', best_thr, ', dice = ', best_dice)
    return best_thr

# for file_name in tqdm(img_files):
#     pred = (cv2.imread(str(pred_dict[file_name.name]), 0) > 0.2 * 255).astype(np.uint8)

#     gt = load_mask(str(file_name).replace('RGB', 'GTL'))
#     labels = measure.label(gt)
#     regions = measure.regionprops(labels)
#     new_mask = gt.copy()

#     for building_ind in range(labels.max()):
#         prop = regions[building_ind]
#         y_min, x_min, y_max, x_max = prop.bbox
#         tmp_gt = gt[y_min:y_max, x_min:x_max]
#         tmp_pred = pred[y_min:y_max, x_min:x_max]

#         if tmp_pred[tmp_gt != 0].sum() == 0:
#             new_mask[labels == building_ind + 1] = 0

#     cv2.imwrite(str(new_mask_path / (file_name.stem + '.png')), new_mask * 255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", '--folds', action="store_true", help="use all folds")
    parser.add_argument("-d", "--delete_small", action="store_true", help="if set, small objects will be deleted from masks")
    parser.add_argument("--model", type=int, default=1, help="model to train")
    parser.add_argument("--img_size", type=int, default=512, help="NN input size")
    parser.add_argument("--wdir", type=str, default=None, help="weights dir, if None - load by model_name")
    args = parser.parse_args()

    models = [None, model_1, model_2, model_3, model_4]
    model_f = models[args.model]
    model, model_name, img_h, img_w = model_f(Adam(1e-3), args.img_size)
    load_best_weights = load_best_weights_min


    res = pd.DataFrame()
    best_thr = search_best_threshold(model, model_name, img_h, img_w, load_best_weights, batch_size=1, start_thr=0.2, end_thr=0.51, delta=0.01, wdir=args.wdir)

    res['thr_no_fold_model_{}'.format(args.model)] = best_thr
    res.to_csv(INPUT_DIR + 'thr.csv')

    gc.collect()
