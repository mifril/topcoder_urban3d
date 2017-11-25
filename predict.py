import argparse
import numpy as np
import pandas as pd
import glob
import os
import cv2
from scipy import ndimage
from tqdm import tqdm

from rle import *
from generators import *
from metric import *
from models import *

from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
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

# def predict(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=True):
#     cur_val_fold = 2
#     load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True)
#     batch_iterator = test_generator(batch_size, img_h, img_w)

#     preds = []
#     names = []
#     for batch in tqdm(batch_iterator):
#         i, X_batch, batch_names = batch
#         cur_pred = model.predict_on_batch(X_batch)
#         if tta:
#             X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
#             cur_pred_flip = model.predict_on_batch(X_batch_flip)
#             cur_pred_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in cur_pred_flip])
#             cur_pred = 0.5 * cur_pred + 0.5 * cur_pred_flip
        
#         preds.append(cur_pred)
#         names.append(batch_names)

#     preds = np.concatenate(preds)
#     names = np.concatenate(names)

#     return preds, names

# def predict_folds(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=True):
#     preds = []
#     names = []

#     for cur_val_fold in range(N_FOLDS):
#         load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True)
#         batch_iterator = test_generator(batch_size, img_h, img_w)

#         fold_preds = []

#         for batch in tqdm(batch_iterator):
#             i, X_batch, batch_names = batch
#             cur_pred = model.predict_on_batch(X_batch)
#             if tta:
#                 X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
#                 cur_pred_flip = model.predict_on_batch(X_batch_flip)
#                 cur_pred_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in cur_pred_flip])
#                 cur_pred = 0.5 * cur_pred + 0.5 * cur_pred_flip
            
#             fold_preds.append(cur_pred)
#             if cur_val_fold == 0:
#                 names.append(batch_names)

#         fold_preds = np.concatenate(fold_preds)
#         preds.append(fold_preds)

#     preds = np.mean(preds, axis=0)
#     names = np.concatenate(names)

#     return preds, names

def predict_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=True, wdir=None):
    cur_val_fold = 0
    load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True, wdir=wdir)
    batch_iterator = test_generator(batch_size, img_h, img_w, cur_val_fold)

    preds = []
    names = []
    cur_img_preds = []
    cur_img = -1
    for batch in tqdm(batch_iterator):
        i, X_batch, batch_names = batch
        if cur_img == -1 or cur_img != i:
            if cur_img != -1:
                preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds), img_h))
            names.append(batch_names)
            cur_img = i
            cur_img_preds = []


        cur_pred = model.predict_on_batch(X_batch)
        if tta:
            X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
            cur_pred_flip = model.predict_on_batch(X_batch_flip)
            cur_pred_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in cur_pred_flip])
            cur_pred = 0.5 * cur_pred + 0.5 * cur_pred_flip
        
        cur_img_preds.append(cur_pred)

    preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds), img_h))

    preds = np.array(preds)
    names = np.concatenate(names)
    print (preds.shape, names.shape)

    return preds, names

def predict_folds_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=True, start_batch=0, n_batches=N_TEST, wdir=None):
    preds = []
    names = []

    for cur_val_fold in range(N_FOLDS):
        load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True, wdir=wdir)
        batch_iterator = test_generator(batch_size, img_h, img_w, cur_val_fold)

        fold_preds = []

        cur_img = -1
        cur_img_preds = []
        for batch in tqdm(batch_iterator):
            i, X_batch, batch_names = batch
            if i < start_batch:
                continue
            if i >= start_batch + n_batches:
                break
            if cur_img == -1 or cur_img != i:
                if cur_img != -1:
                    fold_preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds), img_h))
                if cur_val_fold == 0:
                    names.append(batch_names)
                cur_img = i
                cur_img_preds = []

            cur_pred = model.predict_on_batch(X_batch)
            if tta:
                X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
                cur_pred_flip = model.predict_on_batch(X_batch_flip)
                cur_pred_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in cur_pred_flip])
                cur_pred = 0.5 * cur_pred + 0.5 * cur_pred_flip
            
            cur_img_preds.append(cur_pred)

        fold_preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds), img_h))

        fold_preds = np.array(fold_preds)
        # print ('fold_preds', np.array(fold_preds).shape)
        # fold_preds = np.concatenate(fold_preds)
        preds.append(fold_preds)

    # print ('preds', np.array(preds).shape)
    preds = np.mean(preds, axis=0)
    names = np.concatenate(names)

    return preds, names

def save_predictions(preds, names, pred_dir, threshold=0.4):
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    for pred, name in zip(preds, names):
        mask = cv2.resize(pred, (IMG_W, IMG_H))
        mask = mask > threshold
        mask = np.array(mask, np.uint8) * 255
        print(name, np.unique(mask, return_counts=True))
        f = os.path.join(pred_dir, '{}.png'.format(name))
        cv2.imwrite(f, mask)

def make_submission(out_file, pred_dir, delete_small=True):
    f_submit = open(out_file, "w")
    threshold = 127

    for f in sorted(os.listdir(pred_dir)):
        mask_img_path = os.path.join(pred_dir, f)
        tile_id = f.split('.png')[0]
        mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        labeled_array, num_features = ndimage.label(mask_img)
        if delete_small:
            mask_img, labeled_array, num_features = threshold_small(mask_img, labeled_array, num_features)
        print("Tile: ", tile_id)
        print("Num houses: ", num_features)
        rle_str = rle_to_string(rlencode(labeled_array.flatten()))
        f_submit.write("{tile_id}\n2048,2048\n{rle}\n".format(tile_id=tile_id, rle=rle_str))
        
    f_submit.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--tta", action="store_true", help="TTA usage flag")
    parser.add_argument("-f", '--folds', action="store_true", help="use all folds")
    parser.add_argument("-s", "--only_make_submission", action="store_true", help="if set, will not run predict() and save_predictions()")
    parser.add_argument("-d", "--delete_small", action="store_true", help="if set, small objects will be deleted from masks")
    parser.add_argument("--out_file", default='out', help="submission file")
    parser.add_argument("--pred_dir", default='', help="predictions directory")
    parser.add_argument("--model", type=int, default=1, help="model to train")
    parser.add_argument("--img_size", type=int, default=512, help="NN input size")
    parser.add_argument("--wdir", type=str, default=None, help="weights dir, if None - load by model_name")
    args = parser.parse_args()

    models = [None, model_1, model_2, model_3, model_4]
    model_f = models[args.model]
    model, model_name, img_h, img_w = model_f(Adam(1e-3), args.img_size)

    load_best_weights = load_best_weights_min
    pred_dir=os.path.join(PRED_DIR, args.pred_dir)

    if not args.only_make_submission:
        if args.folds:
            n_batches = 20
            for start_batch in range(0, N_TEST, n_batches):
                print ('Predict batches [{}; {}]'.format(start_batch, start_batch + n_batches))
                preds, names = predict_folds_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=args.tta, start_batch=start_batch, n_batches=n_batches, wdir=args.wdir)
                save_predictions(preds, names, pred_dir)
        else:
            preds, names = predict_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=args.tta, wdir=args.wdir)
            save_predictions(preds, names, pred_dir)
    make_submission(OUTPUT_DIR + args.out_file + '.txt', pred_dir, delete_small=args.delete_small)
    gc.collect()
