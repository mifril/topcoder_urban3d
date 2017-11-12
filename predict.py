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

# works only when tile size is 1024
# UGLY HARDCODE. I hate myself for this
def construct_img_from_tiles(tiles):
    result = np.zeros((IMG_W, IMG_H, 1))
    # print(result.shape, tiles[0].shape, tiles[0][:768, :768].shape)
    result[:768, :768] = tiles[0][:768, :768]
    result[768:1280, :768] = tiles[3][256:768, :768]
    result[1280:, :768] = tiles[6][256:, :768]

    result[:768, 768:1280] = tiles[1][:768, 256:768]
    result[768:1280, 768:1280] = tiles[4][256:768, 256:768]
    result[1280:, 768:1280] = tiles[7][256:, 256:768]

    result[:768, 1280:] = tiles[2][:768, 256:]
    result[768:1280, 1280:] = tiles[5][256:768, 256:]
    result[1280:, 1280:] = tiles[8][256:, 256:]
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

def predict_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=True):
    cur_val_fold = N_FOLDS - 1
    load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True)
    batch_iterator = test_generator(batch_size, img_h, img_w, cur_val_fold)

    preds = []
    names = []
    cur_img_preds = []
    cur_img = -1
    for batch in tqdm(batch_iterator):
        i, X_batch, batch_names = batch
        if cur_img == -1 or cur_img != i:
            if cur_img != -1:
                preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds)))
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

    preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds)))

    preds = np.array(preds)
    names = np.concatenate(names)
    print (preds.shape, names.shape)

    return preds, names

def predict_folds_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=True, start_batch=0, n_batches=N_TEST):
    preds = []
    names = []

    for cur_val_fold in range(N_FOLDS):
        load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True)
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
                    fold_preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds)))
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

        fold_preds.append(construct_img_from_tiles(np.concatenate(cur_img_preds)))

        fold_preds = np.array(fold_preds)
        # print ('fold_preds', np.array(fold_preds).shape)
        # fold_preds = np.concatenate(fold_preds)
        preds.append(fold_preds)

    # print ('preds', np.array(preds).shape)
    preds = np.mean(preds, axis=0)
    names = np.concatenate(names)

    return preds, names

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

# def search_best_threshold(model, model_name, img_h, img_w, load_best_weights, batch_size=32, start_thr=0.3, end_thr=0.71, delta=0.01):
#     for cur_val_fold in range(N_FOLDS):
#         load_best_weights(model, model_name, cur_val_fold, is_train = False, is_folds=True)
#         batch_iterator = test_generator(batch_size, img_h, img_w, cur_val_fold)
   
#         y_pred = []
#         y_val = []
#         cur_batch = 0
#         for val_batch in val_generator(ids_val_split, batch_size, img_h, img_w):
#             X_val_batch, y_val_batch = val_batch
#             y_val.append(y_val_batch)
#             y_pred.append(model.predict_on_batch(X_val_batch))
#             cur_batch += 1
#             if cur_batch > ids_val_split.shape[0]:
#                 break

#     y_val, y_pred = np.concatenate(y_val), np.concatenate(y_pred)

#     best_dice = -1
#     best_thr = -1

#     for cur_thr in tqdm(np.arange(start_thr, end_thr, delta)):
#         cur_dice = get_score(y_val, y_pred, cur_thr)
#         if cur_dice > best_dice:
#             print('thr: {}, val dice: {:.5}'.format(cur_thr, cur_dice))
#             best_dice = cur_dice
#             best_thr = cur_thr
#     return best_thr

def save_predictions(preds, names, pred_dir=PRED_DIR, threshold=0.3):
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    for pred, name in zip(preds, names):
        mask = cv2.resize(pred, (IMG_W, IMG_H))
        mask = mask > threshold
        mask = np.array(mask, np.uint8) * 255
        print(name, np.unique(mask, return_counts=True))
        f = os.path.join(pred_dir, '{}.png'.format(name))
        cv2.imwrite(f, mask)

def make_submission(out_file, pred_dir=PRED_DIR, delete_small=True):
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

# python predict.py -tdf --out_file tta_folds --pred_dir ../output/tta_folds
if __name__ == '__main__':
    model, model_name, img_h, img_w = model_1()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--tta", action="store_true", help="TTA usage flag")
    parser.add_argument("-f", '--folds', action="store_true", help="use all folds")
    parser.add_argument("-s", "--only_make_submission", action="store_true", help="if set, will not run predict() and save_predictions()")
    parser.add_argument("-d", "--delete_small", action="store_true", help="if set, small objects will be deleted from masks")
    parser.add_argument("--out_file", default=model_name, help="submission file")
    parser.add_argument("--pred_dir", default=PRED_DIR, help="predictions directory")
    args = parser.parse_args()


    load_best_weights = load_best_weights_min

    if not args.only_make_submission:
        if args.folds:
            n_batches = 20
            for start_batch in range(0, N_TEST, n_batches):
                print ('Predict batches [{}; {}]'.format(start_batch, start_batch + n_batches))
                preds, names = predict_folds_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=args.tta, start_batch=start_batch, n_batches=n_batches)
                save_predictions(preds, names, pred_dir=args.pred_dir)
        else:
            preds, names = predict_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=args.tta)
            save_predictions(preds, names, pred_dir=args.pred_dir)
    make_submission(OUTPUT_DIR + args.out_file + '.txt', pred_dir=args.pred_dir, delete_small=args.delete_small)
