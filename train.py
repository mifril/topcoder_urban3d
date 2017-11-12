import argparse
import numpy as np
import pandas as pd
import glob
import os
import cv2
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

def train_dice(model, model_name, img_h, img_w, cur_val_fold=N_FOLDS - 1, n_epochs=100, batch_size=32, patience=5, reduce_rate=0.5):   
    load_best_weights_max(model, model_name, cur_val_fold, is_train=True, is_folds=True)

    callbacks = [
        EarlyStopping(monitor='val_dice_loss',
            patience=patience * 2,
            verbose=1,
            min_delta=1e-6,
            mode='max'),
        ReduceLROnPlateau(monitor='val_dice_loss',
            factor=reduce_rate,
            patience=patience,
            verbose=1,
            epsilon=1e-6,
            mode='max'),
        ModelCheckpoint(monitor='val_dice_loss',
            filepath='weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/{val_dice_loss:.6f}-{epoch:03d}.h5',
            save_best_only=True,
            mode='max'),
        TensorBoard(log_dir='logs')]

    model.fit_generator(generator=train_generator(batch_size, img_h, img_w),
        steps_per_epoch=np.ceil(N_TRAIN / float(batch_size)),
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_generator(batch_size, img_h, img_w),
        validation_steps=np.ceil(N_VAL / float(batch_size)))

def train(model, model_name, img_h, img_w, cur_val_fold=N_FOLDS - 1, n_epochs=100, batch_size=32, patience=5, reduce_rate=0.5):   
    load_best_weights_min(model, model_name, cur_val_fold, is_train=True, is_folds=True)

    callbacks = [
        EarlyStopping(monitor='val_loss',
            patience=patience * 2,
            verbose=1,
            min_delta=1e-6,
            mode='min'),
        ReduceLROnPlateau(monitor='val_loss',
            factor=reduce_rate,
            patience=patience,
            verbose=1,
            epsilon=1e-6,
            mode='min'),
        ModelCheckpoint(monitor='val_loss',
            filepath='weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/{val_loss:.6f}-{val_dice_loss:.6f}-{epoch:03d}.h5',
            save_best_only=True,
            mode='min'),
        TensorBoard(log_dir='logs')]

    model.fit_generator(generator=train_generator(batch_size, img_h, img_w),
        steps_per_epoch=np.ceil(N_TRAIN / float(batch_size)),
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_generator(batch_size, img_h, img_w),
        validation_steps=np.ceil(N_VAL / float(batch_size)))

def search_best_threshold(model, model_name, img_h, img_w, load_best_weights, batch_size=32, start_thr=0.3, end_thr=0.71, delta=0.01):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    load_best_weights(model, model_name)
   
    y_pred = []
    y_val = []
    cur_batch = 0
    for val_batch in val_generator(batch_size, img_h, img_w):
        X_val_batch, y_val_batch = val_batch
        y_val.append(y_val_batch)
        y_pred.append(model.predict_on_batch(X_val_batch))
        cur_batch += 1
        if cur_batch > N_VAL:
            break

    y_val, y_pred = np.concatenate(y_val), np.concatenate(y_pred)

    best_dice = -1
    best_thr = -1

    for cur_thr in tqdm(np.arange(start_thr, end_thr, delta)):
        cur_dice = get_score(y_val, y_pred, cur_thr)
        if cur_dice > best_dice:
            print('thr: {}, val dice: {:.5}'.format(cur_thr, cur_dice))
            best_dice = cur_dice
            best_thr = cur_thr
    return best_thr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net')
    parser.add_argument('-f', '--folds', action="store_true", help='train all folds')
    parser.add_argument("--start_fold", type=int, default=0, help="predictions directory")
    parser.add_argument("--start_lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    args = parser.parse_args()

    if args.folds:
        for cur_val_fold in range(args.start_fold, N_FOLDS):
            # model, model_name, img_h, img_w = model_1(lr=args.start_lr)
            model, model_name, img_h, img_w = model_2(lr=args.start_lr)
            # train_dice(model, model_name, img_h, img_w, cur_val_fold, n_epochs=1000, batch_size=1, patience=5, reduce_rate=0.1)
            train(model, model_name, img_h, img_w, cur_val_fold, n_epochs=1000, batch_size=args.batch, patience=5, reduce_rate=0.1)
            gc.collect()
    else:
        model, model_name, img_h, img_w = model_1(lr=args.start_lr)
        train(model, model_name, img_h, img_w, cur_val_fold=N_FOLDS - 1, n_epochs=1000, batch_size=args.batch, patience=5, reduce_rate=0.1)
    gc.collect()