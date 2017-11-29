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

def train_dice(model, model_name, img_h, img_w, cur_val_fold=N_FOLDS - 1, n_epochs=100, batch_size=32, patience=5, reduce_rate=0.5, validate=True, wdir=None):   
    load_best_weights_max(model, model_name, cur_val_fold, is_train=True, is_folds=True, wdir=wdir)

    callbacks = [
        EarlyStopping(monitor='val_dice',
            patience=patience * 2,
            verbose=1,
            min_delta=1e-6,
            mode='max'),
        ReduceLROnPlateau(monitor='val_dice',
            factor=reduce_rate,
            patience=patience,
            verbose=1,
            epsilon=1e-6,
            mode='max'),
        ModelCheckpoint(monitor='val_dice',
            filepath='weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/{val_dice:.6f}-{epoch:03d}.h5',
            save_best_only=True,
            mode='max'),
        # TensorBoard(log_dir='logs')
        ]

    if validate:
        model.fit_generator(generator=train_generator(batch_size, img_h, img_w, cur_val_fold),
            steps_per_epoch=np.ceil(N_TRAIN / float(batch_size)),
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_generator(batch_size, img_h, img_w, cur_val_fold),
            validation_steps=np.ceil(N_VAL / float(batch_size)))
    else:
        model.fit_generator(generator=train_generator(batch_size, img_h, img_w, validate=False),
            steps_per_epoch=np.ceil(N_TRAIN / float(batch_size)),
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks)

def train(model, model_name, img_h, img_w, cur_val_fold=N_FOLDS - 1, n_epochs=100, batch_size=32, patience=5, reduce_rate=0.5, validate=True, wdir=None):   
    load_best_weights_min(model, model_name, cur_val_fold, is_train=True, is_folds=True, wdir=wdir)

    if validate:
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
                filepath='weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/{val_loss:.6f}-{val_dice:.6f}-{epoch:03d}.h5',
                save_best_only=True,
                mode='min')
            ]

        model.fit_generator(generator=train_generator(batch_size, img_h, img_w, cur_val_fold),
            steps_per_epoch=np.ceil(N_TRAIN / float(batch_size)),
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_generator(batch_size, img_h, img_w, cur_val_fold),
            validation_steps=np.ceil(N_VAL / float(batch_size)))
    else:
        callbacks = [
            EarlyStopping(monitor='loss',
                patience=patience * 2,
                verbose=1,
                min_delta=1e-6,
                mode='min'),
            ReduceLROnPlateau(monitor='loss',
                factor=reduce_rate,
                patience=patience,
                verbose=1,
                epsilon=1e-6,
                mode='min'),
            ModelCheckpoint(monitor='loss',
                filepath='weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/{loss:.6f}-{dice:.6f}-{epoch:03d}_no_val.h5',
                save_best_only=True,
                mode='min')
            ]

        model.fit_generator(generator=train_generator(batch_size, img_h, img_w, validate=False),
            steps_per_epoch=np.ceil(N_TRAIN_ALL / float(batch_size)),
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net')
    parser.add_argument('-f', '--folds', action="store_true", help='train all folds')
    parser.add_argument("--start_fold", type=int, default=0, help="predictions directory")
    parser.add_argument("--start_lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--patience", type=int, default=5, help="LRSheduler patience (Early Stopping - 2 * patience)")
    parser.add_argument("--model", type=int, default=1, help="model to train")
    parser.add_argument("--opt", type=str, default='adam', help="model optimiser")
    parser.add_argument("--train_type", type=str, default='loss', help="train type (dice or loss)")
    parser.add_argument("--img_size", type=int, default=512, help="NN input size")
    parser.add_argument("--no_val", action="store_true", help="validation flag")
    parser.add_argument("--wdir", type=str, default=None, help="weights dir, if None - load by model_name")
    args = parser.parse_args()

    models = [None, model_1, model_2, model_3, model_4, model_5]
    opts = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
    model_f = models[args.model]
    validate = not args.no_val

    if args.folds:
        for cur_val_fold in range(args.start_fold, N_FOLDS):
            print('cur_val_fold_begin: ', cur_val_fold)
            opt = opts[args.opt](args.start_lr)
            model, model_name, img_h, img_w = model_f(opt, args.img_size)
            if args.train_type == 'dice':
                train_dice(model, model_name, img_h, img_w, cur_val_fold, n_epochs=1000, batch_size=args.batch, patience=args.patience, reduce_rate=0.1, validate=validate, wdir=args.wdir)
            else:
                train(model, model_name, img_h, img_w, cur_val_fold, n_epochs=1000, batch_size=args.batch, patience=args.patience, reduce_rate=0.1, validate=validate, wdir=args.wdir)
            gc.collect()
    else:
        opt = opts[args.opt](args.start_lr)
        model, model_name, img_h, img_w = model_f(opt)
        if args.train_type == 'dice':
                train_dice(model, model_name, img_h, img_w, cur_val_fold=0, n_epochs=1000, batch_size=args.batch, patience=args.patience, reduce_rate=0.1, validate=validate, wdir=args.wdir)
        else:
            train(model, model_name, img_h, img_w, cur_val_fold=0, n_epochs=1000, batch_size=args.batch, patience=args.patience, reduce_rate=0.1, validate=validate, wdir=args.wdir)
    gc.collect()
