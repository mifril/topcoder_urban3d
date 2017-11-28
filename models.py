import numpy as np
import pandas as pd
import glob
import os
import cv2
from tqdm import tqdm

from rle import *
from generators import *
from metric import *

from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, load_model
from keras import backend as K
import tensorflow as tf
import gc

def get_unet_dilated(img_h, img_w, init_nb=32, loss=bce_dice_loss):
    inputs = Input((img_h, img_w, 3))
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(down1)
    down1pool = MaxPooling2D((2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down2)
    down2pool = MaxPooling2D((2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
    down3pool = MaxPooling2D((2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=1)(down3pool)
    dilate2 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=2)(dilate1)
    dilate3 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=4)(dilate2)
    dilate_all_added = add([dilate1, dilate2, dilate3])
    
    up3 = UpSampling2D((2, 2))(dilate_all_added)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    
    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)


    model = Model(inputs=inputs, outputs=classify)

    return model

def get_unet(img_h, img_w, dropout_val=0.0, batch_norm=False):
    filters = 32

    inputs = Input((img_h, img_w, 3))

    conv_512 = double_conv_layer(inputs, filters, dropout_val, batch_norm)
    pool_256 = MaxPooling2D(pool_size=(2, 2))(conv_512)

    conv_256 = double_conv_layer(pool_256, 2*filters, dropout_val, batch_norm)
    pool_128 = MaxPooling2D(pool_size=(2, 2))(conv_256)

    conv_128 = double_conv_layer(pool_128, 4*filters, dropout_val, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = double_conv_layer(pool_64, 8*filters, dropout_val, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = double_conv_layer(pool_32, 16*filters, dropout_val, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = double_conv_layer(pool_16, 32*filters, dropout_val, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2, 2))(conv_16)

    conv_8 = double_conv_layer(pool_8, 64*filters, dropout_val, batch_norm)

    up_16 = concatenate([UpSampling2D(size=(2, 2))(conv_8), conv_16])
    up_conv_16 = double_conv_layer(up_16, 32*filters, dropout_val, batch_norm)

    up_32 = concatenate([UpSampling2D(size=(2, 2))(conv_16), conv_32])
    up_conv_32 = double_conv_layer(up_32, 16*filters, dropout_val, batch_norm)

    up_64 = concatenate([UpSampling2D(size=(2, 2))(up_conv_32), conv_64])
    up_conv_64 = double_conv_layer(up_64, 8*filters, dropout_val, batch_norm)

    up_128 = concatenate([UpSampling2D(size=(2, 2))(up_conv_64), conv_128])
    up_conv_128 = double_conv_layer(up_128, 4*filters, dropout_val, batch_norm)

    up_256 = concatenate([UpSampling2D(size=(2, 2))(up_conv_128), conv_256])
    up_conv_256 = double_conv_layer(up_256, 2*filters, dropout_val, batch_norm)

    up_512 = concatenate([UpSampling2D(size=(2, 2))(up_conv_256), conv_512])
    up_conv_512 = double_conv_layer(up_512, filters, 0, batch_norm)

    conv_final = Conv2D(1, (1, 1))(up_conv_512)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final)
    return model

# def get_unet(img_h, img_w):
#     inputs = Input((img_h, img_w, 3))
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

#     up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

#     up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

#     model = Model(input=inputs, output=conv10)
#     return model


def get_unet_big(img_h, img_w):
    inputs = Input((img_h, img_w, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv5])
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv4])
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv3])
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

    up10 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv9), conv2])
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)

    up11 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv10), conv1])
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    model = Model(input=inputs, output=conv12)
    return model

def load_best_weights_min(model, model_name, cur_val_fold, is_train = True, is_folds=True, wdir=None):
    if wdir is None:
        if is_folds:
            wdir = 'weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/'
            if cur_val_fold == 0 or is_train and (not os.path.exists(wdir) or not os.listdir(wdir)):
                wdir = 'weights_' + str(model_name) + '/fold_0/'
                wdir_save = 'weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/'
                if not os.path.exists(wdir_save):
                    os.makedirs(wdir_save)
        else:
            wdir = 'weights_' + str(model_name) + '/'

    print('looking for weights in {}'.format(wdir))
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    elif len(os.listdir(wdir)) > 0:
        print(os.listdir(wdir))
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

def load_best_weights_max(model, model_name, cur_val_fold, is_train = True, is_folds=True, wdir=None):
    if wdir is None:
        if is_folds:
            wdir = 'weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/'
            if cur_val_fold == 0 or is_train and (not os.path.exists(wdir) or not os.listdir(wdir)):
                wdir = 'weights_' + str(model_name) + '/fold_0/'
                wdir_save = 'weights_' + str(model_name) + '/fold_' + str(cur_val_fold) + '/'
                if not os.path.exists(wdir_save):
                    os.makedirs(wdir_save)
        else:
            wdir = 'weights_' + str(model_name) + '/'
        
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[-1]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

def model_1(opt=Adam(1e-3), img_size=512):
    model = get_unet(img_size, img_size)
    model_name = 'unet_{}'.format(img_size)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice, 'accuracy'])
    
    return model, model_name, img_size, img_size

def model_2(opt=Adam(1e-3), img_size=512):
    model = get_unet(img_size, img_size, 0.2, True)
    model_name = 'unet_{}_bnd'.format(img_size)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice, 'accuracy'])
    
    return model, model_name, img_size, img_size 

def model_3(opt=Adam(1e-3), img_size=512):
    model = get_unet_dilated(img_size, img_size)
    model_name = 'unet_{}_dilated'.format(img_size)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice, 'accuracy'])
    
    return model, model_name, img_size, img_size 

def model_4(opt=Adam(1e-3), img_size=512):
    model = get_unet_big(img_size, img_size)
    model_name = 'unet_{}_big'.format(img_size)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice, 'accuracy'])
    
    return model, model_name, img_size, img_size
