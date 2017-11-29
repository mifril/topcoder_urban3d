import glob
import os

from metric import *
from linknet import *
from unet import *

from keras.optimizers import *
from keras.models import load_model

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

def model_5(opt=Adam(1e-3), img_size=512):
    if K.image_dim_ordering() == 'th':
        shape = (3, img_size, img_size)
    else:
        shape = (img_size, img_size, 3)
    
    model = get_linknet(shape, batch_norm=True, dropout=0.3, batch_norm_in=True)
    model_name = 'linknet_{}'.format(img_size)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice, 'accuracy'])
    
    return model, model_name, img_size, img_size
