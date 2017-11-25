import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy

def dice(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice(y_true, y_pred))

def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum()
    union = y_true.sum() + y_pred.sum() + epsilon

    return 2 * (intersection / union).mean()

def get_score(train_masks, preds, thr=0.5):
    return get_dice(train_masks, preds > thr)
    # for i in range(train_masks.shape[0]):
    #     d += dice(train_masks[i], preds[i] > thr)
    # return d / train_masks.shape[0]
