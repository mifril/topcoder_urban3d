import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm
from generators import prepare_folds

INPUT_DIR = '../input/'
MAIN_DIR = 'C://data//'
TRAIN_DIR = MAIN_DIR + 'training/'
TEST_DIR = MAIN_DIR + 'testing/'

IMG_H = 2048
IMG_W = 2048

N_TRAIN_ALL = 174
N_TRAIN = 116
N_VAL = 58
N_FOLDS = 3

def calculate_mean_std():       
    res = pd.DataFrame()

    folds = prepare_folds()
    for cur_val_fold in tqdm(range(N_FOLDS)):
        tiles = [tile for fold in folds[0:cur_val_fold] + folds[cur_val_fold + 1:] for tile in fold]
        images = []
        for i, tile in enumerate(tiles):
            img = cv2.imread(os.path.join(TRAIN_DIR, '{}_RGB.tif'.format(tile)))
            images.append(img)
        images = np.array(images)
        print (images.shape)

        mean = images.mean(axis=(0, 1, 2))

        stds = []
        for i in range(0, N_TRAIN, 58):
            print (images[i:i + 58].shape)
            stds.append(images[i:i + 58].std(axis=(0, 1, 2)))
            print (i, stds)

        res['mean_no_fold_{}'.format(cur_val_fold)] = mean
        res['std_no_fold_{}'.format(cur_val_fold)] = np.mean(stds)

    res.to_csv(INPUT_DIR + 'mean_std.csv')

if __name__ == '__main__':
    calculate_mean_std()
