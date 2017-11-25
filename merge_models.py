import argparse
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import gc


from rle import *
from generators import *
from metric import *
from models import *
from predict import *

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
    args = parser.parse_args()

    models = [None, model_1, model_2, model_3, model_4]
    model_f = models[args.model]
    
    load_best_weights = load_best_weights_min
    pred_dir=os.path.join(PRED_DIR, args.pred_dir)

    if not args.only_make_submission:
        model, model_name, img_h, img_w = model_f(Adam(1e-3), args.img_size)
        preds_1, names = predict_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=args.tta, wdir='weights_unet_512_big_noval\\')

        model, model_name, img_h, img_w = model_f(Adam(1e-3), args.img_size)
        preds_2, names = predict_tiles(model, model_name, img_h, img_w, load_best_weights, batch_size=1, tta=args.tta, wdir='weights_unet_512_big_noval\\fold_0\\')

        save_predictions(0.5 * preds_1 + 0.5 * preds_2, names, pred_dir)

    make_submission(OUTPUT_DIR + args.out_file + '.txt', pred_dir, delete_small=args.delete_small)
    gc.collect()
