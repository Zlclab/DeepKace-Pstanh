import numpy as np, sys, math, os, h5py
from sklearn.metrics import roc_curve, precision_recall_curve, auc, r2_score
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from multiprocessing import Pool
from scipy import stats

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate ids for k-flods cross-validation
def Id_k_folds(seqs_num, k_folds, ratio):
    train_ids = []
    test_ids = []
    valid_ids = []
    if k_folds == 1:
        train_num = int(seqs_num * 0.7)
        test_num = seqs_num - train_num
        valid_num = int(train_num * ratio)
        train_num = train_num - valid_num
        index = range(seqs_num)
        train_ids.append(np.asarray(index[:train_num]))
        valid_ids.append(np.asarray(index[train_num:train_num + valid_num]))
        test_ids.append(np.asarray(index[train_num + valid_num:]))
    else:
        each_fold_num = int(math.ceil(seqs_num / k_folds))
        for fold in range(k_folds):
            index = range(seqs_num)
            index_slice = index[fold * each_fold_num:(fold + 1) * each_fold_num]
            index_left = list(set(index) - set(index_slice))
            test_ids.append(np.asarray(index_slice))
            train_num = len(index_left) - int(len(index_left) * ratio)
            train_ids.append(np.asarray(index_left[:train_num]))
            valid_ids.append(np.asarray(index_left[train_num:]))

    return (train_ids, test_ids, valid_ids)
