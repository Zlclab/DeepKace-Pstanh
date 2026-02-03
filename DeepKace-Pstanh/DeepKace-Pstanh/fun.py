import numpy as np
import sys, math, os, h5py
import matplotlib

matplotlib.use('Agg')



def Id_k_folds(seqs_num, k_folds):
    train_ids = []
    valid_ids = []

    each_fold_num = int(math.ceil(seqs_num / k_folds))

    for fold in range(k_folds):
        index = np.arange(seqs_num)
        valid_slice = index[fold * each_fold_num:(fold + 1) * each_fold_num]

        train_left = list(set(index) - set(valid_slice))
        train_ids.append(np.asarray(train_left))
        valid_ids.append(np.asarray(valid_slice))

    return (train_ids, valid_ids)

