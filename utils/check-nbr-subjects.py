import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools

def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx,1]
    in_cohort_nbr = data[idx,3]
    indices = np.where(data[:,1] == subj)[0]
    idx_same_leg = np.where(data[indices,3] == in_cohort_nbr)
    # print(indices[idx_same_leg])
    return indices[idx_same_leg]


def main(args):
    idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    lit = os.path.basename(args.root).split('_')[0]
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    dataset = np.load(dp)
    # print(os.path.basename(args.root).split('_')[0])
    # lit = os.path.basename(args.root).split('_')[0]
    # dp = os.path.join(args.data, 'data_') + 'Herta-Moller.npz'
    # dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']

    subjects = []

    for fold in range(1,6):
        idx = np.load(os.path.join(args.root, f'idx_{fold}.npz'))
        subject_indices = []
        nbr_in_fold = 0
        for i in idx['val_idx']:
            if i not in subject_indices:
                subject_indices = get_same_subject(info_file, i)
                nbr_in_fold += 1

        # print(nbr_in_fold)
        subjects.append(nbr_in_fold)

    print(subjects)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
