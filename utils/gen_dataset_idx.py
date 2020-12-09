import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser


def gen_train_val(info_file, ratio):
    meta_data = pd.read_csv(info_file, delimiter=',')

    data = np.array(meta_data.values[5:, :4], dtype=int)
    train_size = int(np.round(ratio * (1 + data[-1, 1])))
    subj_idx = np.random.choice((1 + data[-1, 1]), train_size, replace=False)
    train_idx = []
    val_idx = []
    val_subj = []

    for a in data:
        if a[1] in subj_idx:
            train_idx = np.append(train_idx, a[0])
        else:
            val_idx = np.append(val_idx, a[0])
            if a[1] not in val_subj:
                val_subj = np.append(val_subj, a[1])

    train_idx = train_idx.astype(np.int16)
    val_idx = val_idx.astype(np.int16)
    subj_idx = subj_idx.astype(np.int16)
    val_subj = val_subj.astype(np.int16)
    return train_idx, val_idx, subj_idx, val_subj


def gen_rnd(y, train_ratio, val_ratio):
    train_size = int(np.round(train_ratio * len(y)))
    train_idx = np.random.choice(len(y), train_size,
                                 replace=False)

    non_train = []
    for a in range(len(y)):
        if a not in train_idx:
            non_train = np.append(non_train, a)

    val_size = int(np.round(val_ratio * (1 + len(y))))
    val_idx = np.random.choice(len(y), val_size,
                               replace=False)
    test_idx = []
    for a in range(len(y)):
        if a not in train_idx and a not in val_idx:
            test_idx = np.append(test_idx, a)

    return train_idx, val_idx, test_idx


def gen_train_val_test(info_file, train_ratio, val_ratio=''):
    if val_ratio == '':
        val_ratio = (1 - train_ratio) / 2
    if train_ratio + val_ratio >= 1:
        val_ratio = (1 - train_ratio) / 2

    # try:
    meta_data = pd.read_csv(info_file, delimiter=',')
    # except IOError as e:
    #     return gen_rnd(train_ratio, val_ratio)
    print(train_ratio)
    print(val_ratio)
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    # print(meta_data.values[first_data, :])

    data = np.array(meta_data.values[first_data:, :4], dtype=int)
    train_size = int(np.round(train_ratio * (1 + data[-1, 1])))
    subj_idx = np.random.choice((1 + data[-1, 1]), train_size, replace=False)
    train_idx = []
    non_train = []

    for a in data:
        if a[1] in subj_idx:
            train_idx = np.append(train_idx, a[0])
        else:
            non_train = np.append(non_train, a[1])

    val_size = int(np.round(val_ratio * (1 + data[-1, 1])))
    val_subj = np.random.choice(non_train, val_size, replace=False)
    val_idx = []
    test_idx = []

    for a in data:
        if a[1] in val_subj:
            val_idx = np.append(val_idx, a[0])
        elif a[1] not in subj_idx:
            test_idx = np.append(test_idx, a[0])

    train_idx = train_idx.astype(np.int16)
    val_idx = val_idx.astype(np.int16)
    test_idx = test_idx.astype(np.int16)
    return train_idx, val_idx, test_idx


def main(args):
    # train, test, train_subj, val_subj = gen_train_val(args.info_file, 0.9)
    # # train, val, test = gen_train_val_test(args.info_file, 0.8)
    # # print(train.shape)
    # # print(val.shape)
    # # print(test.shape)

    sp = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    # np.savez(sp, train_idx=train, test_idx=test,
    #          train_subj=train_subj, test_subj=val_subj)
    data = np.load(sp)
    train = data['train_idx']
    test = data['test_idx']
    train_subj = data['train_subj']
    test_subj = data['test_subj']

    print(train)
    print(test)
    print(train_subj)
    print(test_subj)
    #
    # for a in train:
    #     if a in val:
    #         print('FAN {} in train och val'.format(a))
    #     if a in test:
    #         print('FAN {} in train och test'.format(a))
    #
    # for a in val:
    #     if a in test:
    #         print('FAN {} in val och test'.format(a))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('info_file')
    args = parser.parse_args()
    main(args)
