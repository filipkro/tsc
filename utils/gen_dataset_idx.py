import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser


def gen_train_val(info_file, ratio):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)
    print(data)
    print(meta_data.values)
    print(first_data)
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

def gen_xtra_test(info_file, xtra, old):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)
    subjects = np.arange(103)
    print(len(subjects))
    # print(subjects)
    # print(old)
    for sub in old:
        idx = np.where(subjects == sub)[0][0]
        subjects = np.delete(subjects, idx)
    subj_idx = np.random.choice(subjects, xtra, replace=False)

    print(subj_idx)
    print(subjects)
    subjects = np.append(old, subj_idx)
    print(subjects)
    assert len(subjects) == len(np.unique(subjects))
    train_idx = []
    train_subj = []
    test_idx = []

    for a in data:
        if a[1] in subjects:
            test_idx = np.append(test_idx, a[0])
        else:
            train_idx = np.append(train_idx, a[0])
            if a[1] not in train_subj:
                train_subj = np.append(train_subj, a[1])

    train_idx = train_idx.astype(np.int16)
    test_idx = test_idx.astype(np.int16)
    subjects = subjects.astype(np.int16)
    train_subj = train_subj.astype(np.int16)
    return train_idx, test_idx, train_subj, subjects

def get_same_subject(data, idx):
    subj = data[idx,1]
    indices = np.where(data[:,1] == subj)[0]
    print(indices)
    return indices

def gen_tv_from_test(info_file, ratio, test_subj):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)
    train_size = int(np.round(ratio * (1 + data[-1, 1] - len(test_subj))))

    ti = []
    for i in data:
        if i[1] in test_subj:
            ti.append(i[0])

    data = np.delete(data, ti, axis=0)
    tv_subjects = list(set(data[:,1]))
    subj_idx = np.random.choice(tv_subjects, train_size, replace=False)
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

def gen_test_idx(info_file, test_subjects):
    test_idxs = []
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    # print(f'lol{meta_data.values[1,1]}')
    print(test_subjects)
    # if 'hip' in meta_data.values[1,1] or 'trunk' in meta_data.values[1,1]:
    #     print('ok')
    to_remove = np.where(test_subjects == 104)[0]
    test_subjects = np.delete(test_subjects, to_remove)
    to_remove = np.where(test_subjects == 72)[0]
    test_subjects = np.delete(test_subjects, to_remove)
    test_subjects = np.append(test_subjects, 97)
    test_subjects = np.append(test_subjects, 78)
    # else:
    # print('lol')

    print(test_subjects)

    for row in data:
        if row[1] in test_subjects:
            test_idxs.append(row[0])

    print(len(test_idxs))
    print(len(test_subjects))

    return test_idxs, test_subjects


def get_POE_field(info_file):
    data = pd.read_csv(info_file, delimiter=',')
    poe = data.values[np.where(data.values[:, 0] == 'Action:')[0][0], 1]
    return poe.split('_')[-1]

def main(args):
    subjs = np.load(args.idxs)['test_subj']

    test_idxs, subjs = gen_test_idx(args.info_file, subjs)

    print(len(test_idxs))
    print(len(subjs))

    poe = get_POE_field(args.info_file)
    print(poe)
    # assert False

    save_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-{poe}.npz'
    np.savez(save_path, test_idx=test_idxs, test_subj=subjs)
    # train, test, train_subj, val_subj = gen_train_val(args.info_file, 0.9)
    # # train, val, test = gen_train_val_test(args.info_file, 0.8)
    # # print(train.shape)
    # # print(val.shape)
    # # print(test.shape)
    # old_subjs = [4, 13, 14, 42, 45, 52, 73, 90, 92, 93]
    #
    # train_idx, test_idx, train_subj, test_subj = gen_xtra_test(args.info_file, 10, old_subjs)
    #
    # sp = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx2.npz'
    # np.savez(sp, train_idx=train_idx, test_idx=test_idx,
    #          train_subj=train_subj, test_subj=test_subj)
    # data = np.load(sp)
    # train = data['train_idx']
    # test = data['test_idx']
    # train_subj = data['train_subj']
    # test_subj = data['test_subj']
    #
    # print(train)
    # print(test_idx)
    # # print(train_subj)
    # print(test_subj)
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
    parser.add_argument('--idxs', default='')
    parser.add_argument('--old', default='')
    args = parser.parse_args()
    main(args)
