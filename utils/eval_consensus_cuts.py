import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser, ArgumentTypeError
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy
import seaborn

uncert_data = ['femval', 'hip', 'KMFP', 'trunk']

femval_cert = {'1':1, '4':2, '8':2, '13':1, '14':1,'19':1, '23':1, '25':1, '42':1, '45':2, '52':1, '54':1, '67':2, '73':1, '74':2, '78':1, '86':2, '90':1, '92':1, '93':1, '97':2, '103':1}
trunk_cert = {'1':1, '4':3, '8':2, '13':1, '14':1,'19':2, '23':1, '25':1, '42':1, '45':2, '52':2, '54':1, '67':1, '73':1, '74':1, '78':2, '86':1, '90':1, '92':1, '93':1, '97':2, '103':1}
hip_cert = {'1':1, '4':2, '8':1, '13':1, '14':1,'19':2, '23':1, '25':1, '42':2, '45':1, '52':1, '54':1, '67':1, '73':1, '74':2, '78':2, '86':1, '90':1, '92':2, '93':2, '97':3, '103':1}

kmfp_cert = {'1':1, '4':1, '8':1, '13':1, '14':3,'19':1, '23':1, '25':2, '42':1, '45':1, '52':1, '54':1, '67':1, '73':1, '74':1, '78':1, '86':1, '90':2, '92':1, '93':1, '97':1, '103':1}

OUT_DIR = '/home/filipkr/Documents/xjob/consensus_w_cuts'



def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx, 1]
    in_cohort_nbr = data[idx, 3]
    indices = np.where(data[:, 1] == subj)[0]
    idx_same_leg = np.where(data[indices, 3] == in_cohort_nbr)
    print(indices[idx_same_leg])
    print(type(indices[idx_same_leg]))
    return indices[idx_same_leg], subj


def get_POE_field(info_file):
    data = pd.read_csv(info_file, delimiter=',')
    poe = data.values[np.where(data.values[:, 0] == 'Action:')[0][0], 1]
    return poe.split('_')[-1]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, savename=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # plt.show()

    if savename != '':
        plt.savefig(savename)
        plt.close()


def plot_confusion_matrix_mean(all_matrices, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues, savename=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.mean(all_matrices,axis=0)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes,fontsize=14)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:.2f}\n+- {np.std(all_matrices[:,i,j]):.2f}',
                 horizontalalignment="center", fontsize=15,
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.subplots_adjust(bottom=0.124, top=0.938)

    if savename != '':
        plt.savefig(savename)
        plt.close()

def get_subj_idx(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx, 1]
    return subj


def main(args):
   
    # lit = args.lit
    lit = args.root.split('/')[-1]
    print(lit)
    num_folds = 10

    print(lit)

    # if 'trunk' in lit or 'Nadine' in lit:
    #     ds_name = 'trunk_test'
    # elif 'hip' in lit or 'Sigrid' in lit:
    #     ds_name = 'hip_test'
    # elif 'femval' in lit or 'Olga' in lit:
    #     ds_name = 'femval_test'
    # elif 'kmfp' in lit or 'Mikhail' in lit:
    #     ds_name = 'kmfp_test'
    # elif 'fms' in lit or 'Bertrand' in lit:
    #     ds_name = 'fem_med_shank_test'
    # elif 'foot' in lit or 'Harold' in lit:
    #     ds_name = 'foot_test'

    # dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + ds_name + '.npz'
    # dp100 = '/home/filipkr/Documents/xjob/data/datasets/data_' + ds_name + '_len100.npz'
    # info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + ds_name + '-info.txt'
    # dataset = np.load(dp)
    # dataset100 = np.load(dp100)
    # # poe = get_POE_field(info_file)
    
    # # print(os.path.basename(args.root).split('_')[0])
    # # lit = os.path.basename(args.root).split('_')[0]
    # # dp = os.path.join(args.data, 'data_') + 'Herta-Moller.npz'
    # # dataset = np.load(dp)

    # # print(poe)
    # x = dataset['mts']
    # x100 = dataset100['mts']
    # y = dataset['labels']    

    if 'Olga-Tokarczuk' in lit:
        models = ['coral-100-7000', 'xx-coral-100-7000','xx-conf-100-7000','xx-conf-3020', 'xx-conf-9000']
        weights = np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,0,0],[0,0,1/3], [0,1/3,0]])
        # xx-conf-100-11000
        models = ['coral-100-7000', 'xx-coral-100-7000','xx-conf-100-7000', 'xx-conf-9000','xx-conf-100-11000']
        weights = np.array([[1/3,1.25/3,1/3],[1/3,1.25/3,1/3],[1/3,0,0], [0,1.25/3,0],[0,0,1/3]])
        # weights = np.array([[0.5/3,1.5/3,1/3],[0.5/3,1.5/3,1/3],[0.5/3,0,0], [0,1.5/3,0],[0,0,1/3]])
        # models = ['xx-conf-100-3004',
        #           'xx-coral-100-3000']
        # weights = np.array([[1/3, 0, 0],
        #                     [2/3, 1, 1]])
        # models = ['coral-3000', 'xx-conf-100-3004',
        #           'xx-coral-100-3000', 'reg-3030', 'xx-conf-3020']
        # weights = np.array([[1 / 4, 1 / 3, 0.25], [1 / 4, 0, 0],
        #                     [1 / 4, 1 / 3, 0.25], [1/4, 1/3, 0.2],
        #                     [0,0,0.3]])
    elif 'Nadine-Gordimer' in lit:
        models = ['x-incep-600', 'coral-511', 'coral-702', 'coral-x-501',
                  'conf-x-700']
        weights = np.array([[0, 0, 0.8 / 3], [0.3, 0.35, 0.8 / 3], [0.3, 0.35, 0.8 / 3],
                            [0, 0.4, 0.2], [0.4, 0, 0]])
        models = ['coral-100-3100', 'coral-3100', 'xx-conf-100-3100', 'xx-coral-100-3100',
                  'xx-coral-100-3101']
        weights = np.array([[1/4, 1/4, 1/4], [1/4, 1/4, 1/4], [1/4, 0, 0],
                            [0, 1/4, 1/4], [1/4, 1/4, 1/4]])
        weights = np.array([[1/4, 1/3, 1/4], [1/4, 1/3, 1/4], [1/4, 0, 0],
                            [0, 0, 1/4], [1/4, 1/3, 1/4]])
        models = ['coral-100-3100', 'coral-3100', 'xx-conf-100-3100','conf-100-12000', 'xx-coral-100-3100']
        weights = np.array([[1/3, 1.05/3, 1/3], [1/3, 1.05/3, 1/3], [1/3, 0, 0], [0, 1.05/3, 0],
                            [0, 0, 1/3]])
        weights = np.array([[1/3, 1.05/3, 1/3], [1/3, 1.05/3, 1/3], [1/3, 0, 0], [0, 1.05/3, 0],
                            [0, 0, 1/3]])

    elif 'Albert-Camus' in lit:
        models = ['x-incep-600', 'conf-x-800', 'coral-x-500', 'coral-x-501', 'conf-x-707',
                  'conf-x-708']
        weights = np.array([[0.25, 0, 0], [0.35, 0, 0], [0.2, 0.5, 0], [0.2, 0.5, 0],
                            [0, 0, 0.5], [0, 0, 0.5]])
    elif 'Sigrid-Undset' in lit:
        models = ['conf-100', 'coral-100', 'coral-100-100', 'xx-conf-100-1100', 'xx-conf-1200']
        weights = np.array([[0.4, 0, 0], [0.3, 0.3, 0.2], [0.3, 0.35, 0.5], [0, 0, 0.3],
                            [0, 0.35, 0]])
        models = ['conf-100', 'coral-100', 'coral-100-100', 'xx-conf-1100', 'xx-conf-1200']
        weights = np.array([[0.4, 0, 0], [0.3, 0.3, 0.2], [0.3, 0.3, 0.4], [0, 0, 0.4],
                            [0, 0.4, 0]])

        models = ['conf-100', 'conf-100-10000', 'coral-100-100', 'xx-conf-1200', 'conf-10001', 'coral-10000', 'coral-100-10000','xx-coral-100-10003','coral-100-10003']
        weights = np.array([[0.1, 0, 0], [0.3,0,0], [0.1, 0.1, 0.1],
                            [0, 0.1, 0],[0,0.3,0], [0.1,0.1,0.1],[0.1,0.1,0.1], [0,0,0.35], [0.3,0.3,0.35]])
        models = ['conf-100-10000', 'coral-100-100', 'conf-10001', 'coral-10000', 'coral-100-10000','xx-coral-100-10003','coral-100-10003']
        weights = np.array([[0.4,0,0], [0.1, 0.1, 0.1],[0,0.4,0], [0.1,0.1,0.1],[0.1,0.1,0.1], [0,0,1.1*0.35], [0.3,0.3,0.35]])
        models = ['conf-100-10000', 'conf-10001', 'xx-coral-100-10003']
        weights = np.array([[1,0,0], [0,1,0], [0,0,1.5]])
        models = ['coral-13000','coral-100-13000', 'conf-100-10000', 'coral-100-100', 'conf-10001', 'xx-coral-100-10003','coral-100-10003']
        weights = np.array([[0.15,0.15,0.15],[0.15,0.15,0.15],[0.4,0,0], [0.15, 0.15, 0.15],[0,0.4,0], [0,0,0.4], [0.15,0.15,0.15]])
        models = ['coral-100-13000', 'coral-13000', 'conf-100-10000', 'conf-10001', 'xx-coral-100-10003']
        weights = np.array([[1/4,1.08/4,1.5/4],[1/4,1.08/4,1.5/4],[1/2,0,0], [0,1.08/2,0], [0,0,1.5/2]])
        # models = ['conf-100-10000', 'conf-10001', 'xx-coral-100-10003']
        # weights = np.array([[1,0,0], [0,1,0], [0,0,1.5]])
        # models =['coral-100-13000']
        # weights = np.array([[1,1,1]])


    elif 'Mikhail-Sholokhov' in lit:
        models = ['conf-3020', 'inception-3010', 'xx-conf-100-3015', 'xx-conf-3010', 'xx-conf-3020', 'xx-conf-3025', 'xx-inception-100-3010', 'xx-inception-3010']
        weights = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0, 0.2], [0.4, 0, 0],
                            [0, 0.2, 0], [0,0,0.2], [0.2,0.2,0.2], [0.2,0.2,0.2]])
        models = ['inception-3010', 'xx-conf-100-3015', 'xx-conf-3010', 'conf-100-13000', 'xx-conf-3025', 'xx-inception-100-3010', 'xx-inception-3010']
        weights = np.array([[0.25, 0.2, 0.2], [0, 0, 0.2], [0.25, 0, 0],
                            [0, 0.4, 0], [0,0,0.2], [0.25,0.2,0.2], [0.25,0.2,0.2]])
        # models = ['inception-3010', 'xx-conf-100-3015', 'xx-conf-3010', 'conf-100-13000', 'xx-conf-3025']
        # weights = np.array([[0.5, 0.95*0.5, 1/3], [0, 0, 1/3], [0.5, 0, 0],
        #                     [0, 0.95*0.5, 0], [0,0,1/3]])
        models = ['inception-3010', 'xx-inception-3010', 'xx-conf-3010', 'conf-100-13000', 'xx-conf-3025']
        weights = np.array([[1/3, 0.95*1/3, 1/3],[1/3, 0.95*1/3, 1/3], [1/3, 0, 0],
                            [0, 0.95*1/3, 0], [0,0,1/3]])

    elif 'Isaac-Bashevis-Singer' in lit:
        models = ['coral-100-11', 'coral-100-10', 'xx-conf-100-11', 'conf-15', 'xx-coral-100-10']
        weights = np.array([[1/3,0.9/3,1/3], [1/3,0.9/3,1/3],[1/3,0,0],[0,0.95/3,0],[0,0,1/3]])
    elif 'Bertrand-Russell' in lit:
        # models = ['coral-0', 'coral-1', 'coral-2']
        # weights = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
        models = ['coral-0', 'coral-1', 'coral-2', 'conf-0', 'conf-1', 'conf-2']
        weights = np.array([[1/4, 1/4, 1/4], [1/4, 1/4, 1/4], [1/4, 1/4, 1/4],
                            [1/4, 0, 0], [0, 1/4, 0], [0, 0, 1/4]])
    elif 'Harold-Pinter' in lit:
        models = ['coral-0', 'coral-1', 'coral-100-0', 'conf-100-0',
                  'coral-1-conf', 'conf-2']
        weights = np.array([[1/4, 1/4, 1/4], [1/4, 1/4, 1/4], [1/4, 1/4, 1/4],
                            [1/4, 0, 0], [0, 1/4, 0], [0, 0, 1/3]])
    elif lit == 'trunk_con':
        models = ['coral-2', 'conf-100-0', 'conf-1', 'xx-conf-100-2']
        models = ['coral-2', 'conf-100-0', 'xx-conf-100-2']
        # models = ['coral-2']
        weights = np.array([[1/2,0.9/3,1.1/2], [1.1/2, 0, 0], [0,1.9/3,0], [0,0,1.1/2]])
        weights = np.array([[0.7/2,0.9,1.1/2], [1.1/2, 0, 0], [0,0,1.1/2]])
        # models = ['coral-2','xx-conf-100-2']
        # weights = np.array([[0.7/2,0.9/3,1.1/2], [1.1/2, 0, 0], [0,0,1.1/2]])
        # weights = np.array([[1,0.9,1]])
    elif lit == 'hip_con':
        models = ['coral-2', 'conf-100-0', 'xx-conf-1', 'xx-conf-100-2']
        models = ['coral-2', 'conf-100-0', 'xx-conf-100-2']
        weights = np.array([[1/2,0.85/2,1.1/2], [1.2/2, 0, 0], [0,0.9/2,0],[0,0,1.1/2]])
        weights = np.array([[1/2,0.9,1.1/2], [1.2/2, 0, 0], [0,0,1.1/2]])
    elif lit == 'femval_con':
        models = ['coral-2', 'conf-100-0', 'conf-1', 'conf-2']
        models = ['coral-2', 'conf-100-0', 'conf-2']
        weights = np.array([[1/2,0.7/2,0.8/2], [1/2, 0, 0], [0,1.3/2,0], [0,0,1.5/2]])
        weights = np.array([[1/2,1,1.1/2], [1/2, 0, 0], [0,0,1.1/2]])
        # models = ['coral-2']
        # weights = np.array([1,1,1])
    elif lit == 'kmfp_con':
        models = ['coral-3', 'xx-conf-100-0']
        weights = np.array([[0,0.85,1.5], [1,0,0]])
    elif lit == 'fms_con':
        models = ['coral-3', 'conf-0', 'conf-100-1', 'xx-conf-2']
        models = ['coral-3', 'conf-0', 'xx-conf-2']
        weights = np.array([[1/3,0.9/2,1/2], [2/3, 0, 0], [0,1/2,0], [0,0,1.0/2]])
        weights = np.array([[1/3,0.9,1.1/2], [2/3, 0, 0], [0,0,1.05/2]])
    elif lit == 'foot_con':
        models = ['coral-3', ':::xx-conf-0', 'conf-2']
        weights = np.array([[1.1/4,0.8,0.8/2], [3/4, 0, 0], [0,0,0.8/2]])
    elif lit == 'trunk_aug':
        models = ['coral-1', 'conf-0', 'conf-100-1', 'conf-2']
        weights = np.array([[0.5,0.5,0.5], [0.5,0,0],[0,0.5,0],[0,0,0.5]])
        # weights = np.array([[0.6,0.6,0.6], [0.4,0,0],[0,0.4,0],[0,0,0.4]])
    elif lit == 'hip_aug':
        models = ['coral-1', 'conf-0', 'conf-100-1', 'conf-100-2']
        weights = np.array([[0.5,0.5,0.6], [0.5, 0, 0], [0, 0.5, 0], [0,0,0.5]])
        # weights = np.array([[0.6,0.6,0.7], [0.4, 0, 0], [0, 0.4, 0], [0,0,0.4]])
    elif lit == 'femval_aug':
        models = ['coral-1', 'conf-0', 'conf-1', 'conf-2']
        weights = np.array([[0.5,0.5,0.8],[0.5,0,0], [0,0.5,0],[0,0,0.3]])
        weights = np.array([[0.6,0.6,1],[0.4,0,0], [0,0.4,0],[0,0,0.1]])
    elif lit == 'kmfp_aug':
        models = ['coral-1', 'conf-0', 'conf-2']
        weights = np.array([[0.5,1,0.8], [0.5,0,0], [0,0,0.4]])
        weights = np.array([[0.6,1.1,0.9], [0.4,0,0], [0,0,0.3]])
    elif lit == 'fms_aug':
        models = ['coral-1', 'conf-0', 'conf-100-1']
        weights = np.array([[0.5,0.5,1.1], [0.5,0,0], [0,0.5,0]])
        weights = np.array([[0.6,0.6,1.1], [0.4,0,0], [0,0.4,0]])
    elif lit == 'foot_aug':
        models = ['coral-1', 'conf-100-0', 'conf-100-2']
        weights = np.array([[0.5,1,0.8], [0.5,0,0],[0,0,0.4]])
        weights = np.array([[0.6,1,0.9], [0.4,0,0],[0,0,0.3]])

    print(args.root)
    print(models)

    ensembles = [os.path.join(args.root, i) for i in models] if 'ensembles' in args.root else [
        args.root + str(i) for i in models]

    

    if args.confusion:
        path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_confusion-test.csv'
        confusion_file = open(path, 'w')


    if 'trunk' in lit or 'Nadine' in lit:
        ds_name = 'trunk_test'
    elif 'hip' in lit or 'Sigrid' in lit:
        ds_name = 'hip_test'
    elif 'femval' in lit or 'Olga' in lit:
        ds_name = 'femval_test'
    elif 'kmfp' in lit or 'Mikhail' in lit:
        ds_name = 'kmfp_test'
    elif 'fms' in lit or 'Bertrand' in lit:
        ds_name = 'fem_med_shank_test'
    elif 'foot' in lit or 'Harold' in lit:
        ds_name = 'foot_test'
    
    cuts = ['', '-cut5', '-cut10', '-cut15', '-cut20', '-cut_rate0.05',
            '-cut_rate0.1', '-cut_rate0.15']

    cuts = ['', '-cut5', '-cut20', '-cut_rate0.05', '-cut_rate0.15']
    cuts = ['', '-cut5', '-cut_rate0.05', '-cut_rate0.1']
    # cuts = ['', '-cut5', '-cut_rate0.05']
    cut_results = {}

    dp = f'/home/filipkr/Documents/xjob/data/datasets/data_{ds_name}.npz'

    dataset = np.load(dp)
    y = dataset['labels']
    all_cut_preds = np.zeros((num_folds, len(cuts), len(y), 3))
    # corrects
    lens = []
    
    for cut_index, cut in enumerate(cuts):

        f1 = []
        acc = []
        acc_thresh = []
        f1_thresh = []
        rep_acc = []
        rep_f1 = []
        precision = []
        precision_thresh = []
        precision_rep = []
        recall = []
        recall_thresh = []
        recall_rep = []

        cnf_all_folds = np.zeros((3, 3))
        cnf_all_folds_thresh = np.zeros((3, 3))
        all_confusions = np.zeros((num_folds,3,3))
        all_confusions_thresh = np.zeros((num_folds,3,3))
        all_confusions_ignored = np.zeros((num_folds,3,3))
        all_rep_cnf = np.zeros((num_folds,3,3))

        prob_reps = -1* np.ones((num_folds,150,3,3))
        prob_reps_pred = -1* np.ones((num_folds,150,3,3))

        individual_accs = np.zeros((num_folds, len(models)))
        individual_f1 = np.zeros((num_folds, len(models)))

        dp = f'/home/filipkr/Documents/xjob/data/datasets/data_{ds_name}{cut}.npz'
        dp100 = f'/home/filipkr/Documents/xjob/data/datasets/data_{ds_name}_len100{cut}.npz'
        info_file = f'/home/filipkr/Documents/xjob/data/datasets/data_{ds_name}-info.txt'

        print(dp)
        print(dp100)
        dataset = np.load(dp)
        dataset100 = np.load(dp100)

        # print(poe)
        x = dataset['mts']
        x100 = dataset100['mts']
        y = dataset['labels']
        lens.append(len(y))

        for fold in range(1, num_folds + 1):

            # if fold == 10 and lit == 'Sigrid-Undset':
            #     weights = np.array([[1/4,1.7/4,1.5/4],[1/4,1.7/4,1.5/4],[1/2,0,0], [0,1.7/2,0], [0,0,1.5/2]])
            #     print('LOLOLOLOLOLOLOL')
            paths = [os.path.join(
                root, f'model_fold_{fold}.hdf5') for root in ensembles]

            all_probs = np.zeros((len(models), x.shape[0], 3))

            corr_rep = []
            pred_rep = []
            # '''note'''

            for model_i, model_path in enumerate(paths):

                input = x100 if '-100-' in model_path else x
                # print(model_path)
                model = keras.models.load_model(model_path, custom_objects={
                                                'CoralOrdinal': coral.CoralOrdinal,
                                                'OrdinalCrossEntropy':
                                                coral.OrdinalCrossEntropy,
                                                'MeanAbsoluteErrorLabels':
                                                coral.MeanAbsoluteErrorLabels,
                                                'ConfusionCrossEntropy':
                                                ConfusionCrossEntropy})
                # x = x_test
                result = model.predict(input)
                probs = coral.ordinal_softmax(
                    result).numpy() if 'coral' in model_path else result

                probs = probs * weights[model_i, ...]

                all_probs[model_i, ...] = probs

                subject_indices = []

                y_corr = []
                y_pred_comb = []
                # for i in test_subj:
                # for i in ind_t['test_idx']:
                for i in range(len(y)):
                    # print(int(y[i]))

                    if i not in subject_indices:
                        subject_indices, _ = get_same_subject(info_file, i)

                        y_subj = y[subject_indices]
                        pred_subj = probs[subject_indices, ...]

                        y_corr.append(int(np.median(y_subj)))
                        y_pred_comb.append(np.argmax(np.mean(pred_subj, axis=0)))

                individual_accs[fold-1, model_i] = accuracy_score(y_corr,y_pred_comb)
                individual_f1[fold-1, model_i] = f1_score(y_corr, y_pred_comb,
                                    labels=[0, 1, 2], average='macro')

            subject_indices = []
            preds = []
            truth = []

            preds_thresh = []
            truth_thresh = []
            preds_ignored = []
            truth_ignored = []

            ensemble_probs = np.sum(all_probs, axis=0)
            all_cut_preds[fold-1, cut_index, ...] = ensemble_probs
            # print(ensemble_probs)

            ensemble_probs = (ensemble_probs > 0.2) * ensemble_probs

            # print('ENSEMBLE')
            # print(ensemble_probs[test_subj, ...])
            # print(np.mean(ensemble_probs[test_subj, ...], axis=0))

            # if fold == 10:
                # print(ensemble_probs)

            hist_index = 0
            # for i in test_subj:
            # for i in ind_t['test_idx']:
            # print(len(y))
            # print(ensemble_probs.shape)
            print()
            for i in range(len(y)):
                # print(i)
                prob_reps[fold-1, hist_index, int(y[i]), :] = ensemble_probs[i, ...]
                predicted_class = np.argmax(ensemble_probs[i, ...])

                prob_reps_pred[fold-1, hist_index, int(y[i]), predicted_class] = ensemble_probs[i, predicted_class]
                hist_index += 1

                corr_rep.append(y[i])
                pred_rep.append(np.argmax(ensemble_probs[i, ...]))


            all_rep_cnf[fold-1, ...] = confusion_matrix(corr_rep, pred_rep, labels=[0,1,2])
            print(all_rep_cnf[fold-1, ...])
            rep_f1.append(f1_score(corr_rep, pred_rep,
                                labels=[0, 1, 2], average='macro'))
            rep_acc.append(accuracy_score(corr_rep, pred_rep))
            precision_rep.append(precision_score(corr_rep, pred_rep,
                                labels=[0, 1, 2], average='macro'))
            recall_rep.append(recall_score(corr_rep, pred_rep,
                                labels=[0, 1, 2], average='macro'))

            hist_index = 0
            # for i in test_subj:
            # for i in ind_t['test_idx']:
            for i in range(len(y)):
                if i not in subject_indices:
                    subject_indices, global_ind = get_same_subject(info_file, i)

                    y_subj = y[subject_indices]
                    pred_subj = ensemble_probs[subject_indices, ...]
                    # np.sum(
                    #    all_probs[:, subject_indices, ...], axis=0)  # / len(models)
                    # print(np.sum(pred_subj, axis=0))
                    # if np.sum(pred_subj) > 0.8:
                    summed = np.mean(pred_subj, axis=0)

                    test_median = True

                    if test_median:
                        max_like = np.max(pred_subj, axis=1)
                        pred = np.argmax(pred_subj, axis=1)

                        ''' test
                        conf_samples = max_like > 0.35
                        summed = np.sum(pred_subj[conf_samples,:], axis=0)/np.sum(conf_samples)
                        '''

                        corr_combined = int(np.ceil(np.median(y_subj)))
                        pred_combined = int(np.ceil(np.median(pred)))

                        pred_combined = int(np.argmax(np.mean(pred_subj, axis=0)))
                        preds.append(int(np.round(pred_combined)))
                        truth.append(int(np.round(corr_combined)))

                        # thres = 0.35 if lit == 'Sigrid-Undset' or lit == 'Mikhail-Sholokhov' else 0.4
                        if lit == 'Sigrid-Undset' or lit == 'Mikhail-Sholokhov':
                            thres = 0.35
                        # elif lit == 'Bertrand-Russell':
                        #     thres = 0.5
                        else:
                            thres = 0.4
                        # thres = 0
                        # thres = 0.4
                        if np.max(summed) > thres:
                            preds_thresh.append(int(np.round(pred_combined)))
                            truth_thresh.append(int(np.round(corr_combined)))
                        else:
                            preds_ignored.append(int(np.round(pred_combined)))
                            truth_ignored.append(int(np.round(corr_combined)))

                       
                        hist_index += 1

            ensemble_acc = accuracy_score(truth, preds)
            # print('TITTTAAAAAAA HAAAAR')
            # print(len(preds))

            labels = np.sort(np.unique(np.append(truth, preds)))
            f1.append(f1_score(truth, preds, labels=labels, average='macro'))
            acc.append(accuracy_score(truth, preds))
            precision.append(precision_score(truth, preds, labels=labels, average='macro'))
            recall.append(recall_score(truth, preds, labels=labels, average='macro'))

            labels = np.sort(np.unique(np.append(truth_thresh, preds_thresh)))
            f1_thresh.append(f1_score(truth_thresh, preds_thresh, labels=labels, average='macro'))
            acc_thresh.append(accuracy_score(truth_thresh, preds_thresh))
            precision_thresh.append(precision_score(truth_thresh, preds_thresh, labels=labels, average='macro'))
            recall_thresh.append(recall_score(truth_thresh, preds_thresh, labels=labels, average='macro'))


            cnf = confusion_matrix(truth_thresh, preds_thresh, labels=[0, 1, 2])
            cnf_all_folds_thresh = cnf_all_folds_thresh + cnf
            all_confusions_thresh[fold-1,...] = cnf
            cnf = confusion_matrix(truth_ignored, preds_ignored, labels=[0, 1, 2])
            all_confusions_ignored[fold-1,...] = cnf
            cnf = confusion_matrix(truth, preds, labels=[0, 1, 2])
            cnf_all_folds = cnf_all_folds + cnf
            all_confusions[fold-1,...] = cnf

            if args.confusion and False:
                confusion_file.write(f'Confusion for fold {fold},,\n')
                cnf_str = str(cnf).replace(' [', '')
                cnf_str = cnf_str.replace('[', '')
                cnf_str = cnf_str.replace(']','')
                cnf_str = cnf_str.replace(' ',',')
                confusion_file.write(cnf_str)
                confusion_file.write('\n,,\n')

            print(f'individual accs: {individual_accs[fold-1,...]}')
            print(f'ensemble acc: {ensemble_acc}')
            print(f'ensemble confusion:\n{cnf}')
            print(f'f1 score: {f1[-1]}')

        cut_results[cut] = {'acc_thresh': acc_thresh, 'acc': acc,
                            'f1_thresh': f1_thresh, 'f1': f1, 'rep_f1': rep_f1,
                            'rep_acc': rep_acc, 'recall_rep': recall_rep,
                            'recall': recall, 'recall_thresh': recall_thresh,
                            'precision_rep': precision_rep,
                            'precision': precision,
                            'precision_thresh': precision_thresh,
                            'all_confusions': all_confusions,
                            'all_confusions_thresh': all_confusions_thresh,
                            'all_rep_cnf': all_rep_cnf,
                            'all_confusions_ignored': all_confusions_ignored}


    print('----------------------')

    for cut in cut_results.keys():
        print(f'CUT: {cut}')
        acc_thresh = cut_results[cut]['acc_thresh']
        acc = cut_results[cut]['acc']
        f1_thresh = cut_results[cut]['f1_thresh']
        f1 = cut_results[cut]['f1']
        rep_f1 = cut_results[cut]['rep_f1']
        rep_acc = cut_results[cut]['rep_acc']
        recall_rep = cut_results[cut]['recall_rep']
        recall = cut_results[cut]['recall']
        recall_thresh = cut_results[cut]['recall_thresh']
        precision_rep = cut_results[cut]['precision_rep']
        precision = cut_results[cut]['precision']
        precision_thresh = cut_results[cut]['precision_thresh']
        all_confusions = cut_results[cut]['all_confusions']
        all_confusions_thresh = cut_results[cut]['all_confusions_thresh']
        all_rep_cnf = cut_results[cut]['all_rep_cnf']
        all_confusions_ignored = cut_results[cut]['all_confusions_ignored']

        fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.1})
        fig.suptitle(f'Accuracies, {cut}', fontsize=18)
        plt.subplots_adjust(bottom=0.12, top=0.89)
        p = seaborn.histplot(data=acc_thresh,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[0])
        plt.sca(axs[0])
        plt.yticks(fontsize=12)
        p.set_ylabel(f'Threshold', fontsize=14)
        axs[0].axes.xaxis.set_ticklabels([])

        p = seaborn.histplot(data=acc,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[1])
        plt.sca(axs[1])
        plt.yticks(fontsize=12)
        p.set_ylabel(f'Combined', fontsize=14)
        # p.title(f'CUT: {cut}')
        axs[1].axes.xaxis.set_ticklabels([])
        p = seaborn.histplot(data=rep_acc,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[2])
        plt.sca(axs[2])
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=15)
        p.set_ylabel(f'Repetitions', fontsize=14)
        p.set_xlabel('Accuracies', fontsize=15)
        # p.title(f'CUT: {cut}')

        fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.1})
        fig.suptitle(f'F1 scores, {cut}', fontsize=18)
        plt.subplots_adjust(bottom=0.12, top=0.89)
        p = seaborn.histplot(data=f1_thresh,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[0])
        plt.sca(axs[0])
        plt.yticks(fontsize=12)
        p.set_ylabel(f'Threshold', fontsize=14)
        axs[0].axes.xaxis.set_ticklabels([])

        p = seaborn.histplot(data=f1,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[1])
        plt.sca(axs[1])
        plt.yticks(fontsize=12)
        p.set_ylabel(f'Combined', fontsize=14)
        axs[1].axes.xaxis.set_ticklabels([])

        p = seaborn.histplot(data=rep_f1,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[2])
        plt.sca(axs[2])
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=15)
        p.set_ylabel(f'Repetitions', fontsize=14)
        p.set_xlabel('F1 scores', fontsize=15)

        print('---------------------')
        # print(f'ensemble confusion:\n{cnf_all_folds}')
        # print(f'f1: {f1}')
        print(f'combined metrics, {cut}')
        print(f'f1: {np.mean(f1)} +- {np.std(f1)}')
        print(f'f1 thresh: {np.mean(f1_thresh)} +- {np.std(f1_thresh)}')
        print('\n')
        # print(f'acc: {acc}')
        print(f'acc: {np.mean(acc)} +- {np.std(acc)}')
        print(f'acc: {np.mean(acc_thresh)} +- {np.std(acc_thresh)}')
        print('--------------------')
        print('-----------------')
        print('-----------------')

        # print(f'rep f1: {rep_f1}')
        print(f'repetition metrics, {cut}')
        print(f'rep f1: {np.mean(rep_f1)} +- {np.std(rep_f1)}')
        # print(f'rep acc: {rep_acc}')
        print(f'rep acc: {np.mean(rep_acc)} +- {np.std(rep_acc)}')

        print('----')
        print(f'Recall, rep, comb, thresh, {cut}')
        print(f'{np.mean(recall_rep)} +- {np.std(recall_rep)}')
        print(f'{np.mean(recall)} +- {np.std(recall)}')
        print(f'{np.mean(recall_thresh)} +- {np.std(recall_thresh)}')
        # print('Recall certainties')
        # print(f'{np.mean(recall_c1)} +- {np.std(recall_c1)}')
        # print(f'{np.mean(recall_c2)} +- {np.std(recall_c2)}')
        # print(f'{np.mean(recall_c3)} +- {np.std(recall_c3)}')
        print('----')
        print(f'Precision, rep, comb, thresh, {cut}')
        print(f'{np.mean(precision_rep)} +- {np.std(precision_rep)}')
        print(f'{np.mean(precision)} +- {np.std(precision)}')
        print(f'{np.mean(precision_thresh)} +- {np.std(precision_thresh)}')

        print('-----------------------')

        if args.confusion:
            confusion_file.write(f'Confusions for all repetitions,,\n')
            for fold, c in enumerate(all_rep_cnf):
                confusion_file.write(f'matrix for fold {fold+1},,\n')
                cnf_str = str(c.astype(int)).replace(' [', '')
                cnf_str = str(cnf_str).replace('[ ', '')
                cnf_str = str(cnf_str).replace('[', '')
                cnf_str = str(cnf_str).replace(']', '')
                cnf_str = cnf_str.replace(' ',',')
                confusion_file.write(cnf_str)
                confusion_file.write('\n')
            confusion_file.write(f'Average of above,,\n')
            cnf_str = str(np.mean(all_rep_cnf,axis=0)).replace(' [', '')
            cnf_str = str(cnf_str).replace('[ ', '')
            cnf_str = str(cnf_str).replace('[', '')
            cnf_str = str(cnf_str).replace(']', '')
            cnf_str = cnf_str.replace(' ',',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n\n')

            confusion_file.write(f'Confusions for combined scores,,\n')
            for fold, c in enumerate(all_confusions):
                confusion_file.write(f'matrix for fold {fold+1},,\n')
                cnf_str = str(c.astype(int)).replace(' [', '')
                cnf_str = str(cnf_str).replace('[ ', '')
                cnf_str = str(cnf_str).replace('[', '')
                cnf_str = str(cnf_str).replace(']', '')
                cnf_str = cnf_str.replace(' ',',')
                confusion_file.write(cnf_str)
                confusion_file.write('\n')
            confusion_file.write(f'Average of above,,\n')
            cnf_str = str(np.mean(all_confusions,axis=0)).replace(' [', '')
            cnf_str = str(cnf_str).replace('[ ', '')
            cnf_str = str(cnf_str).replace('[', '')
            cnf_str = str(cnf_str).replace(']', '')
            cnf_str = cnf_str.replace(' ',',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n\n')

            confusion_file.write(f'Confusions for combined scores with threshold,,\n')
            for fold, c in enumerate(all_confusions_thresh):
                confusion_file.write(f'matrix for fold {fold+1},,\n')
                cnf_str = str(c.astype(int)).replace(' [', '')
                cnf_str = str(cnf_str).replace('[ ', '')
                cnf_str = str(cnf_str).replace('[', '')
                cnf_str = str(cnf_str).replace(']', '')
                cnf_str = cnf_str.replace(' ',',')
                confusion_file.write(cnf_str)
                confusion_file.write('\n')
            confusion_file.write(f'Average of above,,\n')
            cnf_str = str(np.mean(all_confusions_thresh,axis=0)).replace(' [', '')
            cnf_str = str(cnf_str).replace('[ ', '')
            cnf_str = str(cnf_str).replace('[', '')
            cnf_str = str(cnf_str).replace(']', '')
            cnf_str = cnf_str.replace(' ',',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n\n')

            confusion_file.close()

        plot_confusion_matrix_mean(all_confusions, [0,1,2], title=f'Combined scores, {cut}')
        plot_confusion_matrix_mean(all_confusions_thresh, [0,1,2], title=f'Combined scores, with threshold, {cut}')
        plot_confusion_matrix_mean(all_rep_cnf, [0,1,2], title=f'Repetitions, {cut}')
        plot_confusion_matrix_mean(all_confusions_ignored, [0,1,2], title=f'Ignored, {cut}')

    # hist_save_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_prob_matrices-test.npz'
    # np.savez(hist_save_path, reps=prob_reps, reps_max=prob_reps_pred)

    # stuff_sp = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '-stat-stuff.npz'
    # np.savez(stuff_sp, rep_acc=rep_acc, rep_f1=rep_f1, comb_acc=acc, comb_f1=f1, thresh_acc=acc_thresh, thresh_f1=f1_thresh, acc_ind=individual_accs, f1_ind=individual_accs)
    print(lens)

    cut_preds = np.mean(all_cut_preds, axis=1)
    # print(cut_preds[0, ...])
    subject_indices = []

    # preds_reps = []
    acc_reps = []
    f1_reps = []
    prec_reps = []
    recall_reps = []
    acc_comb = []
    f1_comb = []
    prec_comb = []
    recall_comb = []

    conf_reps = np.zeros((num_folds, 3, 3))
    conf_combs = np.zeros((num_folds, 3, 3))
    plt.close('all')
    for f in range(num_folds):
        
        preds_combined = []
        truth_combined = []
        preds_reps = np.argmax(cut_preds[f, ...], axis=1)

        labels = np.sort(np.unique(np.append(y, preds_reps)))

        acc_reps.append(accuracy_score(y, preds_reps))
        f1_reps.append(f1_score(y, preds_reps, labels=labels, average='macro'))
        prec_reps.append(precision_score(y, preds_reps, labels=labels, average='macro'))
        recall_reps.append(recall_score(y, preds_reps, labels=labels, average='macro'))
        
        subject_indices = []
        for i in range(len(y)):
            print(i)
            try:
                if not i in subject_indices:
                    subject_indices, global_ind = get_same_subject(info_file, i)
                    # if type(si) == list:
                    #     subject_indices.extend(si[0])
                    # else:
                    #     subject_indices.extend(si[0])
                    y_subj = y[subject_indices]
                    pred_subj = cut_preds[f, subject_indices, ...]

                    
                    corr_combined = int(np.ceil(np.median(y_subj)))
                    pred_combined = int(np.argmax(np.mean(pred_subj, axis=0)))
                    preds_combined.append(int(np.round(pred_combined)))
                    truth_combined.append(int(np.round(corr_combined)))
            except ValueError:
                print(i)
                print(subject_indices)
                print(i in subject_indices)
                print(i not in subject_indices)
        acc_comb.append(accuracy_score(truth_combined, preds_combined))
        f1_comb.append(f1_score(truth_combined, preds_combined, labels=labels, average='macro'))
        prec_comb.append(precision_score(truth_combined, preds_combined, labels=labels, average='macro'))
        recall_comb.append(recall_score(truth_combined, preds_combined, labels=labels, average='macro'))

        cnf_rep = confusion_matrix(y, preds_reps, labels=[0, 1, 2])
        print(cnf_rep)
        conf_reps[f, ...] = cnf_rep
        cnf_comb = confusion_matrix(truth_combined, preds_combined, labels=[0, 1, 2])
        print(cnf_comb)
        conf_combs[f, ...] = cnf_comb
    print()
    print('--------------')
    print(f'acc rep: {np.mean(acc_reps)} +- {np.std(acc_reps)}')
    print(f'acc comb: {np.mean(acc_comb)} +- {np.std(acc_comb)}')
    print()
    print(f'f1 rep: {np.mean(f1_reps)} +- {np.std(f1_reps)}')
    print(f'f1 comb: {np.mean(f1_comb)} +- {np.std(f1_comb)}')
    print()
    print(f'prec rep: {np.mean(prec_reps)} +- {np.std(prec_reps)}')
    print(f'prec comb: {np.mean(prec_comb)} +- {np.std(prec_comb)}')
    print()
    print(f'recall rep: {np.mean(recall_reps)} +- {np.std(recall_reps)}')
    print(f'recall comb: {np.mean(recall_comb)} +- {np.std(recall_comb)}')

    plot_confusion_matrix_mean(conf_reps, [0,1,2], title='Repetition scores',
                               savename=os.path.join(OUT_DIR, 
                                                     f'{lit}/{lit}-conf-rep.png'))
    plot_confusion_matrix_mean(conf_combs, [0,1,2], title='Combined scores',
                               savename=os.path.join(OUT_DIR, 
                                                     f'{lit}/{lit}-conf-comb.png'))

    fig, axs = plt.subplots(2, gridspec_kw={'hspace':0.1})
    fig.suptitle(f'F1 scores', fontsize=18)
    plt.subplots_adjust(bottom=0.12, top=0.89)
    p = seaborn.histplot(data=f1_comb,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[0])
    plt.sca(axs[0])
    plt.yticks(fontsize=12)
    p.set_ylabel(f'Combined', fontsize=14)
    axs[0].axes.xaxis.set_ticklabels([])

    p = seaborn.histplot(data=f1_reps,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[1])
    plt.sca(axs[1])
    plt.yticks(fontsize=12)
    p.set_ylabel(f'Repetitions', fontsize=14)
    # axs[1].axes.xaxis.set_ticklabels([])
    plt.savefig(os.path.join(OUT_DIR, f'{lit}/{lit}-f1-hist.png'))

    fig, axs = plt.subplots(2, gridspec_kw={'hspace':0.1})
    fig.suptitle(f'Accuracy scores', fontsize=18)
    plt.subplots_adjust(bottom=0.12, top=0.89)
    p = seaborn.histplot(data=acc_comb,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[0])
    plt.sca(axs[0])
    plt.yticks(fontsize=12)
    p.set_ylabel(f'Combined', fontsize=14)
    axs[0].axes.xaxis.set_ticklabels([])

    p = seaborn.histplot(data=acc_reps,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[1])
    plt.sca(axs[1])
    plt.yticks(fontsize=12)
    p.set_ylabel(f'Repetitions', fontsize=14)
    # axs[1].axes.xaxis.set_ticklabels([])
    plt.savefig(os.path.join(OUT_DIR, f'{lit}/{lit}-acc-hist.png'))

    with open(os.path.join(OUT_DIR, f'{lit}/{lit}-confs-reps.txt'), 'w') as fo:
        fo.write(np.array2string(conf_reps))
    with open(os.path.join(OUT_DIR, f'{lit}/{lit}-confs-combined.txt'), 'w') as fo:
        fo.write(np.array2string(conf_combs))

    with open(os.path.join(OUT_DIR, f'{lit}/{lit}-stats.txt'), 'w') as fo:
        fo.write(f'accuracy repettitions: {np.mean(acc_reps)} +- {np.std(acc_reps)}\n')
        fo.write(f'accuracy combined scores: {np.mean(acc_comb)} +- {np.std(acc_comb)}\n\n')
        fo.write(f'f1 repettitions: {np.mean(f1_reps)} +- {np.std(f1_reps)}\n')
        fo.write(f'f1 combined scores: {np.mean(f1_comb)} +- {np.std(f1_comb)}\n\n')
        fo.write(f'precision repettitions: {np.mean(prec_reps)} +- {np.std(prec_reps)}\n')
        fo.write(f'precision combined scores: {np.mean(prec_comb)} +- {np.std(prec_comb)}\n\n')
        fo.write(f'recall repettitions: {np.mean(recall_reps)} +- {np.std(recall_reps)}\n')
        fo.write(f'recall combined scores: {np.mean(recall_comb)} +- {np.std(recall_comb)}\n\n')

        
    plt.show()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--uncert', default='')
    parser.add_argument('--confusion', type=str2bool,
                        nargs='?', default=False)
    # parser.add_argument('--lit', type=str)
    args = parser.parse_args()
    main(args)
