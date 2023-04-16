import numpy as np
from argparse import ArgumentParser
from tsfresh import extract_relevant_features
import pandas as pd
from sklearn import discriminant_analysis
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import itertools


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

def idx_same_subject(meta, subject):
    # subj = meta[idx,1]
    indices = np.where(meta[:, 1] == subject)[0]
    return indices

def read_meta_data(info_file):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    meta_data = np.array(meta_data.values[first_data:, :4], dtype=int)
    return meta_data

def get_POE_field(info_file):
    data = pd.read_csv(info_file, delimiter=',')
    poe = data.values[np.where(data.values[:, 0] == 'Action:')[0][0], 1]
    return poe.split('_')[-1]

def main(args):
    # data = np.load(args.data)
    # X_np = data['mts']
    # Y_np = data['labels']

    if 'ensemble-test' in args.root:
        lit = args.root.split('/')[-1]
    else:
        lit = os.path.basename(args.root).split('_')[0]
    print(lit)
    num_folds = 5
    if lit[-2].isdigit():
        assert lit[-1].isdigit()
        num_folds = int(lit[-2:])
        lit = lit[:-2]
        print(num_folds)
        print(lit)
    elif lit[-1].isdigit():
        print('lol')
        assert False
    print(lit)

    data_path = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '_len100.npz'
    data = np.load(data_path)
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info-fix.txt'
    poe = get_POE_field(info_file)
    idx_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-{poe}.npz'
    idx_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-fix.npz'
    test_idx = np.load(idx_path)['test_idx'].astype(np.int)
    meta_data = read_meta_data(info_file)

    feats = np.load(f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/feats-{poe}.npy')
    labels = data['labels']

    print(feats.shape)
    # assert False

    if poe == 'femval':
        model = 'xx-conf-100-11000'
    elif poe == 'trunk':
        model = 'xx-coral-100-3100'
    elif poe == 'hip':
        model = 'xx-coral-100-10003'
    elif poe == 'KMFP':
        model = 'xx-conf-3025'
    else:
        assert False

    path = os.path.join(args.root, model)
    x_test = feats[test_idx, :]
    y_test = labels[test_idx, 0]

    all_confusions = np.zeros((num_folds,3,3))
    all_accs = []
    all_f1 = []

    for fold in range(1, num_folds + 1):
        train_idx = np.load(os.path.join(path, f'idx_{fold}.npz'))['train_idx']
        val_idx = np.load(os.path.join(path, f'idx_{fold}.npz'))['val_idx']
        x_train = feats[train_idx, :]
        y_train = labels[train_idx, 0]
        x_test = feats[val_idx, :]
        y_test = labels[val_idx, 0]

        x_test = np.delete(feats, train_idx, axis=0)
        y_test = np.delete(labels, train_idx, axis=0)
        # x_test = np.delete(feats, np.unique(np.append(train_idx, val_idx)), axis=0)
        # y_test = np.delete(labels, np.unique(np.append(train_idx, val_idx)), axis=0)

        # non_unique = 0
        # for i in test_idx:
        #     if i in train_idx:
        #         print(f'{i} in both!!!')
        #         non_unique += 1
        #
        # print(non_unique)

        sv_machine = SVC()
        sv_machine.fit(x_train, y_train)
        lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
        transform = lda.fit_transform(x_train,y_train)

        # preds = lda.transform(x_test)
        # for i in range(3):
        #     idx = np.where(y_test==i)[0]
        #     label = f'class {i}'
        #     plt.scatter(preds[idx,0], preds[idx,1], label=label)

        # plt.show()

        preds = lda.predict(x_test)

        # preds = sv_machine.predict(x_test)
        cnf = metrics.confusion_matrix(y_test, preds)
        acc = metrics.accuracy_score(y_test, preds)
        f1 = metrics.f1_score(y_test, preds, labels=[0, 1, 2], average='macro')

        # all_confusions[fold-1,...] = cnf
        all_accs.append(acc)
        all_f1.append(f1)
        print(cnf)
        print(acc)
        print(f1)


    print(f'acc: {np.mean(all_accs)} +- {np.std(all_accs)}')
    print(f'f1: {np.mean(all_f1)} +- {np.std(all_f1)}')
    # plot_confusion_matrix_mean(all_confusions, [0,1,2], title='Baseline repetitions')

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    main(args)
