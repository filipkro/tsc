import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy

def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx,1]
    in_cohort_nbr = data[idx,3]
    indices = np.where(data[:,1] == subj)[0]
    idx_same_leg = np.where(data[indices,3] == in_cohort_nbr)
    print(indices[idx_same_leg])
    return indices[idx_same_leg]

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
    plt.show()

    if savename != '':
        plt.savefig(savename)
        plt.close()


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
    # ind_val = np.load(os.path.join(args.root, 'indices.npz'))
    ind_t = np.load(idx_path)
    # test_idx = np.append(ind_val['val_idx'], ind_t['test_idx'].astype(np.int))
    # # test_idx = ind_t['test_idx'].astype(np.int)
    # # test_idx = ind_val['val_idx'].astype(np.int)
    # x_test = x[test_idx, ...]
    # y_test = y[test_idx]
    # print(test_idx)
    cnf_all_folds = np.zeros((3, 3))

    for fold in range(1,6):
        model_path = os.path.join(args.root, f'model_fold_{fold}.hdf5')
        idx = np.load(os.path.join(args.root, f'idx_{fold}.npz'))
        # x_val = x[idx['val_idx']]
        # y_val = y[idx['val_idx']]
        # # x_val = x[ind_t['test_idx']]
        # # y_val = y[ind_t['test_idx']]
        model = keras.models.load_model(model_path, custom_objects={'CoralOrdinal': coral.CoralOrdinal, 'OrdinalCrossEntropy': coral.OrdinalCrossEntropy, 'MeanAbsoluteErrorLabels': coral.MeanAbsoluteErrorLabels, 'ConfusionCrossEntropy': ConfusionCrossEntropy})
        # model = keras.models.load_model(model_path, custom_objects={'ConfusionCrossEntropy': ConfusionCrossEntropy})

        # assert False
        subject_indices = []
        correct = 0
        corr_mean = 0
        pred_combined = []
        y_combined = []
        for i in idx['val_idx']:
            if i not in subject_indices:
                subject_indices = get_same_subject(info_file, i)

                x_subj = x[subject_indices, ...]
                y_subj = y[subject_indices]
                result = model.predict(x_subj)
                if 'coral' in model_path:
                    result = coral.ordinal_softmax(result).numpy()
                # print('result for indices: {}'.format(subject_indices))

                if 398 in subject_indices:
                    print(result)
                    print(y_subj)
                    print(np.mean(result, axis=0))
                # print('likelihoods')
                # print(result)
                for row in range(result.shape[0]):
                    pred = result[row,...]
                    i = np.argmax(pred)
                    if pred[i] > 0.7:
                        pred[i] = 1
                        pred[~i] = 0
                        new_pred = [1 * (j == i) for j in range(3)]
                        result[row,...] = new_pred
                # print(result)
                # print('correct')
                # print(y_subj)
                # print('summed likelihoods')
                # print(np.sum(result, axis=0))
                # print('true label')
                # print(np.median(y_subj))
                if np.max(np.mean(result, axis=0)) > 0.5:
                    pred_combined.append(np.argmax(np.sum(result, axis=0)))
                    y_combined.append(int(np.median(y_subj)))
                    correct += 1*(int(np.median(y_subj)) == np.argmax(np.sum(result, axis=0)))
                    corr_mean += 1*(int(np.round(np.mean(y_subj))) == np.argmax(np.sum(result, axis=0)))
                # print('\n \n')

        # print(correct)
        # print(corr_mean)
        combined_cm = confusion_matrix(y_combined, pred_combined, labels=[0,1,2])
        # print(combined_cm)
        cnf_all_folds = cnf_all_folds + combined_cm
        plot_confusion_matrix(combined_cm, [0,1,2], title='combined score')

    plot_confusion_matrix(cnf_all_folds, [0,1,2], title='combined score, all folds')
    assert False



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
