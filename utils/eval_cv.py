import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import coral_ordinal as coral

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
    print(x.shape)
    print(y.shape)
    # ind_val = np.load(os.path.join(args.root, 'indices.npz'))
    ind_t = np.load(idx_path)
    # test_idx = np.append(ind_val['val_idx'], ind_t['test_idx'].astype(np.int))
    # # test_idx = ind_t['test_idx'].astype(np.int)
    # # test_idx = ind_val['val_idx'].astype(np.int)
    # x_test = x[test_idx, ...]
    # y_test = y[test_idx]
    # print(test_idx)

    for fold in range(1,6):
        model_path = os.path.join(args.root, f'model_fold_{fold}.hdf5')
        idx = np.load(os.path.join(args.root, f'idx_{fold}.npz'))
        x_val = x[idx['val_idx']]
        y_val = y[idx['val_idx']]

        print(x_val.shape)
        print(y_val.shape)
        assert False
        # x_val = x[ind_t['test_idx']]
        # y_val = y[ind_t['test_idx']]
        model = keras.models.load_model(model_path, custom_objects={'CoralOrdinal': coral.CoralOrdinal, 'OrdinalCrossEntropy': coral.OrdinalCrossEntropy, 'MeanAbsoluteErrorLabels': coral.MeanAbsoluteErrorLabels})
        result = model.predict(x_val)

        # print(result)
        probs = coral.ordinal_softmax(result).numpy()
        # print(probs)
        # print(np.append(probs, y_val, axis=1))

        incorr = 0
        corr = 0

        for row in range(probs.shape[0]):
            pred = probs[row,...]
            i = np.argmax(pred)
            if pred[i] > 0.8:
                pred[i] = 1
                pred[~i] = 0
                new_pred = [1 * (j == i) for j in range(probs.shape[-1])]
                probs[row,...] = new_pred
            # elif np.min(pred) < 0.1:
            #     i = np.argmin(pred)
            #     pred[i] = 0
            #     result[row,...] = pred
                # i = np.where(pred < 0.3)[0]
                # pred
                # new_pred = [1 * (j == i) for j in range(3)]
                # pred[i] = 0
                # result[row,...] = pred

            zeros = np.where(probs[row, ...] == 0)
            if len(zeros) > 0:
                i = zeros[0]
                incorr = incorr + 1 if y_val[row] in i else incorr
                # if y_val[row] in i:
                #     incorr += 1


                if np.sum(probs[row, ...]) == 1:
                    corr = corr + 1 if y_val[row] == np.argmax(probs[row, ...]) else corr


        print(np.append(probs, y_val, axis=1))
        print(f'incorrect: {incorr} of {row + 1}, {incorr / (row + 1)}')
        print(f'correct: {corr} of {row + 1}, {corr / (row + 1)}')

        y_pred = np.argmax(probs, axis=1)
        cnf_matrix = confusion_matrix(y_val, y_pred)
        plot_confusion_matrix(cnf_matrix, [0,1,2], title='Each repetition')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
