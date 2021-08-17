import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
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
    # ind_val = np.load(os.path.join(args.root, 'indices.npz'))
    ind_t = np.load(idx_path)
    # test_idx = np.append(ind_val['val_idx'], ind_t['test_idx'].astype(np.int))
    # # test_idx = ind_t['test_idx'].astype(np.int)
    # # test_idx = ind_val['val_idx'].astype(np.int)
    # x_test = x[test_idx, ...]
    # y_test = y[test_idx]
    # print(test_idx)
    cnf_all_folds = np.zeros((3, 3))
    models = [300,303,304]#[20, 21, 22]#[11,12,13]#[122, 137] #[122, 137, 130]#, 138, 139, 140, 141]
    ensembles = [args.root + str(i) for i in models]

    fold_acc = 0
    fold_accuracies = np.zeros(len(models))

    for fold in range(1,6):
        paths = [os.path.join(root, f'model_fold_{fold}.hdf5') for root in ensembles]
        idx = np.load(os.path.join(ensembles[0], f'idx_{fold}.npz'))
        x_val = x[idx['val_idx']]
        y_val = y[idx['val_idx']]

        accuracies = np.zeros(len(models))
        cnfs = np.zeros((len(models),3,3))
        all_probs = np.zeros((len(models),x_val.shape[0], 3))

        for model_i, model_path in enumerate(paths):
            model = keras.models.load_model(model_path, custom_objects={'CoralOrdinal': coral.CoralOrdinal, 'OrdinalCrossEntropy': coral.OrdinalCrossEntropy, 'MeanAbsoluteErrorLabels': coral.MeanAbsoluteErrorLabels})

            result = model.predict(x_val)
            probs = coral.ordinal_softmax(result).numpy()
            probs = probs * np.array([0.8,1.1,1.1])
            # if model_i == 1:
            #     probs = probs * np.array([0.05,0.05,0.5])
            for row in range(probs.shape[0]):
                pred = probs[row,...]
                i = np.argmax(pred)
                if pred[i] > 1:
                    pred[i] = 1
                    pred[~i] = 0
                    new_pred = [1 * (j == i) for j in range(3)]
                    probs[row,...] = new_pred
            y_pred = np.argmax(probs, axis=1)
            cnf_matrix = confusion_matrix(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)
            # print(cnf_matrix)

            accuracies[model_i] = acc
            cnfs[model_i, ...] = cnf_matrix
            all_probs[model_i, ...] = probs

        ensemble = np.sum(all_probs, axis=0)/len(models)
        y_pred = np.argmax(ensemble, axis=1)
        cnf_all_folds = cnf_all_folds + confusion_matrix(y_val, y_pred, labels=[0,1,2])
        acc = accuracy_score(y_val, y_pred)

        # print(f'individual cnfs: {cnfs}')
        # print(f'ensemble: {cnf_matrix}')

        print(f'individual acc: {accuracies}')
        print(f'ensemble: {acc}')
        fold_acc += acc
        fold_accuracies = fold_accuracies + accuracies


    print('----------------------------')
    print(f'mean individual acc: {fold_accuracies/5}')
    print(f'mean ensemble acc: {fold_acc / 5}')
    print(f'ensemble confusion:\n{cnf_all_folds}')

    plot_confusion_matrix(cnf_all_folds, [0,1,2])
    #
    #
    #
    #
    #
    #
    #
    #
    #         subject_indices = []
    #         correct = 0
    #         total = 97
    #         pred_combined = []
    #         y_combined = []
    #
    #         for i in idx['val_idx']:
    #             if i not in subject_indices:
    #                 subject_indices = get_same_subject(info_file, i)
    #
    #                 x_subj = x[subject_indices, ...]
    #                 y_subj = y[subject_indices]
    #                 result = model.predict(x_subj)
    #                 if 'coral' in model_path:
    #                     result = coral.ordinal_softmax(result).numpy()
    #
    #                 pred_combined.append(np.argmax(np.sum(result, axis=0)))
    #                 y_combined.append(int(np.median(y_subj)))
    #                 correct += 1*(int(np.median(y_subj)) == np.argmax(np.sum(result, axis=0)))
    #
    #                 combined_cm = confusion_matrix(y_combined, pred_combined, labels=[0,1,2])
    #                 acc = accuracy_score(y_combined, pred_combined)
    #                 accuracies[model_i] = acc
    #                 cnfs[model_i, ...] = combined_cm
    #
    #
    # for fold in range(1,6):
    #     model_path = os.path.join(args.root, f'model_fold_{fold}.hdf5')
    #     idx = np.load(os.path.join(args.root, f'idx_{fold}.npz'))
    #     # x_val = x[idx['val_idx']]
    #     # y_val = y[idx['val_idx']]
    #     # # x_val = x[ind_t['test_idx']]
    #     # # y_val = y[ind_t['test_idx']]
    #     model = keras.models.load_model(model_path, custom_objects={'CoralOrdinal': coral.CoralOrdinal, 'OrdinalCrossEntropy': coral.OrdinalCrossEntropy, 'MeanAbsoluteErrorLabels': coral.MeanAbsoluteErrorLabels})
    #
    #
    #
    #     # assert False
    #     subject_indices = []
    #     correct = 0
    #
    #     corr_mean = 0
    #     pred_combined = []
    #     y_combined = []
    #     for i in idx['val_idx']:
    #         if i not in subject_indices:
    #             subject_indices = get_same_subject(info_file, i)
    #
    #             x_subj = x[subject_indices, ...]
    #             y_subj = y[subject_indices]
    #             result = model.predict(x_subj)
    #             if 'coral' in model_path:
    #                 result = coral.ordinal_softmax(result).numpy()
    #             print('result for indices: {}'.format(subject_indices))
    #             # print('likelihoods')
    #             # print(result)
    #             for row in range(result.shape[0]):
    #                 pred = result[row,...]
    #                 i = np.argmax(pred)
    #                 if pred[i] > 0.7:
    #                     pred[i] = 1
    #                     pred[~i] = 0
    #                     new_pred = [1 * (j == i) for j in range(3)]
    #                     result[row,...] = new_pred
    #             print(result)
    #             print('correct')
    #             print(y_subj)
    #             print('summed likelihoods')
    #             print(np.sum(result, axis=0))
    #             print('true label')
    #             print(np.median(y_subj))
    #             pred_combined.append(np.argmax(np.sum(result, axis=0)))
    #             y_combined.append(int(np.median(y_subj)))
    #             correct += 1*(int(np.median(y_subj)) == np.argmax(np.sum(result, axis=0)))
    #             corr_mean += 1*(int(np.round(np.mean(y_subj))) == np.argmax(np.sum(result, axis=0)))
    #             print('\n \n')
    #
    #     print(correct)
    #     print(corr_mean)
    #     combined_cm = confusion_matrix(y_combined, pred_combined, labels=[0,1,2])
    #     print(combined_cm)
    #     cnf_all_folds = cnf_all_folds + combined_cm
    #     plot_confusion_matrix(combined_cm, [0,1,2], title='combined score')
    #
    # plot_confusion_matrix(cnf_all_folds, [0,1,2], title='combined score, all folds')
    # assert False



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
