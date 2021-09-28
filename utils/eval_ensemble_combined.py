import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy

uncert_data = ['femval', 'hip', 'KMFP', 'trunk']


def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx, 1]
    in_cohort_nbr = data[idx, 3]
    indices = np.where(data[:, 1] == subj)[0]
    idx_same_leg = np.where(data[indices, 3] == in_cohort_nbr)
    # print(indices[idx_same_leg])
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


def main(args):
    idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    if 'ensemble-test' in args.root:
        lit = args.root.split('/')[-1]
    else:
        lit = os.path.basename(args.root).split('_')[0]
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    dataset = np.load(dp)
    poe = get_POE_field(info_file)
    if poe in uncert_data:
        uncert_file = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/uncertainties-' + poe + '.npy'
        uncertainties = np.load(uncert_file)
        uncert = True
    # print(os.path.basename(args.root).split('_')[0])
    # lit = os.path.basename(args.root).split('_')[0]
    # dp = os.path.join(args.data, 'data_') + 'Herta-Moller.npz'
    # dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']
    # ind_val = np.load(os.path.join(args.root, 'indices.npz'))
    ind_t = np.load(idx_path)
    # test_idx = np.append(ind_val['val_idx'], ind_t['test_idx'].astype(np.int))
    test_idx = ind_t['test_idx'].astype(np.int)
    # # test_idx = ind_val['val_idx'].astype(np.int)
    x_test = x[test_idx, ...]
    y_test = y[test_idx]
    # print(test_idx)
    cnf_all_folds = np.zeros((3, 3))
    cnf_all_folds2 = np.zeros((3, 3))
    cnf_all_folds3 = np.zeros((3, 3))
    # [11,13,14,15]#[20, 21, 22,23,24]#[21,23,24]#[30,31,32,33]#[140, 141]
    models = [300, 303, 304]

    if lit == 'Olga-Tokarczuk':
        models = ['conf-19', 'conf-20', 'coral-300', 'coral-303', 'coral-304']
        weights = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.8 / 3, 1 / 3, 1 / 3],
                            [0.8 / 3, 1 / 3, 1 / 3], [0.8 / 3, 1 / 3, 1 / 3]])
        models = ['conf-18', 'conf-20', 'coral-x-300',
                  'coral-x-303', 'coral-x-304']
        # models = ['conf-0', 'conf-3', 'coral-0', 'coral-1', 'coral-2']
        weights = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.8 / 3, 1 / 3, 1 / 3],
                            [0.8 / 3, 1 / 3, 1 / 3], [0.8 / 3, 1 / 3, 1 / 3]])

        models = ['conf-18', 'conf-20', 'coral-x-300',
                  'coral-x-303', 'coral-x-304', 'coral-605']
        weights = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.8 / 3, 1 / 3, 0.8 / 3],
                            [0.8 / 3, 1 / 3, 0.8 / 3], [0.8 / 3, 1 / 3, 0.8 / 3],
                            [0, 0, 0.2]])

        models = ['conf-20', 'coral-x-300',
                  'coral-x-303', 'coral-x-304', 'coral-605']
        weights = np.array([[0.2, 0, 0], [0.8 / 3, 1 / 3, 0.8 / 3],
                            [0.8 / 3, 1 / 3, 0.8 / 3], [0.8 / 3, 1 / 3, 0.8 / 3],
                            [0, 0, 0.2]])

        models = ['conf-20', 'coral-x-303',
                  'coral-x-304', 'coral-605', 'conf-x-707']
        weights = np.array([[0.2, 0, 0], [0.8 / 2, 0.9 / 2, 0.8 / 2],
                            [0.8 / 2, 0.9 / 2, 0.8 / 2], [0, 0, 0.2], [0, 0.1, 0]])
        # weights = np.array([[0.2, 0, 0], [0.8 / 2, 0.8 / 2, 0.8 / 2],
        #                     [0.8 / 2, 0.8 / 2, 0.8 / 2], [0, 0, 0.2], [0, 0.2, 0]])
    elif lit == 'Nadine-Gordimer':
        models = ['x-incep-600', 'coral-511', 'coral-702', 'coral-x-501',
                 'conf-x-700']
        weights = np.array([[0, 0, 0.8/3], [0.3, 0.35, 0.8/3],[0.3, 0.35, 0.8/3],
                            [0, 0.4, 0.2], [0.4, 0, 0]])
    elif lit == 'Albert-Camus':
        models = ['x-incep-600', 'conf-x-800', 'coral-x-500', 'coral-x-501', 'conf-x-707',
                 'conf-x-708']
        weights = np.array([[0.25, 0, 0], [0.35, 0, 0], [0.2, 0.5, 0],[0.2, 0.5, 0],
                            [0, 0, 0.5], [0, 0, 0.5]])



    # factor = 2/9
    # weights = np.array([[1, 0, 0], [1, 0, 0], [1-2*factor,1 +2*factor,1+2*factor],
    #                     [1-2*factor,1 +2*factor,1+2*factor], [1-2*factor,1 +2*factor,1+2*factor]])
    # models = ['conf-19', 'coral-300', 'coral-303', 'coral-304']
    # weights = np.array([[0.25, 0, 0], [0.75 / 3, 1 / 3, 1 / 3],
    #                     [0.75 / 3, 1 / 3, 1 / 3], [0.75 / 3, 1 / 3, 1 / 3]])
    # [122, 137, 130]#, 131, 132]#, 138]#, 139, 140, 141]
    ensembles = [os.path.join(args.root, i) for i in models] if 'ensemble-test' in args.root else [
        args.root + str(i) for i in models]

    fold_ens = 0
    fold_individual = np.zeros(len(models))

    nbr_of_3 = 0
    correctly_ignored = 0
    incorrectly_ignored = 0

    correctly_kept = 0
    incorrectly_kept = 0

    incorr_info = []

    probs0 = []
    probs1 = []
    probs2 = []
    probs0w1 = []
    probs1w0 = []

    for fold in range(1, 6):
        paths = [os.path.join(
            root, f'model_fold_{fold}.hdf5') for root in ensembles]
        idx = np.load(os.path.join(ensembles[0], f'idx_{fold}.npz'))
        x_val = x[idx['val_idx']]
        y_val = y[idx['val_idx']]

        accuracies = np.zeros(len(models))
        cnfs = np.zeros((len(models), 3, 3))
        all_probs = np.zeros((len(models), x.shape[0], 3))
        correct_individual = np.zeros(len(models))
        correct_ensemble = 0

        subject_indices = []
        individual_acc = np.zeros(len(models))

        for model_i, model_path in enumerate(paths):
            model = keras.models.load_model(model_path, custom_objects={
                                            'CoralOrdinal': coral.CoralOrdinal,
                                            'OrdinalCrossEntropy':
                                            coral.OrdinalCrossEntropy,
                                            'MeanAbsoluteErrorLabels':
                                            coral.MeanAbsoluteErrorLabels,
                                            'ConfusionCrossEntropy':
                                            ConfusionCrossEntropy})
            # x = x_test
            result = model.predict(x)
            probs = coral.ordinal_softmax(
                result).numpy() if 'coral' in model_path else result
            # probs = probs * np.array([0.8, 1.1, 1.1])
            # probs = probs * weights[model_i, ...] #/ np.sum(weights[model_i, ...])
            # if model_i == 1:
            #     probs = probs * np.array([0.0,0,0.5])
            # if model_i == 2:
            #     probs = probs * np.array([0,0,0.0])
            for row in range(probs.shape[0]):
                pred = probs[row, ...]
                i = np.argmax(pred)
                if pred[i] > 1:#0.8:
                    pred[i] = 1
                    pred[~i] = 0
                    new_pred = [1 * (j == i) for j in range(3)]
                    probs[row, ...] = new_pred
            # / np.sum(weights[model_i, ...])
            probs = probs * weights[model_i, ...]
            all_probs[model_i, ...] = probs

            subject_indices = []
            total = 0
            correct = 0

            for i in idx['val_idx']:
                if i not in subject_indices:
                    subject_indices, _ = get_same_subject(info_file, i)

                    y_subj = y[subject_indices]
                    pred_subj = probs[subject_indices, ...]

                    correct += 1 * (int(np.median(y_subj)) ==
                                    np.argmax(np.mean(pred_subj, axis=0)))

                    total += 1
            individual_acc[model_i] = correct / total

        subject_indices = []
        total = 0
        correct = 0
        preds = []
        truth = []

        preds2 = []
        truth2 = []
        preds3 = []
        truth3 = []

        for i in idx['val_idx']:
            if i not in subject_indices:
                subject_indices, global_ind = get_same_subject(info_file, i)

                y_subj = y[subject_indices]
                pred_subj = np.sum(
                    all_probs[:, subject_indices, ...], axis=0)  # / len(models)
                # print(np.sum(pred_subj, axis=0))
                # if np.sum(pred_subj) > 0.8:
                summed = np.mean(pred_subj, axis=0)
                # print(summed)
                if uncert:
                    print('jennys uncertainty')
                    print(uncertainties[global_ind])
                    nbr_of_3 = nbr_of_3 + \
                        1 if uncertainties[global_ind] == 3 else nbr_of_3

                test_median = True

                if test_median:
                    max_like = np.max(pred_subj, axis=1)
                    pred = np.argmax(pred_subj, axis=1)

                    # for likelihoods, corr in zip(pred_subj, y_subj):
                    #     print(f'shapes: {likelihoods.shape}, {corr.shape}')
                    #     mean_probs = np.mean()
                    #     if corr == 0:
                    #         probs0.append(likelihoods[0])
                    #         probs1w0.append(likelihoods[1])
                    #     elif corr ==  1:
                    #         probs1.append(likelihoods[1])
                    #         probs0w1.append(likelihoods[0])
                    #     elif corr == 2:
                    #         probs2.append(likelihoods[2])

                    mean_prob = np.mean(pred_subj,axis=0)

                    print(f'correct: {y_subj}')
                    print(f'preds: {pred}')
                    print(f'probs: {max_like}')

                    corr_combined = int(np.ceil(np.median(y_subj)))
                    pred_combined = int(np.ceil(np.median(pred)))
                    pred_combined = int(np.argmax(np.mean(pred_subj, axis=0)))
                    print(f'corr combined: {corr_combined}')
                    print(f'pred combined: {pred_combined}')
                    print(f'men prob: {np.median(max_like)}')

                    preds.append(int(np.round(pred_combined)))
                    truth.append(int(np.round(corr_combined)))

                    # if np.mean(max_like) > 0:
                    #     if uncertainties[global_ind] == 1:
                    #         preds.append(pred_combined)
                    #         truth.append(corr_combined)
                    #         # probs1.append(np.mean(max_like))
                    #     elif uncertainties[global_ind] == 2:
                    #         preds2.append(pred_combined)
                    #         truth2.append(corr_combined)
                    #         # probs2.append(np.mean(max_like))
                    #     elif uncertainties[global_ind] == 3:
                    #         preds3.append(pred_combined)
                    #         truth3.append(corr_combined)
                            # probs3.append(np.mean(max_like))
                    #
                    # if np.mean(max_like) > 0.5:
                    #     preds.append(int(np.round(pred_combined)))
                    #     truth.append(int(np.round(corr_combined)))
                    # elif pred_combined == 1 and np.mean(max_like) > 0.45:
                    #     preds.append(int(np.round(pred_combined)))
                    #     truth.append(int(np.round(corr_combined)))

                    # if np.mean(max_like) > 0.6:
                    #     preds.append(int(np.round(pred_combined)))
                    #     truth.append(int(np.round(corr_combined)))
                    # elif pred_combined == 1 and np.mean(max_like) > 0.5:
                    #     preds.append(int(np.round(pred_combined)))
                    #     truth.append(int(np.round(corr_combined)))

                    # for a in range(pred_subj.shape[0]):
                    #     if np.sum(pred_subj[a, :]) < 0.95:
                    #         correctly_ignored += 1*(int(np.argmax(pred_subj[a,:])) != y_subj[a])
                    #         incorrectly_ignored += 1*(int(np.argmax(pred_subj[a,:])) == y_subj[a])
                    #     else:
                    #         correctly_kept += 1*(int(np.argmax(pred_subj[a,:])) == y_subj[a])
                    #         incorrectly_kept += 1*(int(np.argmax(pred_subj[a,:])) != y_subj[a])
                    # if np.max(summed) > 0.45:
                    # if np.max(summed) > 0.4:
                    # if np.sum(summed) > 0.98 and True:
                else:
                    # np.max(summed) > 0.45 and np.max(summed) < 0.6:
                    if np.max(summed) > 0.4 or False:
                        pred = np.argmax(np.sum(pred_subj, axis=0))
                        correct += 1 * (int(np.round(np.median(y_subj))) ==
                                        pred)
                        # preds.append(pred)
                        # truth.append(int(np.round(np.median(y_subj))))
                        max_like = np.max(pred_subj, axis=1)
                        pred = np.argmax(pred_subj, axis=1)

                        corr_combined = np.median(y_subj)
                        pred_combined = np.median(pred)

                        preds.append(int(np.round(pred_combined)))
                        truth.append(int(np.round(corr_combined)))

                        # if uncertainties[global_ind] == 1:
                        #     preds.append(int(np.round(pred_combined)))
                        #     truth.append(int(np.round(corr_combined)))
                        # elif uncertainties[global_ind] == 2:
                        #     preds2.append(int(np.round(pred_combined)))
                        #     truth2.append(int(np.round(corr_combined)))
                        # elif uncertainties[global_ind] == 3:
                        #     preds3.append(int(np.round(pred_combined)))
                        #     truth3.append(int(np.round(corr_combined)))

                    # if pred != int(np.round(np.median(y_subj))):
                    #     info = {'indxs': subject_indices, 'probs': all_probs[:, subject_indices, ...], 'true': y_subj, 'preds': pred_subj,  'fold': fold}
                    #
                    #     incorr_info.append(info)

                    # if not (int(np.round(np.median(y_subj))) ==
                    #                 np.argmax(summed)) and True:
                    # print(f'correct? {(int(np.round(np.median(y_subj))) == np.argmax(summed))}')
                    # print(subject_indices)
                    # print(pred_subj)
                    # print(summed)
                    # print(y_subj)
                    # for ii in range(len(y_subj)):
                    #     if not (y_subj[ii] == np.argmax(pred_subj[ii, :])):
                    #         print(f'incorrect: {subject_indices[ii]}')
                    #         print(all_probs[:, subject_indices[ii], ...])
                    #         print(pred_subj[ii, :])
                    #         print(y_subj[ii])
                    # print('')
                # correct += 1 * (int(np.median(y_subj)) ==
                #                 np.argmax(np.mean(pred_subj, axis=0)))
                # preds.append(np.argmax(np.mean(pred_subj, axis=0)))
                # truth.append(int(np.median(y_subj)))
                total += 1

        ensemble_acc = accuracy_score(truth, preds)
        fold_ens += ensemble_acc
        fold_individual = fold_individual + individual_acc

        cnf = confusion_matrix(truth2, preds2, labels=[0, 1, 2])
        cnf_all_folds2 = cnf_all_folds2 + cnf
        cnf = confusion_matrix(truth3, preds3, labels=[0, 1, 2])
        cnf_all_folds3 = cnf_all_folds3 + cnf
        cnf = confusion_matrix(truth, preds, labels=[0, 1, 2])
        cnf_all_folds = cnf_all_folds + cnf

        print(f'individual accs: {individual_acc}')
        print(f'ensemble acc: {ensemble_acc}')
        print(f'ensemble confusion:\n{cnf}')

        # assert fold != 2

    print('----------------------')
    print(f'correctly ignored: {correctly_ignored}')
    print(f'incorrectly ignored: {incorrectly_ignored}')
    print(f'correctly kept: {correctly_kept}')
    print(f'incorrectly kept: {incorrectly_kept}')

    print('---------------------')
    print(f'individual accs: {fold_individual/5}')
    print(f'ensemble acvc: {fold_ens/5}')
    print(f'ensemble confusion:\n{cnf_all_folds}')

    print('-----------------------')
    print(incorr_info)

    plot_confusion_matrix(cnf_all_folds, [0, 1, 2])
    plot_confusion_matrix(cnf_all_folds2, [0, 1, 2])
    plot_confusion_matrix(cnf_all_folds3, [0, 1, 2])

    print(f'nbr of 3s: {nbr_of_3}')
    plt.figure()
    plt.hist(probs0, bins=10)
    plt.figure()
    plt.hist(probs1,bins=10)
    plt.figure()
    plt.hist(probs2,bins=10)
    plt.figure()
    plt.hist(probs0w1)
    plt.figure()
    plt.hist(probs1w0)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--uncert', default='')
    args = parser.parse_args()
    main(args)
