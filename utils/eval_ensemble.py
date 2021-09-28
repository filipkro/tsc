import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from argparse import ArgumentParser, ArgumentTypeError
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy

uncert_data = ['femval', 'hip', 'KMFP', 'trunk']


def get_same_subject(info_file, idx):
    '''extract repetitions made by the same subject,
    based on information in info_file'''
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
    '''get POE evaluated in dataset'''
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
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=15,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    '''fontsizes and subplot adjustments was suitable for me and my use'''
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.subplots_adjust(bottom=0.124, top=0.938)
    # plt.show()

    if savename != '':
        plt.savefig(savename)
        plt.close()


def main(args):
    '''Evaluates the ensembles for classification, plots confusion matrices'''
    idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    if 'ensemble-test' in args.root:
        lit = args.root.split('/')[-1]
    else:
        lit = os.path.basename(args.root).split('_')[0]

    # if any(char.isdigit() for char in lit):
    num_folds = 5
    if lit[-2].isdigit():
        assert lit[-1].isdigit()
        num_folds = int(lit[-2:])
        lit = lit[:-2]
    elif lit[-1].isdigit():
        print('lol')
        assert False

    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    dp100 = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '_len100' + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    dataset = np.load(dp)
    dataset100 = np.load(dp100)
    poe = get_POE_field(info_file)
    if poe in uncert_data:
        uncert_file = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/uncertainties-' + poe + '.npy'
        uncertainties = np.load(uncert_file)
        uncert = True

    x = dataset['mts']
    x100 = dataset100['mts']
    y = dataset['labels']
    ind_t = np.load(idx_path)
    test_idx = ind_t['test_idx'].astype(np.int)
    cnf_all_folds = np.zeros((3, 3))
    cnf_all_folds2 = np.zeros((3, 3))
    cnf_all_folds3 = np.zeros((3, 3))

    '''model choice based on dataset name'''
    if lit == 'Olga-Tokarczuk':
        models = ['coral-100-7000', 'xx-coral-100-7000', 'xx-conf-100-7000',
                  'xx-conf-100-11000', 'xx-conf-9000']
        weights = np.array([[1/3, 1.25/3, 1/3], [1/3, 1.25/3, 1/3],
                            [1/3, 0, 0], [0, 0, 1/3], [0, 1.25/3, 0]])
    elif lit == 'Nadine-Gordimer':
        models = ['coral-100-3100', 'coral-3100', 'xx-conf-100-3100',
                  'xx-coral-100-3100', 'conf-100-12000']
        weights = np.array([[1/3, 1.15/3, 1/3], [1/3, 1.15/3, 1/3],
                            [1/3, 0, 0], [0, 0, 1/3], [0, 1.15/3, 0]])
    elif lit == 'Albert-Camus':
        models = ['x-incep-600', 'conf-x-800', 'coral-x-500', 'coral-x-501',
                  'conf-x-707', 'conf-x-708']
        weights = np.array([[0.25, 0, 0], [0.35, 0, 0], [0.2, 0.5, 0],
                            [0.2, 0.5, 0], [0, 0, 0.5], [0, 0, 0.5]])
    elif lit == 'Sigrid-Undset':
        models = ['coral-100-13000', 'coral-13000', 'conf-100-10000',
                  'conf-10001', 'xx-coral-100-10003']
        weights = np.array([[1/4, 1.05/4, 1.5/4], [1/4, 1.05/4, 1.5/4],
                            [1/2, 0, 0], [0, 1.05/2, 0], [0, 0, 1.5/2]])
    elif lit == 'Mikhail-Sholokhov':
        models = ['inception-3010', 'xx-inception-3010', 'xx-conf-3010',
                  'conf-100-13000', 'xx-conf-3025']
        weights = np.array([[1/3, 1.25*1/3, 1.25/3], [1/3, 1.25*1/3, 1.25/3],
                            [1/3, 0, 0], [0, 1.25*1/3, 0], [0, 0, 1.25/3]])
    elif lit == 'Isaac-Bashevis-Singer':
        models = ['coral-100-11', 'coral-100-10', 'xx-conf-100-11', 'conf-15',
                  'xx-coral-100-10']
        weights = np.array([[1/3, 1.15/3, 1/3], [1/3, 1.15/3, 1/3],
                            [1/3, 0, 0], [0, 1.15/3, 0],[0, 0, 1/3]])

        # models = ['coral-100-11']
        # weights = np.array([[1,1,1]])

    ensembles = [os.path.join(args.root, i) for i in models] \
        if 'ensembles' in args.root else [args.root + str(i) for i in models]

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

    f1 = []
    acc = []
    rep_acc = []
    rep_f1 = []

    '''arrays to save predicted probabilities'''
    prob_mean_all = -1 * np.ones((num_folds, 12, 3, 3))
    prob_mean_pred = -1 * np.ones((num_folds, 12, 3, 3))
    prob_reps = -1 * np.ones((num_folds, 58, 3, 3))
    prob_reps_pred = -1 * np.ones((num_folds, 58, 3, 3))
    probs_one = -1 * np.ones((num_folds, 58))

    max_nbrs = 0

    rep_list = []
    rep_labels = []

    if args.confusion:
        path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_confusion.csv'
        confusion_file = open(path, 'w')

    for fold in range(1, num_folds + 1):
        paths = [os.path.join(
            root, f'model_fold_{fold}.hdf5') for root in ensembles]
        idx = np.load(os.path.join(ensembles[0], f'idx_{fold}.npz'))

        y_val = y[idx['val_idx']]
        max_nbrs = np.max((len(y_val), max_nbrs))

        accuracies = np.zeros(len(models))
        cnfs = np.zeros((len(models), 3, 3))
        all_probs = np.zeros((len(models), x.shape[0], 3))
        correct_individual = np.zeros(len(models))
        correct_ensemble = 0

        subject_indices = []
        individual_acc = np.zeros(len(models))

        '''sanity check to make sure no repetition from training data is
        evaluated'''
        non_unique = 0
        for i in idx['val_idx']:
            if i in idx['train_idx']:
                print(f'{i} in both!!!')
                non_unique += 1

        print(f'Number of reps in both train and val data: {non_unique}')

        for model_i, model_path in enumerate(paths):
            x_val = x100[idx['val_idx']
                         ] if '-100-' in model_path else x[idx['val_idx']]

            input = x100 if '-100-' in model_path else x
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
            # probs = probs * np.array([0.8, 1.1, 1.1])
            # probs = probs * weights[model_i, ...] #/ np.sum(weights[model_i, ...])
            # if model_i == 1:
            #     probs = probs * np.array([0.0,0,0.5])
            # if model_i == 2:
            #     probs = probs * np.array([0,0,0.0])
            for row in range(probs.shape[0]):
                pred = probs[row, ...]
                i = np.argmax(pred)
                if pred[i] > 1:  # 0.8:
                    pred[i] = 1
                    pred[~i] = 0
                    new_pred = [1 * (j == i) for j in range(3)]
                    probs[row, ...] = new_pred
            # / np.sum(weights[model_i, ...])
            probs = probs * weights[model_i, ...]

            # print(f'model: {model_i}')
            # print('probs')
            # print(probs)
            all_probs[model_i, ...] = probs

            subject_indices = []
            total = 0
            correct = 0


            for i in idx['val_idx']:
                # print(int(y[i]))

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

        ensemble_probs = np.sum(all_probs, axis=0)

        print(ensemble_probs.shape)

        hist_index = 0
        for i in idx['val_idx']:
            prob_reps[fold-1, hist_index, int(y[i]), :] = ensemble_probs[i, ...]
            predicted_class = np.argmax(ensemble_probs[i, ...])
            probs_one[fold-1,hist_index] = ensemble_probs[i,1]
            prob_reps_pred[fold-1, hist_index, int(y[i]), predicted_class] = ensemble_probs[i, predicted_class]
            hist_index += 1

        rep_preds = np.argmax(np.sum(all_probs[:, idx['val_idx'], ...],
                                     axis=0), axis=1)
        rep_f1.append(f1_score(y_val, rep_preds,
                               labels=[0, 1, 2], average='macro'))
        rep_acc.append(accuracy_score(y_val, rep_preds))

        rep_list = np.append(rep_list, rep_preds, axis=0)
        rep_labels = np.append(rep_labels, y_val[:,0], axis=0)

        hist_index = 0
        for i in idx['val_idx']:
            if i not in subject_indices:
                subject_indices, global_ind = get_same_subject(info_file, i)

                y_subj = y[subject_indices]
                pred_subj = ensemble_probs[subject_indices, ...]
                summed = np.mean(pred_subj, axis=0)
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
                    ''' test
                    conf_samples = max_like > 0.35
                    summed = np.sum(pred_subj[conf_samples,:], axis=0)/np.sum(conf_samples)
                    '''

                    mean_prob = np.mean(pred_subj, axis=0)

                    corr_combined = int(np.ceil(np.median(y_subj)))
                    pred_combined = int(np.ceil(np.median(pred)))
                    # pred_combined = int(np.argmax(np.mean(pred_subj[conf_samples, ...], axis=0)))
                    pred_combined = int(np.argmax(np.mean(pred_subj, axis=0)))
                    prob_mean_all[fold-1, hist_index, corr_combined, :] = summed
                    if np.max(summed) > 0.0:
                        preds.append(int(np.round(pred_combined)))
                        truth.append(int(np.round(corr_combined)))

                        pred_class = int(np.round(pred_combined))
                        prob_mean_pred[fold-1, hist_index, corr_combined, pred_class] = summed[pred_class]

                    hist_index += 1
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

                total += 1

        ensemble_acc = accuracy_score(truth, preds)
        fold_ens += ensemble_acc
        fold_individual = fold_individual + individual_acc

        f1.append(f1_score(truth, preds, labels=[0, 1, 2], average='macro'))
        acc.append(accuracy_score(truth, preds))

        cnf = confusion_matrix(truth2, preds2, labels=[0, 1, 2])
        cnf_all_folds2 = cnf_all_folds2 + cnf
        cnf = confusion_matrix(truth3, preds3, labels=[0, 1, 2])
        cnf_all_folds3 = cnf_all_folds3 + cnf
        cnf = confusion_matrix(truth, preds, labels=[0, 1, 2])
        cnf_all_folds = cnf_all_folds + cnf

        if args.confusion:
            confusion_file.write(f'Confusion for fold {fold},,\n')
            cnf_str = str(cnf).replace(' [', '')
            cnf_str = cnf_str.replace('[', '')
            cnf_str = cnf_str.replace(']','')
            cnf_str = cnf_str.replace(' ',',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n,,\n')

        print(f'individual accs: {individual_acc}')
        print(f'ensemble acc: {ensemble_acc}')
        print(f'ensemble confusion:\n{cnf}')
        print(f'f1 score: {f1[-1]}')

        # assert fold != 2

    print('----------------------')
    print(f'correctly ignored: {correctly_ignored}')
    print(f'incorrectly ignored: {incorrectly_ignored}')
    print(f'correctly kept: {correctly_kept}')
    print(f'incorrectly kept: {incorrectly_kept}')

    print('---------------------')
    print(f'individual accs: {fold_individual/num_folds}')
    print(f'ensemble acvc: {fold_ens/num_folds}')
    print(f'ensemble confusion:\n{cnf_all_folds}')
    print(f'f1: {f1}')
    print(f'f1: {np.mean(f1)} +- {np.std(f1)}')
    print(f'acc: {acc}')
    print(f'acc: {np.mean(acc)} +- {np.std(acc)}')

    print(f'rep f1: {rep_f1}')
    print(f'rep f1: {np.mean(rep_f1)} +- {np.std(rep_f1)}')
    print(f'rep acc: {rep_acc}')
    print(f'rep acc: {np.mean(rep_acc)} +- {np.std(rep_acc)}')

    print('-----------------------')
    print(incorr_info)

    if args.confusion:
        confusion_file.write(f'Sum of confusions all folds,,\n')
        cnf_str = str(cnf_all_folds.astype(int)).replace(' [', '')
        cnf_str = str(cnf_str).replace('[ ', '')
        cnf_str = str(cnf_str).replace('[', '')
        cnf_str = str(cnf_str).replace(']', '')
        cnf_str = cnf_str.replace(' ',',')
        confusion_file.write(cnf_str)
        confusion_file.close()

    plot_confusion_matrix(cnf_all_folds, [0, 1, 2])
    plot_confusion_matrix(cnf_all_folds2, [0, 1, 2])
    plot_confusion_matrix(cnf_all_folds3, [0, 1, 2])

    # print(prob_mean_all)
    # print(prob_reps)
    # print(prob_reps_pred)
    #
    hist_save_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_prob_matrices.npz'
    np.savez(hist_save_path, reps=prob_reps, comb=prob_mean_all, reps_max=prob_reps_pred, comb_max=prob_mean_pred, probs=probs_one)

    print(f'nbr of 3s: {nbr_of_3}')
    # plt.figure()
    # plt.hist(acc, bins=4)
    # plt.figure()
    # plt.hist(rep_acc, bins=4)
    # plt.figure()
    # plt.hist(probs2, bins=10)
    # plt.figure()
    # plt.hist(probs0w1)
    # plt.figure()
    # plt.hist(probs1w0)
    # print(f'repetition mean: {np.mean(rep_list) +- np.std(rep_list)}')
    plot_confusion_matrix(confusion_matrix(rep_labels, rep_list),[0,1,2])

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
    args = parser.parse_args()
    main(args)
