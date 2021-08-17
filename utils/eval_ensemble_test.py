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
    This function prints and plots the confusion matrix,
    along with the corresponding standard deviations.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.mean(all_matrices, axis=0)
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

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:.2f}\n+- {np.std(all_matrices[:,i,j]):.2f}',
                 horizontalalignment="center", fontsize=15,
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.subplots_adjust(bottom=0.124, top=0.938)

    if savename != '':
        plt.savefig(savename)
        plt.close()


def get_subj_idx(info_file, idx):
    ''' get subject index of repetitition'''
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx, 1]
    return subj


def main(args):
    idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx2.npz'
    if 'ensemble-test' in args.root:
        lit = args.root.split('/')[-1]
    else:
        lit = os.path.basename(args.root).split('_')[0]

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

    # load datasets, TODO: fix non static paths
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    dp100 = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '_len100' + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info-fix.txt'
    dataset = np.load(dp)
    dataset100 = np.load(dp100)
    poe = get_POE_field(info_file)
    if poe in uncert_data:
        uncert_file = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/uncertainties-' + poe + '.npy'
        uncertainties = np.load(uncert_file)
        # uncert = True

    idx_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-{poe}.npz'

    if poe == 'femval':
        uncertainties = femval_cert
    elif poe == 'trunk':
        uncertainties = trunk_cert
    elif poe == 'hip':
        uncertainties = hip_cert
    elif poe == 'KMFP':
        uncertainties = kmfp_cert
    else:
        assert False

    x = dataset['mts']
    x100 = dataset100['mts']
    y = dataset['labels']
    ind_t = np.load(idx_path)
    test_idx = ind_t['test_idx'].astype(np.int)

    y_test = y[test_idx]

    # empty confusion matrices
    cnf_all_folds = np.zeros((3, 3))
    cnf_all_folds_thresh = np.zeros((3, 3))
    cnf_all_folds3 = np.zeros((3, 3))
    all_confusions = np.zeros((num_folds,3,3))
    all_confusions_thresh = np.zeros((num_folds,3,3))
    all_confusions_ignored = np.zeros((num_folds,3,3))
    all_rep_cnf = np.zeros((num_folds,3,3))
    confusions_certain = np.zeros((num_folds,3,3))

    # ensemble choices
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

    ensembles = [os.path.join(args.root, i) for i in models] if 'ensembles' \
        in args.root else [args.root + str(i) for i in models]

    fold_ens = 0

    nbr_of_3 = 0

    incorr_info = []

    # results:
    f1 = []
    acc = []
    acc_thresh = []
    f1_thresh = []
    rep_acc = []
    rep_f1 = []
    acc_c1 = []
    acc_c2 = []
    acc_c3 = []
    f1_c1 = []
    f1_c2 = []
    f1_c3 = []
    precision = []
    precision_thresh = []
    precision_rep = []
    recall = []
    recall_thresh = []
    recall_rep = []
    recall_c1 = []
    recall_c2 = []
    recall_c3 = []
    precision_c1 = []
    precision_c2 = []
    precision_c3 = []

    prob_mean_all = -1 * np.ones((num_folds, 30, 3, 3))
    prob_mean_pred = -1 * np.ones((num_folds, 30, 3, 3))
    prob_reps = -1 * np.ones((num_folds, 130, 3, 3))
    prob_reps_pred = -1 * np.ones((num_folds, 130, 3, 3))

    max_nbrs = 0  # to check max size

    rep_list = []
    rep_labels = []
    individual_accs = np.zeros((num_folds, len(models)))
    individual_f1 = np.zeros((num_folds, len(models)))

    certs = np.empty((1, 2))
    corr_certs = np.empty((1, 2))

    if args.confusion:
        path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_confusion-test.csv'
        confusion_file = open(path, 'w')

    # evaluate all models in the ensembles from all folds
    for fold in range(1, num_folds + 1):
        # iterate over folds
        if fold == 10 and lit == 'Sigrid-Undset':
            # weights = np.array([[1/4,1.7/4,1.5/4],[1/4,1.7/4,1.5/4],[1/2,0,0], [0,1.7/2,0], [0,0,1.5/2]])
            # print('LOLOLOLOLOLOLOL')
            print(f'sigrid undset fold 10, weights: {weights}')

        paths = [os.path.join(root,
                              f'model_fold_{fold}.hdf5') for root in ensembles]

        # y_val = y_test

        # max_nbrs = np.max((len(y_val), max_nbrs))  # to check max size

        # all ensemble predictions
        all_probs = np.zeros((len(models), x.shape[0], 3))

        subject_indices = []
        individual_acc = np.zeros(len(models))

        corr_rep = []
        pred_rep = []

        for model_i, model_path in enumerate(paths):
            # iterate over models in ensemble
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

            for row in range(probs.shape[0]):
                pred = probs[row, ...]
                i = np.argmax(pred)
                # upper threshold to round to one-hot, 1 gives no rounding
                if pred[i] > 1:  # 0.8:
                    pred[i] = 1
                    pred[~i] = 0
                    new_pred = [1 * (j == i) for j in range(3)]
                    probs[row, ...] = new_pred

            # predicted probabilities,
            # weighted by the model weight in the ensemble:
            probs = probs * weights[model_i, ...]

            all_probs[model_i, ...] = probs

            all_subject_indices = []

            y_corr = []
            y_pred_comb = []
            for i in ind_t['test_idx']:
                # check the results for the individual models
                if i not in all_subject_indices:
                    # make sure each subject only evaluated once
                    subject_indices, _ = get_same_subject(info_file, i)
                    all_subject_indices.append(subject_indices)
                    y_subj = y[subject_indices]
                    pred_subj = probs[subject_indices, ...]

                    y_corr.append(int(np.median(y_subj)))
                    y_pred_comb.append(np.argmax(np.mean(pred_subj, axis=0)))

            # accuracy and f1 for individual models
            individual_accs[fold-1, model_i] = accuracy_score(y_corr,
                                                              y_pred_comb)
            individual_f1[fold-1, model_i] = f1_score(y_corr, y_pred_comb,
                                                      labels=[0, 1, 2],
                                                      average='macro')

        subject_indices = []
        total = 0
        preds = []
        truth = []

        # results
        preds_thresh = []
        truth_thresh = []
        preds_ignored = []
        truth_ignored = []
        preds3 = []
        truth3 = []
        p1 = []
        t1 = []
        p2 = []
        t2 = []
        p3 = []
        t3 = []

        ensemble_probs = np.sum(all_probs, axis=0)

        # threshold
        ensemble_probs = (ensemble_probs > 0.2) * ensemble_probs

        # if fold == 10:
        #     # debug
        #     print(ensemble_probs)

        for_hist = np.sum(all_probs, axis=0)
        hist_index = 0
        for i in ind_t['test_idx']:
            # check results for ensemble predictions for repetitions
            prob_reps[fold-1, hist_index, int(y[i]), :] = ensemble_probs[i, ...]
            predicted_class = np.argmax(ensemble_probs[i, ...])

            prob_reps_pred[fold-1, hist_index, int(y[i]), predicted_class] = \
                ensemble_probs[i, predicted_class]
            hist_index += 1

            corr_rep.append(y[i][0])
            pred_rep.append(np.argmax(ensemble_probs[i, ...]))

        # ensemble results for repetitions
        all_rep_cnf[fold-1, ...] = confusion_matrix(corr_rep, pred_rep,
                                                    labels=[0, 1, 2])
        rep_f1.append(f1_score(corr_rep, pred_rep,
                               labels=[0, 1, 2], average='macro'))
        rep_acc.append(accuracy_score(corr_rep, pred_rep))
        precision_rep.append(precision_score(corr_rep, pred_rep,
                               labels=[0, 1, 2], average='macro'))
        recall_rep.append(recall_score(corr_rep, pred_rep,
                               labels=[0, 1, 2], average='macro'))
        rep_list = np.append(rep_list, pred_rep, axis=0)
        rep_labels = np.append(rep_labels, corr_rep, axis=0)

        hist_index = 0
        all_subject_indices = []
        for i in ind_t['test_idx']:
            # check results for combined ensemble scores
            if i not in subject_indices:
                subject_indices, global_ind = get_same_subject(info_file, i)
                all_subject_indices.append(subject_indices)

                y_subj = y[subject_indices]
                pred_subj = ensemble_probs[subject_indices, ...]

                summed = np.mean(pred_subj, axis=0)

                pred = np.argmax(pred_subj, axis=1)
                corr_combined = int(np.ceil(np.median(y_subj)))

                # mean or median? -use mean
                pred_combined = int(np.ceil(np.median(pred)))
                pred_combined = int(np.argmax(np.mean(pred_subj, axis=0)))

                '''
                # DEBUGGING
                if corr_combined != pred_combined:
                    print(f'incorrect prediction of subject: {global_ind}')
                    print(f'class {corr_combined} classified as {pred_combined}')
                '''

                prob_mean_all[fold-1, hist_index,
                              corr_combined, :] = summed
                preds.append(int(np.round(pred_combined)))
                truth.append(int(np.round(corr_combined)))

                pred_class = int(np.round(pred_combined))
                prob_mean_pred[fold-1, hist_index, corr_combined,
                               pred_class] = summed[pred_class]


                # for some scatter plots, didn't really give anything...
                certs = np.append(certs,
                                  np.array([[uncertainties[str(global_ind)],
                                           np.max(summed)]]), axis=0)
                corr_certs = np.append(corr_certs,
                                       np.array([[uncertainties[str(global_ind)],
                                                int(int(np.round(pred_combined)) ==
                                                    int(np.round(corr_combined)))]]),
                                       axis=0)

                # threshold to ignore uncertain predictions
                thres = 0.35 if (lit == 'Sigrid-Undset' or
                                 lit == 'Mikhail-Sholokhov') else 0.4
                if np.max(summed) > thres:
                    # samples kept after threshold
                    preds_thresh.append(int(np.round(pred_combined)))
                    truth_thresh.append(int(np.round(corr_combined)))
                else:
                    # ignored by threshold
                    preds_ignored.append(int(np.round(pred_combined)))
                    truth_ignored.append(int(np.round(corr_combined)))

                '''
                # DEBBUGUNG 2-0 INCORRECT
                if (int(np.round(pred_combined)) == 0 and
                    int(np.round(corr_combined)) == 2):

                    print(f'INCORRRR :: 2-0, ind {global_ind}')
                    print(f'preds:: {pred_subj}')
                    print(f'corr:: {y_subj}')
                '''

                # resuklts for Jenny's certainty levels
                if uncertainties[str(global_ind)] == 1:
                    p1.append(int(np.round(pred_combined)))
                    t1.append(int(np.round(corr_combined)))
                elif uncertainties[str(global_ind)] == 2:
                    p2.append(int(np.round(pred_combined)))
                    t2.append(int(np.round(corr_combined)))
                elif uncertainties[str(global_ind)] == 3:
                    p3.append(int(np.round(pred_combined)))
                    t3.append(int(np.round(corr_combined)))
                    # prob_mean_some
                hist_index += 1

                total += 1

        # result metrics
        ensemble_acc = accuracy_score(truth, preds)

        labels = np.sort(np.unique(np.append(truth, preds)))
        f1.append(f1_score(truth, preds, labels=labels, average='macro'))
        acc.append(accuracy_score(truth, preds))
        precision.append(precision_score(truth, preds, labels=labels,
                                         average='macro'))
        recall.append(recall_score(truth, preds, labels=labels,
                                   average='macro'))

        labels = np.sort(np.unique(np.append(truth_thresh, preds_thresh)))
        f1_thresh.append(f1_score(truth_thresh, preds_thresh, labels=labels,
                                  average='macro'))
        acc_thresh.append(accuracy_score(truth_thresh, preds_thresh))
        precision_thresh.append(precision_score(truth_thresh, preds_thresh,
                                                labels=labels,
                                                average='macro'))
        recall_thresh.append(recall_score(truth_thresh, preds_thresh,
                                          labels=labels, average='macro'))

        # confusion matrices
        cnf = confusion_matrix(truth_thresh, preds_thresh, labels=[0, 1, 2])
        cnf_all_folds_thresh = cnf_all_folds_thresh + cnf
        all_confusions_thresh[fold-1, ...] = cnf
        cnf = confusion_matrix(truth_ignored, preds_ignored, labels=[0, 1, 2])
        all_confusions_ignored[fold-1, ...] = cnf
        cnf = confusion_matrix(truth3, preds3, labels=[0, 1, 2])
        cnf_all_folds3 = cnf_all_folds3 + cnf
        cnf = confusion_matrix(truth, preds, labels=[0, 1, 2])
        cnf_all_folds = cnf_all_folds + cnf
        all_confusions[fold-1, ...] = cnf

        # metrics for different certainty levels
        acc_c1.append(accuracy_score(t1, p1))
        labels = np.sort(np.unique(np.append(t1, p1)))
        f1_c1.append(f1_score(t1, p1, labels=labels, average='macro'))
        recall_c1.append(recall_score(t1, p1, labels=labels, average='macro'))
        precision_c1.append(precision_score(t1, p1, labels=labels,
                                            average='macro'))

        acc_c2.append(accuracy_score(t2, p2))
        labels = np.sort(np.unique(np.append(t1, p1)))
        f1_c2.append(f1_score(t2, p2, labels=labels, average='macro'))
        recall_c2.append(recall_score(t2, p2, labels=labels, average='macro'))
        precision_c2.append(precision_score(t2, p2, labels=labels,
                                            average='macro'))

        if len(t3) > 0:
            # certainty level 3 does not alsways exist
            acc_c3.append(accuracy_score(t3, p3))
            labels = np.sort(np.unique(np.append(t1, p1)))
            f1_c3.append(f1_score(t3, p3, labels=labels, average='macro'))
            recall_c3.append(recall_score(t3, p3, labels=labels,
                             average='macro'))
            precision_c3.append(precision_score(t3, p3, labels=labels,
                                average='macro'))

        if args.confusion and False:
            # write confusion matrices to file,
            # currently disabled to not overwrite
            confusion_file.write(f'Confusion for fold {fold},,\n')
            cnf_str = str(cnf).replace(' [', '')
            cnf_str = cnf_str.replace('[', '')
            cnf_str = cnf_str.replace(']', '')
            cnf_str = cnf_str.replace(' ', ',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n,,\n')

        print(f'individual accs: {individual_accs[fold-1,...]}')
        print(f'ensemble acc: {ensemble_acc}')
        print(f'ensemble confusion:\n{cnf}')
        print(f'f1 score: {f1[-1]}')

    print('----------------------')

    # histograms over probabilities
    fig, axs = plt.subplots(3, gridspec_kw={'hspace': 0.1})
    fig.suptitle('Accuracies', fontsize=18)
    plt.subplots_adjust(bottom=0.12, top=0.89)
    p = seaborn.histplot(data=acc_thresh, bins=15, kde=False, binrange=(0, 1),
                         stat='probability', ax=axs[0])
    plt.sca(axs[0])
    plt.yticks(fontsize=12)
    p.set_ylabel('Threshold', fontsize=14)
    axs[0].axes.xaxis.set_ticklabels([])

    p = seaborn.histplot(data=acc, bins=15, kde=False, binrange=(0, 1),
                         stat='probability', ax=axs[1])
    plt.sca(axs[1])
    plt.yticks(fontsize=12)
    p.set_ylabel('Combined', fontsize=14)
    axs[1].axes.xaxis.set_ticklabels([])
    p = seaborn.histplot(data=rep_acc, bins=15, kde=False, binrange=(0, 1),
                         stat='probability', ax=axs[2])
    plt.sca(axs[2])
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=15)
    p.set_ylabel('Repetitions', fontsize=14)
    p.set_xlabel('Accuracies', fontsize=15)

    fig, axs = plt.subplots(3, gridspec_kw={'hspace': 0.1})
    fig.suptitle('F1 scores', fontsize=18)
    plt.subplots_adjust(bottom=0.12, top=0.89)
    p = seaborn.histplot(data=f1_thresh, bins=15, kde=False, binrange=(0, 1),
                         stat='probability', ax=axs[0])
    plt.sca(axs[0])
    plt.yticks(fontsize=12)
    p.set_ylabel('Threshold', fontsize=14)
    axs[0].axes.xaxis.set_ticklabels([])

    p = seaborn.histplot(data=f1, bins=15, kde=False, binrange=(0, 1),
                         stat='probability', ax=axs[1])
    plt.sca(axs[1])
    plt.yticks(fontsize=12)
    p.set_ylabel('Combined', fontsize=14)
    axs[1].axes.xaxis.set_ticklabels([])

    p = seaborn.histplot(data=rep_f1, bins=15, kde=False, binrange=(0, 1),
                         stat='probability', ax=axs[2])
    plt.sca(axs[2])
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=15)
    p.set_ylabel('Repetitions', fontsize=14)
    p.set_xlabel('F1 scores', fontsize=15)

    # histograms over individual accuracies
    hp_models = 3
    fig, axs = plt.subplots(len(models)-hp_models, gridspec_kw={'hspace': 0.1})
    fig.suptitle('Accuracies for individual models', fontsize=18)
    plt.subplots_adjust(bottom=0.12, top=0.89)
    for i in range(len(models)-hp_models):
        p = seaborn.histplot(data=individual_accs[:, i], bins=15, kde=False,
                             binrange=(0, 1), stat='probability', ax=axs[i])
        plt.sca(axs[i])
        plt.yticks(fontsize=12)
        if i < len(models)-hp_models-1:
            axs[i].axes.xaxis.set_ticklabels([])
    plt.xticks(fontsize=15)
    p.set_xlabel('Accuracies', fontsize=15)

    # histograms over individual f1
    fig, axs = plt.subplots(len(models)-hp_models, gridspec_kw={'hspace': 0.1})
    fig.suptitle('F1 scores for individual models', fontsize=18)
    plt.subplots_adjust(bottom=0.12, top=0.89)
    for i in range(len(models)-hp_models):
        p = seaborn.histplot(data=individual_f1[:, i], bins=15, kde=False,
                             binrange=(0, 1), stat='probability', ax=axs[i])
        plt.sca(axs[i])
        plt.yticks(fontsize=12)
        if i < len(models)-hp_models-1:
            axs[i].axes.xaxis.set_ticklabels([])
    plt.xticks(fontsize=15)
    p.set_xlabel('F1 scores', fontsize=15)

    print('---------------------')
    print(f'f1: {np.mean(f1)} +- {np.std(f1)}')
    print(f'f1 thresh: {np.mean(f1_thresh)} +- {np.std(f1_thresh)}')
    print('\n')
    print(f'acc: {np.mean(acc)} +- {np.std(acc)}')
    print(f'acc: {np.mean(acc_thresh)} +- {np.std(acc_thresh)}')
    print('--------------------')
    print('-----------------')
    print('-----------------')
    print('certainties::::')
    print(f'accuracy certainty 1: {np.mean(acc_c1)} +- {np.std(acc_c1)}')
    print(f'accuracy certainty 2: {np.mean(acc_c2)} +- {np.std(acc_c2)}')
    if len(acc_c3) > 0:
        print(f'accuracy certainty 3: {np.mean(acc_c3)} +- {np.std(acc_c3)}')
    print('-----------------')
    print(f'f1 certainty 1: {np.mean(f1_c1)} +- {np.std(f1_c1)}')
    print(f'f1 certainty 2: {np.mean(f1_c2)} +- {np.std(f1_c2)}')
    if len(f1_c3) > 0:
        print(f'f1 certainty 3: {np.mean(f1_c3)} +- {np.std(f1_c3)}')
    print('-----------------')
    print(f'Amounts:: c1: {len(p1)}, c2: {len(p2)}, c3: {len(p3)}')

    print(f'rep f1: {np.mean(rep_f1)} +- {np.std(rep_f1)}')
    print(f'rep acc: {np.mean(rep_acc)} +- {np.std(rep_acc)}')

    print('----')
    print('Recall, rep, comb, thresh')
    print(f'{np.mean(recall_rep)} +- {np.std(recall_rep)}')
    print(f'{np.mean(recall)} +- {np.std(recall)}')
    print(f'{np.mean(recall_thresh)} +- {np.std(recall_thresh)}')
    print('Recall certainties')
    print(f'{np.mean(recall_c1)} +- {np.std(recall_c1)}')
    print(f'{np.mean(recall_c2)} +- {np.std(recall_c2)}')
    print(f'{np.mean(recall_c3)} +- {np.std(recall_c3)}')
    print('----')
    print('Precision, rep, comb, thresh')
    print(f'{np.mean(precision_rep)} +- {np.std(precision_rep)}')
    print(f'{np.mean(precision)} +- {np.std(precision)}')
    print(f'{np.mean(precision_thresh)} +- {np.std(precision_thresh)}')
    print('Precision certainties')
    print(f'{np.mean(precision_c1)} +- {np.std(precision_c1)}')
    print(f'{np.mean(precision_c2)} +- {np.std(precision_c2)}')
    print(f'{np.mean(precision_c3)} +- {np.std(precision_c3)}')

    print('-----------------------')
    print(incorr_info)

    if args.confusion and False:
        # writes confusion matrices to file (disabled),
        # currently overwrites and some incorrect formatting.. fix this
        confusion_file.write('Confusions for all repetitions,,\n')
        for fold, c in enumerate(all_rep_cnf):
            confusion_file.write(f'matrix for fold {fold+1},,\n')
            cnf_str = str(c.astype(int)).replace(' [', '')
            cnf_str = str(cnf_str).replace('[ ', '')
            cnf_str = str(cnf_str).replace('[', '')
            cnf_str = str(cnf_str).replace(']', '')
            cnf_str = cnf_str.replace(' ', ',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n')
        confusion_file.write('Average of above,,\n')
        cnf_str = str(np.mean(all_rep_cnf, axis=0)).replace(' [', '')
        cnf_str = str(cnf_str).replace('[ ', '')
        cnf_str = str(cnf_str).replace('[', '')
        cnf_str = str(cnf_str).replace(']', '')
        cnf_str = cnf_str.replace(' ', ',')
        confusion_file.write(cnf_str)
        confusion_file.write('\n\n')

        confusion_file.write('Confusions for combined scores,,\n')
        for fold, c in enumerate(all_confusions):
            confusion_file.write(f'matrix for fold {fold+1},,\n')
            cnf_str = str(c.astype(int)).replace(' [', '')
            cnf_str = str(cnf_str).replace('[ ', '')
            cnf_str = str(cnf_str).replace('[', '')
            cnf_str = str(cnf_str).replace(']', '')
            cnf_str = cnf_str.replace(' ', ',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n')
        confusion_file.write('Average of above,,\n')
        cnf_str = str(np.mean(all_confusions, axis=0)).replace(' [', '')
        cnf_str = str(cnf_str).replace('[ ', '')
        cnf_str = str(cnf_str).replace('[', '')
        cnf_str = str(cnf_str).replace(']', '')
        cnf_str = cnf_str.replace(' ', ',')
        confusion_file.write(cnf_str)
        confusion_file.write('\n\n')

        confusion_file.write('Confusions for combined scores w threshold,,\n')
        for fold, c in enumerate(all_confusions_thresh):
            confusion_file.write(f'matrix for fold {fold+1},,\n')
            cnf_str = str(c.astype(int)).replace(' [', '')
            cnf_str = str(cnf_str).replace('[ ', '')
            cnf_str = str(cnf_str).replace('[', '')
            cnf_str = str(cnf_str).replace(']', '')
            cnf_str = cnf_str.replace(' ', ',')
            confusion_file.write(cnf_str)
            confusion_file.write('\n')
        confusion_file.write('Average of above,,\n')
        cnf_str = str(np.mean(all_confusions_thresh, axis=0)).replace(' [', '')
        cnf_str = str(cnf_str).replace('[ ', '')
        cnf_str = str(cnf_str).replace('[', '')
        cnf_str = str(cnf_str).replace(']', '')
        cnf_str = cnf_str.replace(' ', ',')
        confusion_file.write(cnf_str)
        confusion_file.write('\n\n')

        confusion_file.close()

    plot_confusion_matrix_mean(all_confusions, [0, 1, 2],
                               title='Combined scores')
    plot_confusion_matrix_mean(all_confusions_thresh, [0, 1, 2],
                               title='Combined scores, with threshold')
    plot_confusion_matrix_mean(all_rep_cnf, [0, 1, 2], title='Repetitions')
    plot_confusion_matrix_mean(all_confusions_ignored, [0, 1, 2],
                               title='Ignored')

    # hist_save_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_prob_matrices-test.npz'
    # np.savez(hist_save_path, reps=prob_reps, comb=prob_mean_all, reps_max=prob_reps_pred, comb_max=prob_mean_pred)

    stuff_sp = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '-stat-stuff.npz'
    np.savez(stuff_sp, rep_acc=rep_acc, rep_f1=rep_f1, comb_acc=acc,
             comb_f1=f1, thresh_acc=acc_thresh, thresh_f1=f1_thresh,
             acc_ind=individual_accs, f1_ind=individual_accs)

    print(confusion_matrix(rep_labels, rep_list))

    # scatter plots of mean accuracies etc, didn't tell anythoing...
    plt.figure()
    plt.scatter(certs[:, 0], certs[:, 1])
    means = np.array([[x, np.mean(certs[certs[:, 0] == x, 1])]
                      for x in range(1, int(np.max(certs[:, 0])) + 1)])
    print(means)
    plt.scatter(means[:, 0], means[:, 1], marker='x', s=100)
    for i in range(1, 4):
        print(f'certainty {i}, acc: {np.mean(corr_certs[corr_certs[:, 0] == i, 1])}')
    plt.show()


def str2bool(v):
    ''' pass bool wtih argparse'''
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
