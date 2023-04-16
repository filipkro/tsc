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
    lit = args.root.split('/')[-1]
    print(lit)
    num_folds = 10

    print(lit)

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

    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + ds_name + '.npz'
    dp100 = '/home/filipkr/Documents/xjob/data/datasets/data_' + ds_name + '_len100.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + ds_name + '-info.txt'
    dataset = np.load(dp)
    dataset100 = np.load(dp100)
    
    x = dataset['mts']
    x100 = dataset100['mts']
    y = dataset['labels']

    all_confusions = np.zeros((num_folds,3,3))
    all_confusions_thresh = np.zeros((num_folds,3,3))
    all_confusions_ignored = np.zeros((num_folds,3,3))
    all_rep_cnf = np.zeros((num_folds,3,3))

    if lit == 'trunk_con':
        models = ['coral-2', 'conf-100-0', 'xx-conf-100-2']
        weights = np.array([[0.7/2,0.9,1.1/2], [1.1/2, 0, 0], [0,0,1.1/2]])
    elif lit == 'hip_con':
        models = ['coral-2', 'conf-100-0', 'xx-conf-100-2']
        weights = np.array([[1/2,0.9,1.1/2], [1.2/2, 0, 0], [0,0,1.1/2]])
    elif lit == 'femval_con':
        models = ['coral-2', 'conf-100-0', 'conf-2']
        weights = np.array([[1/2,1,1.1/2], [1/2, 0, 0], [0,0,1.1/2]])
    elif lit == 'kmfp_con':
        models = ['coral-3', 'xx-conf-100-0']
        weights = np.array([[0,0.85,1.5], [1,0,0]])
    elif lit == 'fms_con':
        models = ['coral-3', 'conf-0', 'xx-conf-2']
        weights = np.array([[1/3,0.9,1.1/2], [2/3, 0, 0], [0,0,1.05/2]])
    elif lit == 'foot_con':
        models = ['coral-3', ':::xx-conf-0', 'conf-2']
        weights = np.array([[1.1/4,0.8,0.8/2], [3/4, 0, 0], [0,0,0.8/2]])
    elif lit == 'trunk_aug':
        models = ['coral-1', 'conf-0', 'conf-100-1', 'conf-2']
        weights = np.array([[0.5,0.5,0.5], [0.5,0,0],[0,0.5,0],[0,0,0.5]])
    elif lit == 'hip_aug':
        models = ['coral-1', 'conf-0', 'conf-100-1', 'conf-100-2']
        weights = np.array([[0.5,0.5,0.6], [0.5, 0, 0], [0, 0.5, 0], [0,0,0.5]])
    elif lit == 'femval_aug':
        models = ['coral-1', 'conf-0', 'conf-1', 'conf-2']
        weights = np.array([[0.6,0.6,1],[0.4,0,0], [0,0.4,0],[0,0,0.1]])
    elif lit == 'kmfp_aug':
        models = ['coral-1', 'conf-0', 'conf-2']
        weights = np.array([[0.6,1.1,0.9], [0.4,0,0], [0,0,0.3]])
    elif lit == 'fms_aug':
        models = ['coral-1', 'conf-0', 'conf-100-1']
        weights = np.array([[0.6,0.6,1.1], [0.4,0,0], [0,0.4,0]])
    elif lit == 'foot_aug':
        models = ['coral-1', 'conf-100-0', 'conf-100-2']
        weights = np.array([[0.6,1,0.9], [0.4,0,0],[0,0,0.3]])

    print(args.root)
    print(models)

    ensembles = [os.path.join(args.root, i) for i in models] if 'ensembles' in args.root else [
        args.root + str(i) for i in models]

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

    prob_mean_all = -1* np.ones((num_folds,50,3,3))
    prob_mean_pred = -1* np.ones((num_folds,50,3,3))
    prob_reps = -1* np.ones((num_folds,150,3,3))
    prob_reps_pred = -1* np.ones((num_folds,150,3,3))

    max_nbrs = 0

    rep_list = []
    rep_labels = []
    individual_accs = np.zeros((num_folds, len(models)))
    individual_f1 = np.zeros((num_folds, len(models)))

    if args.confusion:
        path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_confusion-test.csv'
        confusion_file = open(path, 'w')

    for fold in range(1, num_folds + 1):
        paths = [os.path.join(root, f'model_fold_{fold}.hdf5') 
                 for root in ensembles]

        y_val = y
        max_nbrs = np.max((len(y_val), max_nbrs))
        all_probs = np.zeros((len(models), x.shape[0], 3))

        corr_rep = []
        pred_rep = []

        for model_i, model_path in enumerate(paths):
            input = x100 if '-100-' in model_path else x
            model = keras.models.load_model(model_path, custom_objects={
                                            'CoralOrdinal': coral.CoralOrdinal,
                                            'OrdinalCrossEntropy':
                                            coral.OrdinalCrossEntropy,
                                            'MeanAbsoluteErrorLabels':
                                            coral.MeanAbsoluteErrorLabels,
                                            'ConfusionCrossEntropy':
                                            ConfusionCrossEntropy})

            result = model.predict(input)
            probs = coral.ordinal_softmax(
                result).numpy() if 'coral' in model_path else result

            probs = probs * weights[model_i, ...]
            all_probs[model_i, ...] = probs

            subject_indices = []

            y_corr = []
            y_pred_comb = []
            for i in range(len(y)):
                if i not in subject_indices:
                    subject_indices, _ = get_same_subject(info_file, i)

                    y_subj = y[subject_indices]
                    pred_subj = probs[subject_indices, ...]

                    y_corr.append(int(np.median(y_subj)))
                    y_pred_comb.append(np.argmax(np.mean(pred_subj, axis=0)))

            individual_accs[fold-1, model_i] = accuracy_score(y_corr,
                                                              y_pred_comb)
            individual_f1[fold-1, model_i] = f1_score(y_corr, y_pred_comb,
                                                      labels=[0, 1, 2],
                                                      average='macro')

        subject_indices = []
        preds = []
        truth = []

        preds_thresh = []
        truth_thresh = []
        preds_ignored = []
        truth_ignored = []

        ensemble_probs = np.sum(all_probs, axis=0)
        print(ensemble_probs)
        print(ensemble_probs.shape)

        ensemble_probs = (ensemble_probs > 0.2) * ensemble_probs
        hist_index = 0
        print()
        for i in range(len(y)):
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

        rep_list = np.append(rep_list, pred_rep, axis=0)
        rep_labels = np.append(rep_labels, corr_rep, axis=0)

        hist_index = 0
        for i in range(len(y)):
            if i not in subject_indices:
                subject_indices, _ = get_same_subject(info_file, i)

                y_subj = y[subject_indices]
                pred_subj = ensemble_probs[subject_indices, ...]
                summed = np.mean(pred_subj, axis=0)

                corr_combined = int(np.ceil(np.median(y_subj)))
                pred_combined = int(np.argmax(np.mean(pred_subj, axis=0)))
                preds.append(int(np.round(pred_combined)))
                truth.append(int(np.round(corr_combined)))

                if lit == 'Sigrid-Undset' or lit == 'Mikhail-Sholokhov':
                    thres = 0.35
                else:
                    thres = 0.4

                if np.max(summed) > thres:
                    preds_thresh.append(int(np.round(pred_combined)))
                    truth_thresh.append(int(np.round(corr_combined)))
                else:
                    preds_ignored.append(int(np.round(pred_combined)))
                    truth_ignored.append(int(np.round(corr_combined)))

                hist_index += 1

        ensemble_acc = accuracy_score(truth, preds)

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

        all_confusions_thresh[fold-1,...] = cnf
        cnf = confusion_matrix(truth_ignored, preds_ignored, labels=[0, 1, 2])
        all_confusions_ignored[fold-1,...] = cnf
        cnf = confusion_matrix(truth, preds, labels=[0, 1, 2])
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

    fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.1})
    fig.suptitle(f'Accuracies', fontsize=18)
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
    axs[1].axes.xaxis.set_ticklabels([])
    p = seaborn.histplot(data=rep_acc,bins=15,kde=False, binrange=(0,1), stat='probability', ax=axs[2])
    plt.sca(axs[2])
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=15)
    p.set_ylabel(f'Repetitions', fontsize=14)
    p.set_xlabel('Accuracies', fontsize=15)

    fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.1})
    fig.suptitle(f'F1 scores', fontsize=18)
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
    print('combined metrics')
    print(f'f1: {np.mean(f1)} +- {np.std(f1)}')
    print(f'f1 thresh: {np.mean(f1_thresh)} +- {np.std(f1_thresh)}')
    print('\n')
    print(f'acc: {np.mean(acc)} +- {np.std(acc)}')
    print(f'acc: {np.mean(acc_thresh)} +- {np.std(acc_thresh)}')
    print('--------------------')
    print('-----------------')
    print('-----------------')
    print('repetition metrics')
    print(f'rep f1: {np.mean(rep_f1)} +- {np.std(rep_f1)}')
    print(f'rep acc: {np.mean(rep_acc)} +- {np.std(rep_acc)}')

    print('----')
    print(f'Recall, rep, comb, thresh')
    print(f'{np.mean(recall_rep)} +- {np.std(recall_rep)}')
    print(f'{np.mean(recall)} +- {np.std(recall)}')
    print(f'{np.mean(recall_thresh)} +- {np.std(recall_thresh)}')
    print('----')
    print(f'Precision, rep, comb, thresh')
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

    plot_confusion_matrix_mean(all_confusions, [0,1,2], title='Combined scores')
    plot_confusion_matrix_mean(all_confusions_thresh, [0,1,2], title='Combined scores, with threshold')
    plot_confusion_matrix_mean(all_rep_cnf, [0,1,2], title='Repetitions')
    plot_confusion_matrix_mean(all_confusions_ignored, [0,1,2], title='Ignored')

    if False: # save all metrics for further eval etc
        hist_save_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '_prob_matrices-test.npz'
        np.savez(hist_save_path, reps=prob_reps, comb=prob_mean_all, reps_max=prob_reps_pred, comb_max=prob_mean_pred)

        stuff_sp = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/' + lit + '-stat-stuff.npz'
        np.savez(stuff_sp, rep_acc=rep_acc, rep_f1=rep_f1, comb_acc=acc, comb_f1=f1, thresh_acc=acc_thresh, thresh_f1=f1_thresh, acc_ind=individual_accs, f1_ind=individual_accs)

    print(confusion_matrix(rep_labels, rep_list))

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
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--uncert', default='')
    parser.add_argument('--confusion', type=str2bool,
                        nargs='?', default=False)
    args = parser.parse_args()
    main(args)
