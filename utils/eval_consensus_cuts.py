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

OUT_DIR = '/home/filipkr/Documents/xjob/consensus_w_cuts'



def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx, 1]
    in_cohort_nbr = data[idx, 3]
    indices = np.where(data[:, 1] == subj)[0]
    idx_same_leg = np.where(data[indices, 3] == in_cohort_nbr)

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
    lit = args.root.split('/')[-1]
    print(lit)
    num_folds = 10

    print(lit)

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

    ensembles = [os.path.join(args.root, i) for i in models] \
        if 'ensembles' in args.root else [args.root + str(i) for i in models]


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
   
    if 'trunk' in lit:
        folds = [1, 2, 4, 5, 6, 7, 9, 10]
        cuts = ['-cut_rate0.05', '-cut_rate0.1']
    # hip
    if 'hip' in lit:
        folds = [1, 2, 3, 5, 8, 9]
        cuts = ['', '-cut_rate0.05']
    # femval
    if 'femval' in lit:
        folds = [2, 3, 5, 6, 7, 8, 9, 10]
        cuts = ['-cut5', '-cut_rate0.05', '-cut_rate0.1']
    # kmfp
    if 'kmfp' in lit:
        folds = [2, 5, 6, 9]
        cuts = ['', '-cut5', '-cut_rate0.05']
    # fms
    if 'fms' in lit:
        folds = [1, 2, 6, 8]
        cuts = ['-cut5', '-cut_rate0.05']
    # foot
    if 'foot' in lit:
        folds = [2, 3, 5, 8]
        cuts = ['', '-cut5', '-cut_rate0.05']
    cut_results = {}

    dp = f'/home/filipkr/Documents/xjob/data/datasets/data_{ds_name}.npz'

    dataset = np.load(dp)
    y = dataset['labels']
    all_cut_preds = np.zeros((num_folds, len(cuts), len(y), 3))

    final_predictions = np.zeros((len(y), 3))
    cut_accs = []
    ensemble_accs = []
    
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

        cut_predictions = np.zeros((len(y), 3))
        
        # for fold in range(1, num_folds + 1):
        for fold in folds:
            paths = [os.path.join(root, f'model_fold_{fold}.hdf5')
                     for root in ensembles]

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
                probs = coral.ordinal_softmax(result).numpy() \
                    if 'coral' in model_path else result
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
                                    labels=[0, 1, 2], average='macro')

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
            final_predictions = final_predictions + ensemble_probs
            cut_predictions = cut_predictions + ensemble_probs
            print(final_predictions.shape)

            all_cut_preds[fold-1, cut_index, ...] = ensemble_probs

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

            all_rep_cnf[fold-1, ...] = confusion_matrix(corr_rep, pred_rep,
                                                        labels=[0, 1, 2])
            print(all_rep_cnf[fold-1, ...])
            rep_f1.append(f1_score(corr_rep, pred_rep,
                                labels=[0, 1, 2], average='macro'))
            rep_acc.append(accuracy_score(corr_rep, pred_rep))
            precision_rep.append(precision_score(corr_rep, pred_rep,
                                labels=[0, 1, 2], average='macro'))
            recall_rep.append(recall_score(corr_rep, pred_rep,
                                labels=[0, 1, 2], average='macro'))

            hist_index = 0
            for i in range(len(y)):
                if i not in subject_indices:
                    subject_indices, global_ind = get_same_subject(info_file, i)

                    y_subj = y[subject_indices]
                    pred_subj = ensemble_probs[subject_indices, ...]
                    summed = np.mean(pred_subj, axis=0)

                    test_median = True

                    if test_median:
                        pred = np.argmax(pred_subj, axis=1)

                        corr_combined = int(np.ceil(np.median(y_subj)))
                        pred_combined = int(np.ceil(np.median(pred)))

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
            precision.append(precision_score(truth, preds, labels=labels,
                                             average='macro'))
            recall.append(recall_score(truth, preds, labels=labels,
                                       average='macro'))
            labels = np.sort(np.unique(np.append(truth_thresh, preds_thresh)))
            f1_thresh.append(f1_score(truth_thresh, preds_thresh,
                                      labels=labels, average='macro'))
            acc_thresh.append(accuracy_score(truth_thresh, preds_thresh))
            precision_thresh.append(precision_score(truth_thresh, preds_thresh,
                                                    labels=labels,
                                                    average='macro'))
            recall_thresh.append(recall_score(truth_thresh, preds_thresh,
                                              labels=labels, average='macro'))

            cnf = confusion_matrix(truth_thresh, preds_thresh, labels=[0, 1, 2])
            cnf_all_folds_thresh = cnf_all_folds_thresh + cnf
            all_confusions_thresh[fold-1,...] = cnf
            cnf = confusion_matrix(truth_ignored, preds_ignored,
                                   labels=[0, 1, 2])
            all_confusions_ignored[fold-1,...] = cnf
            cnf = confusion_matrix(truth, preds, labels=[0, 1, 2])
            cnf_all_folds = cnf_all_folds + cnf
            all_confusions[fold-1,...] = cnf

            print(f'individual accs: {individual_accs[fold-1,...]}')
            print(f'ensemble acc: {ensemble_acc}')
            print(f'ensemble confusion:\n{cnf}')
            print(f'f1 score: {f1[-1]}')

            ensemble_accs.append(rep_acc)

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

        cut_accs.append(accuracy_score(np.argmax(cut_predictions, axis=1), y))

    print('----------------------')

    print(final_predictions)
    print(np.argmax(final_predictions, axis=1))
    print(final_predictions.shape)
    print(np.argmax(final_predictions, axis=1).shape)
    print(np.shape(y))
    print(accuracy_score(y, np.argmax(final_predictions, axis=1)))

    print(y)
    print(cut_accs)
    print(ensemble_accs)

    df = pd.DataFrame()
    df['predictions'] = np.argmax(final_predictions, axis=1)
    df['correct'] = y

    df.to_csv(os.path.join(OUT_DIR, f'{lit}-preds.csv'))


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
