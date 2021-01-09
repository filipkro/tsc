import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from grad_cam import make_gradcam_heatmap
from xcm_grad_cam import make_gradcam_heatmap as xcm_hm
from grad_test0 import check_grad

def get_same_subject(info_file, idx):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, :4], dtype=int)

    subj = data[idx,1]
    indices = np.where(data[:,1] == subj)[0]
    print(indices)
    return indices

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
    dataset = np.load('/home/filipkr/Documents/xjob/data/datasets/data_Erik-Axel-Karlfeldt.npz')
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_Erik-Axel-Karlfeldt-info.txt'
    # print(os.path.basename(args.root).split('_')[0])
    # lit = os.path.basename(args.root).split('_')[0]
    # dp = os.path.join(args.data, 'data_') + 'Herta-Moller.npz'
    # dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']
    ind_val = np.load(os.path.join(args.root, 'indices.npz'))
    ind_t = np.load(idx_path)
    test_idx = np.append(ind_val['val_idx'], ind_t['test_idx'].astype(np.int))
    # test_idx = ind_t['test_idx'].astype(np.int)
    # test_idx = ind_val['val_idx'].astype(np.int)
    x_test = x[test_idx, ...]
    y_test = y[test_idx]
    print(test_idx)

    model_path = os.path.join(args.root, 'best_model.hdf5')
    model = keras.models.load_model(model_path)
    result = model.predict(x_test)
    y_pred = np.argmax(result, axis=1)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, [0,1,2], title='Each repetition')
    print(y_test)
    print(result)
    print(cnf_matrix)

    idx = []
    correct = 0
    corr_mean = 0
    pred_combined = []
    y_combined = []
    for i in test_idx:
        if i not in idx:
            idx = get_same_subject(info_file, i)

            x_subj = x[idx, ...]
            y_subj = y[idx]
            result = model.predict(x_subj)
            print('result for indices: {}'.format(idx))
            print('likelihoods')
            print(result)
            print('correct')
            print(y_subj)
            print('summed likelihoods')
            print(np.sum(result, axis=0))
            print('true label')
            print(np.median(y_subj))
            pred_combined.append(np.argmax(np.sum(result, axis=0)))
            y_combined.append(int(np.median(y_subj)))
            correct += 1*(int(np.median(y_subj)) == np.argmax(np.sum(result, axis=0)))
            corr_mean += 1*(int(np.round(np.mean(y_subj))) == np.argmax(np.sum(result, axis=0)))
            print('\n \n')

    print(correct)
    print(corr_mean)
    combined_cm = confusion_matrix(y_combined, pred_combined)
    print(combined_cm)
    plot_confusion_matrix(combined_cm, [0,1,2], title='combined score')
    assert False


    tv = np.append(test_idx, val_idx)
    x_tv = x[tv, ...]
    y_tv = y[tv]


    model_path = os.path.join(args.root, 'best_model.hdf5')
    model = keras.models.load_model(model_path)

    result = model.predict(x_test)
    result_tv = model.predict(x_tv)

    y_pred = np.argmax(result, axis=1)
    y_pred_tv = np.argmax(result_tv, axis=1)

    print('result: {}'.format(result))
    #print('y_pred_like: {}'.format(y_pred_like))
    print('y_pred: {}'.format(y_pred))
    print('y_test: {}'.format(y_test))
    print('result_tv: {}'.format(result_tv))
    #print('y_pred_like_tv: {}'.format(y_pred_like_tv))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    if args.outdir != '':
        savename = os.path.join(args.outdir, 'test.png')
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                          title='Confusion matrix, without normalization',
                          savename=savename)

    cnf_matrix = confusion_matrix(y_tv, y_pred_tv)
    if args.outdir != '':
        savename = os.path.join(args.outdir, 'tv.png')
    # np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                          title='Confusion matrix, without normalization',
                          savename=savename)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
