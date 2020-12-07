import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import itertools
from grad_cam import make_gradcam_heatmap
from xcm_grad_cam import make_gradcam_heatmap as xcm_hm
from grad_test import check_grad
import pandas as pd

def get_same_subject(info_file, idx):
    data = pd.read_csv(info_file, delimiter=',')
    data = np.array(data.values[5:,:4], dtype=int)

    subj = data[idx,1]
    indices = np.where(data[:,1] == subj)[0]
    print(indices)
    return indices

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

def main(args):
    lit = os.path.basename(args.root).split('_')[0]
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    dataset = np.load(dp)
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    x = dataset['mts']
    y = dataset['labels']

    get_same_subject(info_file, 12)

    idx = np.load(os.path.join(args.root, 'idx.npz'))
    test_idx = idx['test']
    # x_test =
    a = test_idx[25]
    print(a)
    subj = get_same_subject(info_file, a)

    # print(y[test_idx])

    model_path = os.path.join(args.root, 'best_model.hdf5')
    model = keras.models.load_model(model_path)

    x_subj = x[subj]

    result = model.predict(x_subj)
    print(np.sum(result,axis=0))
    print(np.median(y[subj]))

    print(result)



if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('train')
    parser.add_argument('root')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
