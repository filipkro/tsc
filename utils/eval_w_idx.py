import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import itertools
from grad_cam import make_gradcam_heatmap
from xcm_grad_cam import make_gradcam_heatmap as xcm_hm
from grad_test0 import check_grad


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


def plot_w_cam(x, cam, y_pred, y_true):
    max_idx = np.where(x[:, 0] < -900)[0][0]
    # plt.plot(x[:max_idx,0])
    plt.plot(x[:max_idx, 0])
    sc = plt.scatter(np.linspace(0, max_idx - 1, max_idx),
                     x[:max_idx, 0], c=cam[:max_idx])
    plt.colorbar(sc)
    plt.show()


def main(args):

    if args.dataset != '':
        dataset = np.load(args.dataset)
        lit = ''
    else:
        print(os.path.basename(args.root).split('_')[0])
        lit = os.path.basename(args.root).split('_')[0]
        dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
        dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']

    ind = np.load(os.path.join(args.root, 'best_model.hdf5'))
    test_idx = ind['test'].astype(np.int)
    train_idx = ind['train'].astype(np.int)
    val_idx = ind['val'].astype(np.int)
    x_train = x[train_idx, ...]
    y_train = y[train_idx]
    x_val = x[val_idx, ...]
    y_val = y[val_idx]
    x_test = x[test_idx, ...]
    y_test = y[test_idx]
    print(test_idx)

    tv = np.append(test_idx, val_idx)
    x_tv = x[tv, ...]
    y_tv = y[tv]


    # x = x_test
    # y = y_test
    # print(x.shape)
    # print(y.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    plot_all = True

    model_path = os.path.join(args.root, 'best_model.hdf5')
    model = keras.models.load_model(model_path)
    # y_pred_like, cam = model.predict(x)
    # y_pred_like_tv, cam_tv = model.predict(x_tv)
    result = model.predict(x_test)
    result_tv = model.predict(x_tv)

    y_pred = np.argmax(result, axis=1)
    y_pred_tv = np.argmax(result_tv, axis=1)
    # print(y_test)
    # print(y_pred)
    # print(result)

    print('result: {}'.format(result))
    print('y_pred_like: {}'.format(y_pred_like))
    print('y_pred: {}'.format(y_pred))
    print('y_test: {}'.format(y_test))
    print('result_tv: {}'.format(result_tv))
    print('y_pred_like_tv: {}'.format(y_pred_like_tv))
    print('y_pred_tv: {}'.format(y_pred_tv))
    print('y_test_tv: {}'.format(y_tv))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    if args.outdir != '':
        savename = os.path.join(args.outdir, 'test.png')
    # np.set_printoptions(precision=2)
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
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
