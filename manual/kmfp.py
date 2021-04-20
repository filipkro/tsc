import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, accuracy_score

from scipy.fft import fft

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main(args):
    data = np.load(args.data)
    x = data['mts']
    y = data['labels']
    print(x.shape)
    print(y.shape)
    idx = np.load(
        '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz')
    # x_test = x[idx['test_idx'], ...]

    mean_diffs = np.zeros((x.shape[0], 3))
    max_diffs = np.zeros((x.shape[0], 3))
    fourier_var = np.zeros((x.shape[0], 3))
    preds = np.zeros(y.shape)
    preds2 = np.zeros(y.shape)
    # peaks = np.zeros(y.shape)

    transforms = np.zeros((x.shape[0], 25))

    feats = np.zeros((x.shape[0], 1))

    inputs = np.zeros((x.shape[0], x.shape[-1], 3))

    for i, (inp, label) in enumerate(zip(x, y)):
        # print(label[0])
        max_idx = np.where(inp[:, 0] < -900)[0][0]

        peak = np.argmax(inp[:max_idx, 0])
        min = np.max((0, peak - 10))
        max = np.min((max_idx, peak + 10))

        fourier = fft(inp[:max_idx, 1], 256)
        transforms[i, ...] = fourier[:25]
        plt.figure(int(label[0]))
        plt.plot(inp[:max_idx, 1])

        # print(np.mean(x[:max_idx, -1]))
        # print()
        # mean_diffs[i, int(label[0])] = np.mean(inp[min:max, 1])
        # max_diffs[i, int(label[0])] = np.max(inp[min:max, 1])
        # fourier_var[i, int(label[0])] = np.var(fourier[:25])
        mean_diffs[i, int(label[0])] = np.mean(inp[:max_idx, 0])
        max_diffs[i, int(label[0])] = np.mean(inp[:max_idx, 1])

        inputs[i, :, int(label[0])] = np.mean(inp[:max_idx, :], axis=0)

        # feats[i, ...] = [np.mean(inp[min:max, -1]), np.var(fourier[:25]),
        #                  np.max(inp[:, 0]) - np.min(inp[:, 0]), np.max(inp[:,-1]), np.min(inp[:,-1])]
        # feats[i, ...] = [np.mean(inp[min:max, -1])]#, np.max(inp[:,-1]), np.min(inp[:,-1])]
        #
        # disc = np.mean(inp[min:max, -1])
        # disc2 = np.var(fourier[:25])
        #
        # if disc < -0.01:
        #     preds[i] = 0
        # elif disc < -0.0005:
        #     preds[i] = 1
        # else:
        #     preds[i] = 2
        #
        # if disc2 > 1:
        #     preds2[i] = 0
        # elif disc2 > 0.1:
        #     preds2[i] = 1
        # else:
        #     preds2[i] = 2

    for i in range(inputs.shape[1]):
        print(f'input {i}:\n{np.mean(inputs[:, i, :], axis=0)} +- {np.std(inputs[:, i, :],axis=0)}')
    # print(
    #     f'Mean: \n{np.mean(mean_diffs, axis=0)} +- {np.std(mean_diffs,axis=0)}')
    # print(f'Max: \n{np.mean(max_diffs, axis=0)} +- {np.std(max_diffs,axis=0)}')
    # print(
    #     f'Fourier: \n{np.mean(fourier_var, axis=0)} +- {np.std(fourier_var,axis=0)}')
    #
    # cm = confusion_matrix(y, preds, labels=range(3))
    # cm2 = confusion_matrix(y, preds2, labels=range(3))
    # print(cm)
    # print(cm2)
    # print(accuracy_score(y, preds))
    # print(accuracy_score(y, preds2))
    # print(np.sum(np.abs(preds - preds2)))
    #
    # # print(idx['train_idx'])
    # # print(transforms.shape)
    # plt.show()

    # plt.plot(transforms.T)
    # plt.show()

    # print(mean_diffs)
    # print(max_diffs)
    # x_train = feats[idx['train_idx']]
    # y_train = y[idx['train_idx']]
    # x_test = feats[idx['test_idx']]
    # y_test = y[idx['test_idx']]
    # print(y_train.shape)
    # print(y_test.shape)
    # print(y_train[:, 0].shape)
    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #
    # clf.fit(x_train, y_train[:, 0])
    # score = clf.score(x_test, y_test[:, 0])
    # score = clf.score(x_train, y_train[:, 0])
    # print(score)
    #
    # y_pred = clf.predict(x_test)
    # cm3 = confusion_matrix(y_test, y_pred, labels=range(3))
    # print(cm3)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    main(args)
