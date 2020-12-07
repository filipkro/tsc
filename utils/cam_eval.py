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

    if args.train.split('.')[-1] == 'npz':
        ind = np.load(args.train)
        print(ind.files)
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
        # print(test_idx)
        # print(y_test)
        # print(y_val)
        # print(y)
        # print(len(y))

        # assert False

        tv = np.append(test_idx, val_idx)
        x_tv = x[tv, ...]
        y_tv = y[tv]
    else:
        train_idx = np.load(args.train)
        x_tv = np.delete(x, train_idx, axis=0)
        y_tv = np.delete(y, train_idx)

        if args.test_idx != '':
            test_idx = np.load(args.test_idx)
            x_test = x_tv[test_idx, ...]
            y_test = y_tv[test_idx]

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

    if len(result) > 3:
        y_pred = np.argmax(result, axis=1)
        y_pred_tv = np.argmax(result_tv, axis=1)
        # print(y_test)
        # print(y_pred)
        # print(result)

        cnf_matrix = confusion_matrix(y_test, y_pred)
        # np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                              title='Confusion matrix, without normalization')

        cnf_matrix = confusion_matrix(y_tv, y_pred_tv)
        # np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                              title='Confusion matrix, without normalization')
        # print(model.summary())
        # plt.show()
        i = 11

        print(y_test[i])
        print(y_pred[i])
        print(result[i,...])
        max_idx = np.where(x_test[i, :, 0] < -900)[0][0]
        # print(model.summary())
        # assert False
        # L1,L2 = xcm_hm(np.expand_dims(x_test[i, :max_idx, ...], 0), model)
        L1, L2 = check_grad(np.expand_dims(x_test[i, :max_idx, ...], 0), model)
        # print(result)
        plt.figure()
        plt.plot(L1)
        plt.figure()
        plt.plot(L2[:,0])
        plt.plot(L2[:,1])
        fig, axs = plt.subplots(2)
        axs[0].plot(x_test[i, :max_idx, 0])
        sc = axs[0].scatter(np.linspace(0, max_idx - 1, max_idx),
                            x_test[i, :max_idx, 0],
                            c=L2[:max_idx,0], cmap='cool',
                            vmin=0, vmax=1)
        axs[1].plot(x_test[i, :max_idx, 1])
        sc = axs[1].scatter(np.linspace(0, max_idx - 1, max_idx),
                            x_test[i, :max_idx, 1],
                            c=L2[:max_idx,1], cmap='cool',
                            vmin=0, vmax=1)
        cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)

        # fig.suptitle('Dataset: {}, Class {} predicted as {}'.format(lit, y_test[i], y_pred[i]))
        plt.show()



        # check_grad(np.expand_dims(x_test[i,:max_idx, ...], 0), model)
        # make_gradcam_heatmap(np.expand_dims(x_test[i,:max_idx, ...], 0), model)

        # assert False
        # hm1, hm2 = xcm_hm(np.expand_dims(x_test[i, ...], 0), model)
        # hm = make_gradcam_heatmap(np.expand_dims(x_test[i, ...], 0), model, 'conv1d_3', [
        #                           'global_average_pooling1d', 'result'])
        # plt.figure()
        # plt.plot(hm)

        # plt.plot(x[:max_idx,0])
        fig, axs = plt.subplots(x_test.shape[2])
        fig.suptitle('Dataset: {}, Class {} predicted as {}'.format(lit, y_test[i], y_pred[i]))
        if x_test.shape[2] > 1:
            for j in range(x_test.shape[2]):
                axs[j].plot(x_test[i, :max_idx, j])
                sc = axs[j].scatter(np.linspace(0, max_idx - 1, max_idx),
                                    x_test[i, :max_idx, j],
                                    c=hm[:max_idx], cmap='cool',
                                    vmin=-1, vmax=1)
                # axs[j].set_axis_off()
            cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)
        else:
            axs.plot(x_test[i, :max_idx, 0])
            sc = axs.scatter(np.linspace(0, max_idx - 1, max_idx),
                             x_test[i, :max_idx, 0], c=hm[:max_idx],
                             cmap='cool', vmin=0, vmax=1)
            # axs.set_axis_off()
            cbar = fig.colorbar(sc, ax=axs, shrink=0.95)

        # sc.set_clim(0,1)
        # sc.set_cmap('cool')

        # plt.title('Class {}, predicted as {}'.format(y_test[i], y_pred[i]))

        plt.show()
    else:

        y_pred_like = result[0]
        cam = result[1]

        y_pred_like_tv = result_tv[0]
        cam_tv = result_tv[1]

        masked = False
        if len(result) > 2 and False:
            mask = result[2]
            mask_tv = result_tv[2]
            masked = True
            print(mask.shape)
            print(np.where(x_test[0, :, 0] < -900)[0][0])
            plt.plot(mask[0, :])
            plt.show()

        y_pred = np.argmax(y_pred_like, axis=1)
        y_pred_tv = np.argmax(y_pred_like_tv, axis=1)

        print('correct:', y_test)
        print('predicted:', y_pred)
        print('predicted:', y_pred_like)

        model.summary()
        print(np.where(x_test[0, :, 0] < -900)[0][0])
        hm = make_gradcam_heatmap(np.expand_dims(x_test[0, ...], 0), model, 'conv1d_2', [
                                  'global_average_pooling1d', 'result'])
        plt.plot(cam[0, :], label='cam')
        plt.plot(hm, label='hm')
        plt.legend()
        plt.show()

        print(cam)
        print(cam.shape)

        cmin = 10
        cmax = -10
        for i in range(x_test.shape[0]):
            idx = np.where(x_test[i, :, 0] < -900)[0][0]
            cmin = np.min((cmin, np.min(cam[i, :idx])))
            cmax = np.max((cmax, np.max(cam[i, :idx])))

        cam = cam / (cmax - cmin) - cmin
        # print(cam)

        # plot_w_cam(x[3,...], cam[3,:], y_pred[1], y[1])

        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                              title='Confusion matrix, without normalization')

        cnf_matrix = confusion_matrix(y_tv, y_pred_tv)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                              title='Confusion matrix, without normalization')
        if plot_all:
            for i in range(len(y_test) // 2):
                fig, axs = plt.subplots(x_test.shape[2])
                # print(axs)
                # if x.shape[2] <= 1:
                #     axs = np.array(axs)
                # print(axs)
                max_idx = np.where(x_test[i, :, 0] < -900)[0][0]
                # plt.plot(x[:max_idx,0])
                if x_test.shape[2] > 1:
                    for j in range(x_test.shape[2]):
                        axs[j].plot(x_test[i, :max_idx, j])
                        sc = axs[j].scatter(np.linspace(0, max_idx - 1, max_idx),
                                            x_test[i, :max_idx, j],
                                            c=cam[i, :max_idx], cmap='cool',
                                            vmin=0, vmax=1)
                        # axs[j].set_axis_off()
                    cbar = fig.colorbar(
                        sc, ax=axs.ravel().tolist(), shrink=0.95)
                else:
                    axs.plot(x_test[i, :max_idx, 0])
                    sc = axs.scatter(np.linspace(0, max_idx - 1, max_idx),
                                     x_test[i, :max_idx,
                                            0], c=cam[i, :max_idx],
                                     cmap='cool', vmin=0, vmax=1)
                    # axs.set_axis_off()
                    cbar = fig.colorbar(sc, ax=axs, shrink=0.95)

                # sc.set_clim(0,1)
                # sc.set_cmap('cool')

                plt.title('Class {}, predicted as {}'.format(
                    y_test[i], y_pred[i]))

        plt.show()

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['0','1','2'],
    #                       title='Confusion matrix, without normalization')
    #
    # plt.figure()
    # plt.plot(x[:,:,0].T)
    #
    # plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('root')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
