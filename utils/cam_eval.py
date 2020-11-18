import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import itertools

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
    plt.plot(x[:max_idx,0])
    sc = plt.scatter(np.linspace(0, max_idx-1, max_idx),x[:max_idx,0], c=cam[:max_idx])
    plt.colorbar(sc)
    plt.show()



def main(args):
    dataset = np.load(args.dataset)
    train_idx = np.load(args.train)
    x = dataset['mts']
    y = dataset['labels']
    x_tv = np.delete(x, train_idx, axis=0)
    y_tv = np.delete(y, train_idx)

    if args.test_idx != '':
        test_idx = np.load(args.test_idx)
        x_test = x_tv[test_idx, ...]
        y_test = y_tv[test_idx]


    x = x_test
    y = y_test
    print(x.shape)
    print(y.shape)
    plot_all = True

    model_path = os.path.join(args.root, 'best_model.hdf5')
    model = keras.models.load_model(model_path)
    y_pred_like, cam = model.predict(x)
    y_pred_like_tv, cam_tv = model.predict(x_tv)

    y_pred = np.argmax(y_pred_like, axis=1)
    y_pred_tv = np.argmax(y_pred_like_tv, axis=1)

    print('correct:', y)
    print('predicted:', y_pred)
    print('predicted:', y_pred_like)


    print(np.where(x[0, :, 0] < -900)[0][0])
    plt.plot(cam[0,:])
    plt.show()

    print(cam)
    print(cam.shape)

    cmin = 10
    cmax = -10
    for i in range(x.shape[0]):
        idx = np.where(x[i, :, 0] < -900)[0][0]
        cmin = np.min((cmin, np.min(cam[i,:idx])))
        cmax = np.max((cmax, np.max(cam[i,:idx])))

    cam = cam / (cmax - cmin) - cmin
    # print(cam)

    # plot_w_cam(x[3,...], cam[3,:], y_pred[1], y[1])

    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=['0','1','2'],
                          title='Confusion matrix, without normalization')

    cnf_matrix = confusion_matrix(y_tv, y_pred_tv)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=['0','1','2'],
                          title='Confusion matrix, without normalization')
    if plot_all:
        for i in range(len(y)//2):
            fig, axs = plt.subplots(x.shape[2])
            # print(axs)
            # if x.shape[2] <= 1:
            #     axs = np.array(axs)
            # print(axs)
            max_idx = np.where(x[i, :, 0] < -900)[0][0]
            # plt.plot(x[:max_idx,0])
            if x.shape[2] > 1:
                for j in range(x.shape[2]):
                    axs[j].plot(x[i,:max_idx,j])
                    sc = axs[j].scatter(np.linspace(0, max_idx-1, max_idx),x[i,:max_idx,j], c=cam[i,:max_idx], cmap='cool', vmin=0, vmax=1)
                    # axs[j].set_axis_off()
                cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)
            else:
                axs.plot(x[i,:max_idx,0])
                sc = axs.scatter(np.linspace(0, max_idx-1, max_idx),x[i,:max_idx,0], c=cam[i,:max_idx], cmap='cool', vmin=0, vmax=1)
                # axs.set_axis_off()
                cbar = fig.colorbar(sc, ax=axs, shrink=0.95)


            # sc.set_clim(0,1)
            # sc.set_cmap('cool')

            plt.title('Class {}, predicted as {}'.format(y[i], y_pred[i]))

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
    parser.add_argument('dataset')
    parser.add_argument('train')
    parser.add_argument('root')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
