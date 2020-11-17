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
    x = np.delete(x, train_idx, axis=0)
    y = np.delete(y, train_idx)

    print(x.shape)
    print(y.shape)

    model_path = os.path.join(args.root, 'best_model.hdf5')
    model = keras.models.load_model(model_path)
    y_pred_like, cam = model.predict(x)

    y_pred = np.argmax(y_pred_like, axis=1)

    print('correct:', y)
    print('predicted:', y_pred)
    print('predicted:', y_pred_like)

    print(cam)
    print(cam.shape)
    cam = cam / (np.max(cam) - np.min(cam)) - np.min(cam)

    # plot_w_cam(x[3,...], cam[3,:], y_pred[1], y[1])

    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)

    for i in range(len(y)//2):
        fig, axs = plt.subplots(2)
        max_idx = np.where(x[i, :, 0] < -900)[0][0]
        # plt.plot(x[:max_idx,0])
        axs[0].plot(x[i,:max_idx,0])
        axs[1].plot(x[i,:max_idx,1])
        sc = axs[0].scatter(np.linspace(0, max_idx-1, max_idx),x[i,:max_idx,0], c=cam[i,:max_idx], cmap='cool', vmin=0, vmax=1)
        axs[0].set_axis_off()
        # sc.set_clim(0,1)
        # sc.set_cmap('cool')
        # fig.colorbar(sc)
        sc = axs[1].scatter(np.linspace(0, max_idx-1, max_idx),x[i,:max_idx,1], c=cam[i,:max_idx], cmap='cool', vmin=0, vmax=1)
        axs[0].set_axis_off()


        cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)
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
    args = parser.parse_args()
    main(args)
