import numpy as np
from argparse import ArgumentParser
from tsfresh import extract_relevant_features
import pandas as pd
from sklearn import discriminant_analysis
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def idx_same_subject(meta, subject):
    # subj = meta[idx,1]
    indices = np.where(meta[:, 1] == subject)[0]
    return indices

def read_meta_data(info_file):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    meta_data = np.array(meta_data.values[first_data:, :4], dtype=int)
    return meta_data

def main(args):
    data = np.load(args.data)
    X_np = data['mts']
    Y_np = data['labels']

    # idx_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-{poe}.npz'
    indices = np.load('/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz')
    info_file = args.data.split('.npz')[0] + '-info.txt'
    meta_data = read_meta_data(info_file)

    # print(indices)

    x = np.reshape(X_np, (-1,4))
    print(x.shape)
    print(np.sum(x[:100,:] - X_np[0,...]))

    xx = np.zeros((x.shape[0], x.shape[1]+2))
    yy = np.zeros((Y_np.shape[0], 2))
    print(xx.shape)
    xx[:,2:] = x

    yy[:,0] = range(1, Y_np.shape[0]+1)
    yy[:,1] = Y_np[:,0]
    id = -1
    time = 0
    for i in range(xx.shape[0]):
        if i % 100 == 0:
            id += 1
            time = 0

        xx[i,:2] = [id, time]
        time +=1

    timeseries = pd.DataFrame(xx, columns=["id", "time", "feat1", "feat2","feat3","feat4"])
    y = pd.Series(Y_np[:,0])

    print(y)


    print(np.array(timeseries.values))
    # assert False

    # features = extract_relevant_features(timeseries, y,
    #                                      column_id='id', column_sort='time')
    # print(features)
    # print(features[0,:])

    num_folds = 5
    kfold = KFold(n_splits=num_folds, random_state=1)
    # X = np.array(features.values)
    X = np.load('/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/feats.npy')

    # np.save('/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/feats.npy', X)
    # Y =

    for train, val in kfold.split(indices['train_subj']):
        # print(f'Fold number {fold} out of {num_folds}')
        train_idx = [idx_same_subject(meta_data, subj) for subj in train]
        val_idx = [idx_same_subject(meta_data, subj) for subj in val]
        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

        x_train = X[train_idx, :]
        y_train = Y_np[train_idx, 0]
        x_val = X[val_idx, :]
        y_val = Y_np[val_idx, 0]

        lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
        sv_machine = SVC()
        sv_machine.fit(x_train, y_train)
        transform = lda.fit_transform(x_train,y_train)
        for i in range(3):
            idx = np.where(y_train==i)[0]
            label = f'class {i}'
            plt.scatter(transform[idx,0], transform[idx,1], label=label)

        preds = lda.transform(x_val)
        pred_labels = lda.predict(x_val)
        plt.figure()
        for i in range(3):
            idx = np.where(y_val==i)[0]
            label = f'class {i}'
            plt.scatter(preds[idx,0], preds[idx,1], label=label)

        cnf = metrics.confusion_matrix(y_val, pred_labels)
        print(cnf)
        print(metrics.accuracy_score(y_val, pred_labels))

        pred_labels = sv_machine.predict(x_val)

        cnf = metrics.confusion_matrix(y_val, pred_labels)
        print(cnf)
        print(metrics.accuracy_score(y_val, pred_labels))

        print(np.shape(y_val))

        plt.legend()
        plt.show()


    # print(y)
    #
    # print(Y_np.shape)
    # print(yy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    main(args)
