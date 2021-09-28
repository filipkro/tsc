import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn import discriminant_analysis
from sklearn import metrics
import pandas as pd
import os
from sklearn.svm import SVC

def main(args):
    data = np.load(args.data)
    x = data['mts']
    y = data['labels']
    print(data.files)
    print(x.shape)
    print(y.shape)
    y = np.squeeze(y)
    print(y.shape)

    print(np.unique(y))



    for fold in range(1,6):
        idxs = np.load(os.path.join(args.idx_path, f'idx_{fold}.npz'))
        print(idxs.files)
        train_idx = idxs['train_idx']
        val_idx = idxs['val_idx']
        print(train_idx.shape)
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[val_idx]
        y_test = y[val_idx]

        new_x = np.reshape(x_train, (len(train_idx), -1), order='F')
        print(new_x.shape)

        new_xtest = np.reshape(x_test, (len(val_idx), -1), order='F')
        print(new_xtest.shape)

        lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
        X_trafo_sk = lda.fit_transform(new_x,y_train)
        print(X_trafo_sk.shape)
        # pd.DataFrame(np.hstack((X_trafo_sk, y))).plot.scatter(x=0, y=1, c=2, colormap='viridis')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        for i in range(3):
            idx = np.where(y_train==i)[0]
            label = f'class {i}'
            plt.scatter(X_trafo_sk[idx,0], X_trafo_sk[idx,1], label=label)
            # ax.scatter(X_trafo_sk[idx,0], X_trafo_sk[idx,1], X_trafo_sk[idx,2], label=label)

        preds = lda.transform(new_xtest)
        pred_labels = lda.predict(new_xtest)

        plt.figure()
        for i in range(3):
            idx = np.where(y_test==i)[0]
            label = f'class {i}'
            plt.scatter(preds[idx,0], preds[idx,1], label=label)

        cnf = metrics.confusion_matrix(y_test, pred_labels)
        print(cnf)
        print(metrics.accuracy_score(y_test, pred_labels))

        sv_machine = SVC()
        sv_machine.fit(new_x, y_train)
        pred_labels = sv_machine.predict(new_xtest)
        cnf = metrics.confusion_matrix(y_test, pred_labels)
        print(cnf)
        print(metrics.accuracy_score(y_test, pred_labels))


        plt.legend()
        plt.show()





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('idx_path')
    args = parser.parse_args()
    main(args)
