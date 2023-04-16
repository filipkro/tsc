import numpy as np
from argparse import ArgumentParser
from tsfresh import extract_relevant_features
import pandas as pd
from sklearn import discriminant_analysis
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def get_POE_field(info_file):
    data = pd.read_csv(info_file, delimiter=',')
    poe = data.values[np.where(data.values[:, 0] == 'Action:')[0][0], 1]
    return poe.split('_')[-1]

def main(args):
    data = np.load(args.data)
    X_np = data['mts']
    Y_np = data['labels']

    info_file = args.data.split('.npz')[0] + '-info.txt'
    poe = get_POE_field(info_file)

    # print(X_np.shape)
    # assert False
    nbr_feats = X_np.shape[-1]
    x = np.reshape(X_np, (-1,nbr_feats))
    # print(x.shape)
    # print(np.sum(x[:100,:] - X_np[0,...]))

    xx = np.zeros((x.shape[0], x.shape[1]+2))
    yy = np.zeros((Y_np.shape[0], 2))
    # print(xx.shape)
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

    cols = ["id", "time"]
    # print(x.shape)
    # assert False
    for i in range(x.shape[1]):
        cols.append(f'feat{i+1}')

    # print(cols)
    # assert False
    timeseries = pd.DataFrame(xx, columns=cols)
    y = pd.Series(Y_np[:,0])

    features = extract_relevant_features(timeseries, y,
                                         column_id='id', column_sort='time')

    save_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/feats-{poe}.npy'
    print(f'features: {features}')
    np.save(save_path, np.array(features.values))

    print('done')
    # np.array(features.values)
    #
    # print(save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    main(args)
