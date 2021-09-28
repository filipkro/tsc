import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def main(args):

    lit = 'Olga-Tokarczuk'
    poe = 'femval'

    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info-fix.txt'
    dataset = np.load(dp)

    idx_path = f'/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-{poe}.npz'

    ind_t = np.load(idx_path)
    test_idx = ind_t['test_idx'].astype(np.int)
    x = dataset['mts'][test_idx]
    y = dataset['labels'][test_idx]

    model = keras.models.load_model(args.model, custom_objects={
                                    'CoralOrdinal': coral.CoralOrdinal,
                                    'OrdinalCrossEntropy':
                                    coral.OrdinalCrossEntropy,
                                    'MeanAbsoluteErrorLabels':
                                    coral.MeanAbsoluteErrorLabels,
                                    'ConfusionCrossEntropy':
                                    ConfusionCrossEntropy})

    mc = 10
    preds = np.zeros((mc, len(test_idx), 3))
    preds2 = np.zeros((mc, len(test_idx), 3))

    for i in range(mc):
        y_sample = model(x, training=True)
        y_sample = coral.ordinal_softmax(y_sample).numpy()
        preds[i, ...] = y_sample
        for j, pred in enumerate(y_sample):
            preds2[i, j, np.argmax(pred)] = 1

    # print(np.mean(preds, axis=0))
    # print('\n')
    # print(np.mean(preds2, axis=0))

    preds = np.mean(preds, axis=0)
    preds2 = np.mean(preds2, axis=0)

    y1 = np.argmax(preds, axis=1)
    y2 = np.argmax(preds2, axis=1)

    # print(y1.shape)
    # print(y2.shape)
    # print(preds - preds2)

    print(accuracy_score(y, y1))
    print(accuracy_score(y, y2))

    y3 = model(x)
    y3 = coral.ordinal_softmax(y3).numpy()
    print(accuracy_score(y, np.argmax(y3, axis=1)))





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    main(args)
