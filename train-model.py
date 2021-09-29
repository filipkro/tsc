from utils.utils import create_directory
from utils.gen_dataset_idx import gen_train_val_test, gen_rnd
import os
import numpy as np
import sys
import sklearn
from sklearn.model_selection import KFold
import utils
from argparse import ArgumentParser, ArgumentTypeError
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import coral_ordinal as coral
import pandas as pd
from keras.utils import to_categorical
from utils.custom_train_loop import train_loop

IDX_PATH = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'


def idx_same_subject(meta, subject):
    # subj = meta[idx,1]
    indices = np.where(meta[:, 1] == subject)[0]
    return indices


def read_meta_data(info_file):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    meta_data = np.array(meta_data.values[first_data:, :4], dtype=int)
    return meta_data


def get_POE_field(info_file):
    '''get POE evaluated in dataset'''
    data = pd.read_csv(info_file, delimiter=',')
    poe = data.values[np.where(data.values[:, 0] == 'Action:')[0][0], 1]
    return poe.split('_')[-1]


def fit_classifier(dp, classifier_name, output_directory, idx):

    dataset = np.load(dp)
    # idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx2.npz'
    indices = np.load(args.idx)
    train_idx = indices['train_idx']
    val_idx = indices['val_idx']

    x = dataset['mts']
    y = dataset['labels']
    nb_classes = len(np.unique(y))

    y_oh = to_categorical(y)
    if 'coral' in classifier_name or 'focal' in classifier_name:
        y_oh = y

    if len(x.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x = x.reshape((x.shape[0], x.shape[1], 1))

    x_train = x[train_idx, ...]
    x_val = x[val_idx, ...]
    y_train = y_oh[train_idx, ...]
    y_val = y_oh[val_idx, ...]

    input_shape = x.shape[1:]

    fold = 1
    acc_per_fold = []
    loss_per_fold = []
    abs_err = []

    cnf_matrix = np.zeros((nb_classes, nb_classes))

    class_weight = None
    if 'coral' in classifier_name:
        class_weight = [1, 1]
        classifier = create_classifier(classifier_name, input_shape,
                                       nb_classes, output_directory,
                                       class_weight=class_weight)
        print(f'class weight: {class_weight}')
    else:
        #class_weight = {0:1, 1:3, 2:5}
        classifier = create_classifier(classifier_name, input_shape,
                                       nb_classes, output_directory)
        print(f'class weight: {class_weight}')

    if 'conf' in classifier_name:
        print(f"U: \n{classifier.model.loss.get_config()['U']}")

    ifile = open(os.path.join(output_directory, 'class_weights.txt'), 'w')
    ifile.write(f'model type: {classifier_name}\n')
    U = classifier.model.loss.get_config()['U'] if 'conf' in classifier_name else class_weight
    ifile.write('Training weight:\n')
    ifile.write(str(U))
    ifile.close()
    print('file done')

    print(classifier.model.summary())

    classifier.fit(x_train, y_train, x_val, y_val)

    scores = classifier.model.evaluate(x_val, y_val, verbose=0)

    probs = classifier.model(x_val, training=False)

    if 'coral' in classifier_name:
        # print(probs)
        probs = coral.ordinal_softmax(probs)
        # print(probs)

        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y[val_idx], preds)

        print(f'Score: {classifier.model.metrics_names[0]} of {scores[0]}; {classifier.model.metrics_names[1]} of {scores[1]}; Accuracy of {acc}')
        abs_err.append(scores[1])
        acc_per_fold.append(acc)
        loss_per_fold.append(scores[0])
    else:
        preds = np.argmax(probs, axis=1)
        print(f'Score for fold {fold}: {classifier.model.metrics_names[0]} of {scores[0]}; {classifier.model.metrics_names[1]} of {scores[1]}')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])

    cm_tmp = confusion_matrix(y[val_idx], preds, labels=range(nb_classes))
    cnf_matrix = cnf_matrix + cm_tmp
    print(cm_tmp)

    classifier.model.save(os.path.join(output_directory, 'model.hdf5'))



    print('Confusion matrix all folds:')
    print(cnf_matrix)

    ifile = open(os.path.join(output_directory, 'x-val.txt'), 'w')
    ifile.write('Average scores for all folds: \n')
    ifile.write(
        f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)}) \n')
    ifile.write(f'> Loss: {np.mean(loss_per_fold)} \n \n')
    ifile.write('Confusion matrix all folds: \n')
    ifile.write(str(cnf_matrix))
    # ifile.write(classifier.model.summary())
    ifile.close()
    print(acc_per_fold)

    # assert False


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=2, class_weight=None):
    if classifier_name == 'inception-conf':
        from classifiers import inception_conf
        return inception_conf.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=False, lr=0.005)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=False, lr=0.005)
    if classifier_name == 'inception-coral':
        from classifiers import inception_coral
        return inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'bayes-inception-coral':
        from classifiers import bayes_inception_coral
        return bayes_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'simple-bays-coral':
        from classifiers import simple_bays_coral
        return simple_bays_coral.Classifier_SIMPLE_BAYS_CORAL(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, lr=0.005)
    if classifier_name == 'inception-focal':
        from classifiers import inception_focal
        return inception_focal.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'x-inception':
        from classifiers import x_inception
        return x_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception':
        from classifiers import xx_inception
        return xx_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-focal':
        from classifiers import xx_inception_focal
        return xx_inception_focal.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=600, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-conf':
        from classifiers import xx_inception_conf
        return xx_inception_conf.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-coral':
        from classifiers import xx_inception_coral
        return xx_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=True, class_weight=class_weight)


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

    print(tf.test.is_gpu_available())
    classifier_name = args.classifier.replace('_', '-')
    # rate = args.dataset.split('-')[-1].split('.')[0]
    lit = os.path.basename(args.dataset).split('_')[1].split('.')[0]
    lit = lit + '-len100' if 'len100' in args.dataset else lit
    root_dir = args.root + '/' + classifier_name + '/' + lit + '/'
    if args.itr == '':
        itr = 0
        for prev in os.listdir(root_dir):
            if lit in prev:
                prev_itr = int(prev.split('_')[2])
                itr = np.max((itr, prev_itr + 1))

        sitr = '_itr_' + str(itr)
    else:
        sitr = args.itr
    if sitr == '_itr_0':
        sitr = ''

    print(sitr)

    output_directory = root_dir + lit + sitr + '/'

    print(output_directory)

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', args.dataset, classifier_name, sitr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)
        fit_classifier(args.dataset, classifier_name,
                       output_directory, args.idx)
        print('DONE')

        create_directory(output_directory + '/DONE')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('dataset')
    parser.add_argument('classifier')
    parser.add_argument(
        '--idx', default='/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz')
    parser.add_argument('--itr', default='')
    parser.add_argument('--merge_class', type=str2bool,
                        nargs='?', default=False)
    args = parser.parse_args()
    main(args)
