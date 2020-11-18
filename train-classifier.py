from utils.utils import create_directory
import os
import numpy as np
import sys
import sklearn
import utils
from argparse import ArgumentParser


def fit_classifier(dp, trp, tep, classifier_name, output_directory):

    dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']

    print('x shape::', x.shape)

    train_size = int(np.round(0.85 * len(y)))

    train_idx = np.load(trp) if trp != '' else np.random.choice(len(y),
                                                                train_size,
                                                                replace=False)
    train_idx = np.random.choice(len(y), train_size, replace=False)
    x_train = x[train_idx, ...]
    x = np.delete(x, train_idx, axis=0)
    y_train = y[train_idx]
    y = np.delete(y, train_idx)
    test_size = int(np.round(0.5 * len(y)))
    test_idx = np.load(tep) if tep != '' else np.random.choice(len(y),
                                                               test_size,
                                                               replace=False)
    y_test = np.delete(y, test_idx)
    x_test = np.delete(x, test_idx, axis=0)

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    print(nb_classes)

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(
        classifier_name, input_shape, nb_classes, output_directory)
    print('created classifier')
    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=2):
    if classifier_name == 'fcn-simple':
        from classifiers import small_fcn
        return small_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'masked-fcn':
        from classifiers import masked_fcn
        return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=verbose)
    if classifier_name == 'masked-fcn-big':
        from classifiers import masked_fcn_big
        return masked_fcn_big.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'masked-resnet':
        from classifiers import masked_resnet
        return masked_resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'masked-inception':
        from classifiers import masked_inception
        return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=4, nb_filters=8, kernel_size=15, nb_epochs=500, bottleneck_size=8)
    if classifier_name == 'inception-simple':
        from classifiers import inception_simple
        return inception_simple.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, kernel_size=31)


def main(args):
    classifier_name = args.classifier.replace('_', '-')
    # rate = args.dataset.split('-')[-1].split('.')[0]
    lit = os.path.basename(args.dataset).split('_')[1].split('.')[0]

    root_dir = args.root + '/' + classifier_name + '/' + lit + '/'
    itr = 0
    for prev in os.listdir(root_dir):
        if lit in prev:
            prev_itr = int(prev.split('_')[2])
            itr = np.max((itr, prev_itr + 1))

    sitr = '_itr_' + str(itr)
    if sitr == '_itr_0':
        sitr = ''

    print(sitr)

    output_directory = root_dir + lit + sitr + '/'

    print(output_directory)

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ',  args.archive, args.dataset, classifier_name, sitr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)
        fit_classifier(args.dataset, args.train_idx, args.test_idx,
                       classifier_name, output_directory)
        print('DONE')

        create_directory(output_directory + '/DONE')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('dataset')
    parser.add_argument('classifier')
    # parser.add_argument('itr')
    parser.add_argument('--train_idx', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--archive', default='VA')
    args = parser.parse_args()
    main(args)
