from utils.utils import create_directory
import os
import numpy as np
import sys
import sklearn
import utils
from argparse import ArgumentParser

def get_data(data_path, test_path=''):
    # if data_path.split('.')[-1] == '.npz':
    ind = np.load(data_path)
    test_idx = ind['test_idx']
    train_idx = ind['train_idx']
    val_idx = ind['val_idx']

    return train_idx, test_idx, val_idx



def fit_classifier(dp, trp, tep, classifier_name, output_directory):

    dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']
    print(trp)
    print('x shape::', x.shape)
    if trp.split('.')[-1] == 'npz':
        ind = np.load(trp)
        test_idx = ind['test_idx'].astype(np.int)
        train_idx = ind['train_idx'].astype(np.int)
        val_idx = ind['val_idx'].astype(np.int)
        x_train = x[train_idx, ...]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
    else:
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
        y_val = np.delete(y, test_idx)
        x_val = np.delete(x, test_idx, axis=0)

    nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))
    print(nb_classes)
    print(x_train.shape)
    print(y_train.shape)



    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_val), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()




    # save orignal y because later we will use binary
    y_true = np.argmax(y_val, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))

    # print(x_train.shape)
    # if classifier_name == 'xcm':
    #     x_train = np.expand_dims(x_train, axis=-1)
    #     print(x_train.shape)

    input_shape = x_train.shape[1:]
    classifier = create_classifier(
        classifier_name, input_shape, nb_classes, output_directory)
    print('created classifier')
    print(x_train.shape)
    print(y_train.shape)
    print(classifier.model.summary())
    # assert False
    classifier.fit(x_train, y_train, x_val, y_val, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=2):
    if classifier_name == 'masked-fcn':
        from classifiers import masked_fcn
        return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=30000, kernel_size=32, filters=64, batch_size=32, depth=2)
        #return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=20000)
        #return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=20000, kernel_size=21, filters=32, batch_size=128, depth=3)
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
        return masked_resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose, nb_epochs=20000)
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
        #return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=4, nb_filters=16, kernel_size=21, nb_epochs=40000, bottleneck_size=8)
        return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=64, kernel_size=15, nb_epochs=40000, bottleneck_size=16, use_residual=False)
        #return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=6, nb_filters=32, kernel_size=41, nb_epochs=60000, bottleneck_size=32)
        #return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=4, nb_filters=8, kernel_size=15, nb_epochs=10000, bottleneck_size=8)
        #return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=6, nb_filters=32, kernel_size=41, nb_epochs=2, bottleneck_size=32)
    if classifier_name == 'xcm':
        from classifiers import xcm
        return xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=20000, verbose=verbose)
    if classifier_name == 'masked-xcm':
        from classifiers import masked_xcm
        #return masked_xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=20000, verbose=verbose)
        return masked_xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=20000, verbose=verbose, filters=64, depth=2)


def main(args):
    classifier_name = args.classifier.replace('_', '-')
    # rate = args.dataset.split('-')[-1].split('.')[0]
    lit = os.path.basename(args.dataset).split('_')[1].split('.')[0]

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
        fit_classifier(args.dataset, args.train_idx, args.test_idx,
                       classifier_name, output_directory)
        print('DONE')

        create_directory(output_directory + '/DONE')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('dataset')
    parser.add_argument('classifier')
    parser.add_argument('--itr', default='')
    parser.add_argument('--train_idx', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--archive', default='VA')
    args = parser.parse_args()
    main(args)
