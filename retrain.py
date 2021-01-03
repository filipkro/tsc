from utils.utils import create_directory
from utils.gen_dataset_idx import gen_train_val, gen_tv_from_test
import os
import numpy as np
import sys
import sklearn
from sklearn.model_selection import KFold
import utils
from argparse import ArgumentParser, ArgumentTypeError
import tensorflow as tf
from sklearn.metrics import confusion_matrix

IDX_PATH = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'


def fit_classifier(dp, trp, tep, classifier_name, output_directory, idx):

    dataset = np.load(dp)
    indices = np.load(idx)

    x = dataset['mts']
    y = dataset['labels']
    merge_class = False
    if merge_class:
        idx = np.where(y == 1)[0]
        y[idx] = 0
        idx = np.where(y == 2)[0]
        y[idx] = 1

    if x.shape[1] > 300:
        # long ts - suggests sequences not split
        x_train = x[indices['train_subj'], ...]
        y_train = y[indices['train_subj']]
        y_test = y[indices['test_subj']]
    else:
        x_train = x[indices['train_idx'], ...]
        y_train = y[indices['train_idx']]
        y_test = y[indices['test_idx']]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    print(nb_classes)
    y_train_orig = y_train.copy()
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y), axis=0).reshape(-1, 1))
    y_one_hot = enc.transform(y.reshape(-1, 1)).toarray()
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    input_shape = x_train.shape[1:]
    info_file = dp.split('.npz')[0] + '-info.txt'
    train_idx, val_idx, train_subj, val_subj = gen_tv_from_test(info_file, 0.85, indices['test_subj'])

    np.savez(os.path.join(output_directory, 'indices.npz'), train_idx=train_idx,
            val_idx=val_idx, train_subj=train_subj, val_subj=val_subj)

    x_val = x[val_idx, ...]
    x_train = x[train_idx, ...]
    y_val = y_one_hot[val_idx, ...]
    y_train = y_one_hot[train_idx, ...]

    classifier = create_classifier(classifier_name, input_shape,
                                   nb_classes, output_directory)

    print(classifier.model.summary())

    class_weight = {0: 1, 1: 1.2, 2: 1.5}
    classifier.fit(x_train, y_train, x_val, y_val, class_weight=class_weight)
    scores = classifier.model.evaluate(x_val, y_val, verbose=0)
    preds = classifier.model(x_val, training=False)
    print('correct: {}'.format(np.argmax(y_val, axis=1)))
    print('prediccted: {}'.format(np.argmax(preds, axis=1)))
    # cnf_matrix = confusion_matrix(, )

    print('Loss: {} \nAccuracy: {}'.format(scores[0], scores[1]))
    #print(cnf_matrix)

    # assert False


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=2):
    if classifier_name == 'masked-fcn':
        from classifiers import masked_fcn
        return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=5000, kernel_size=101, filters=8, batch_size=16, depth=3)
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
        return masked_resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose, nb_epochs=5000)
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
        return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=64, kernel_size=31, nb_epochs=5000, bottleneck_size=32, use_residual=False)
    if classifier_name == 'masked-inception-mod':
        from classifiers import masked_inception_mod
        return masked_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=3, nb_filters=128, kernel_size=31, nb_epochs=1500, bottleneck_size=64, use_residual=False, batch_size=16)
        return masked_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=64, kernel_size=31, nb_epochs=5000, bottleneck_size=128, use_residual=False, batch_size=32)
    if classifier_name == 'xcm':
        from classifiers import xcm
        return xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose)
    if classifier_name == 'masked-xcm':
        from classifiers import masked_xcm
        return masked_xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51, 31, 11], decay=False)
    if classifier_name == 'masked-xcm-mod':
        from classifiers import masked_xcm_mod
        #return masked_xcm_mod.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=32, depth=3, window=301, decay=False)
        return masked_xcm_mod.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=64, depth=2, window=41, decay=False)
    if classifier_name == 'net1d':
        from classifiers import net1d
        return net1d.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51, 31, 11], decay=False)
    if classifier_name == 'net1d-v2':
        from classifiers import net1d_v2
        return net1d_v2.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=32, depth=2, window=41, decay=False)
    if classifier_name == 'cnn2d':
        from classifiers import cnn2d
        return cnn2d.Classifier_CNN2D(output_directory, input_shape, nb_classes, nb_epochs=8000, verbose=verbose, filters=4, depth=2, decay=False, window=121, batch_size=32)
    if classifier_name == 'net1d-mod':
        from classifiers import net1d_mod
        return net1d_mod.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51, 31, 11], decay=False)


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
                       classifier_name, output_directory, args.idx)
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
    parser.add_argument('--gen_idx', type=str2bool, nargs='?', default=False)
    parser.add_argument('--train_idx', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--archive', default='VA')
    args = parser.parse_args()
    main(args)
