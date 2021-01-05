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

IDX_PATH = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'


def fit_classifier(dp, classifier_name, output_directory, idx):

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
    enc.fit(np.concatenate((y_train), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    input_shape = x_train.shape[1:]

    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold = 1
    acc_per_fold = []
    loss_per_fold = []
    abs_err = []

    if 'coral' in classifier_name:
        y_train = y_train_orig

    cnf_matrix = np.zeros((3, 3))
    for train, test in kfold.split(x_train[:, 0], y_train[:, 0]):

        print(f'Fold number {fold} out of {num_folds}')

        classifier = create_classifier(classifier_name, input_shape,
                                       nb_classes, output_directory)

        if fold == 1:
            print(classifier.model.summary())

        class_weight = {0: 1, 1: 1.5, 2: 3}
        classifier.fit(x_train[train, ...], y_train[train, ...],
                       x_train[test, ...], y_train[test, ...],
                       class_weight=class_weight)

        scores = classifier.model.evaluate(x_train[test, ...],
                                           y_train[test, ...], verbose=0)

        probs = classifier.model(x_train[test, ...], training=False)

        if 'coral' in classifier_name:
            # print(probs)
            probs = coral.ordinal_softmax(probs)
            # print(probs)

            preds = np.argmax(probs, axis=1)

            acc = accuracy_score(y_train_orig[test], preds)

            print(
                f'Score for fold {fold}: {classifier.model.metrics_names[0]} of {scores[0]}; {classifier.model.metrics_names[1]} of {scores[1]}; Accuracy of {acc}')
            abs_err.append(scores[1])
            acc_per_fold.append(acc)
            loss_per_fold.append(scores[0])
        else:
            preds = np.argmax(probs, axis=1)
            print(
                f'Score for fold {fold}: {classifier.model.metrics_names[0]} of {scores[0]}; {classifier.model.metrics_names[1]} of {scores[1]}')
            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])

        cm_tmp = confusion_matrix(y_train_orig[test], preds)
        cnf_matrix = cnf_matrix + cm_tmp
        print(cm_tmp)
        fold += 1

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    if 'coral' in classifier_name:
        print(f'> Mean absolute error: {np.mean(abs_err)}')
    print('------------------------------------------------------------------------')

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
    if classifier_name == 'masked-resnet-mod':
        from classifiers import masked_resnet_mod
        return masked_resnet_mod.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose, nb_epochs=5000, batch_size=32, n_feature_maps=64, depth=4)
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
        return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=5000, bottleneck_size=32, use_residual=True)
    if classifier_name == 'masked-inception-mod':
        from classifiers import masked_inception_mod
        return masked_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=41, nb_epochs=2000, bottleneck_size=32, use_residual=True, lr=0.005)
    if classifier_name == 'x-inception':
        from classifiers import x_inception
        return x_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=32, kernel_size=41, nb_epochs=2000, bottleneck_size=16, use_residual=True, lr=0.005)
    if classifier_name == 'x-inception-coral':
        from classifiers import x_inception_coral
        return x_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=32, kernel_size=41, nb_epochs=2000, bottleneck_size=16, use_residual=True, lr=0.005)
    if classifier_name == 'coral-inception-mod':
        from classifiers import coral_inception_mod
        return coral_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=41, nb_epochs=2000, bottleneck_size=64, use_residual=True, lr=0.005)
    if classifier_name == 'xcm':
        from classifiers import xcm
        return xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose)
    if classifier_name == 'masked-xcm':
        from classifiers import masked_xcm
        return masked_xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51, 31, 11], decay=False)
    if classifier_name == 'masked-xcm-mod':
        from classifiers import masked_xcm_mod
        return masked_xcm_mod.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[128, 128, 64], depth=2, window=[41, 31, 21], decay=False, batch_size=32)
        # return masked_xcm_mod.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64, 128], depth=2, window=[51,31,21,11], decay=False)
    if classifier_name == 'masked-xcm-2d':
        from classifiers import masked_xcm_2d
        return masked_xcm_2d.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=2000, verbose=verbose, filters=128, depth=2, window=41, decay=False, batch_size=32)
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
        return net1d_mod.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[128, 128], depth=2, window=[51, 31], decay=False, batch_size=32)


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
    parser.add_argument('--gen_idx', type=str2bool, nargs='?', default=False)
    parser.add_argument('--train_idx', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--archive', default='VA')
    args = parser.parse_args()
    main(args)
