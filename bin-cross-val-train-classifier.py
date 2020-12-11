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


    idx = np.where(y_train == 1)[0]
    y01 = y_train.copy()
    y12 = y_train.copy()
    y01[idx] = 0
    y12[idx] = 2
    nb_classes = 2

    y01_orig = y01.copy()
    y12_orig = y12.copy()

    enc01 = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc12 = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc01.fit(np.concatenate((y01), axis=0).reshape(-1, 1))
    y01 = enc01.transform(y01.reshape(-1, 1)).toarray()
    enc12.fit(np.concatenate((y12), axis=0).reshape(-1, 1))
    y12 = enc12.transform(y12.reshape(-1, 1)).toarray()
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    input_shape = x_train.shape[1:]

    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold = 1
    acc_per_fold01 = []
    loss_per_fold01 = []

    acc_per_fold12 = []
    loss_per_fold12 = []


    cnf_matrix01 = np.zeros((2,2))
    cnf_matrix12 = np.zeros((2,2))
    for train, test in kfold.split(x_train[:, 0], y_train[:, 0]):

        print(f'Fold number {fold} out of {num_folds}')

        classifier = create_classifier(classifier_name, input_shape,
                                       nb_classes, output_directory)

        if fold == 1:
            print(classifier.model.summary())

        class_weight = {0:1.0, 1:50.0}
        classifier.fit(x_train[train, ...], y01[train, ...],
                       x_train[test, ...], y01[test, ...],
                       class_weight=class_weight)

        scores01 = classifier.model.evaluate(x_train[test, ...],
                                           y01[test, ...], verbose=0)

        preds01 = classifier.model(x_train[test, ...], training=False)
        preds01 = np.argmax(preds01, axis=1)
        idx = np.where(preds01 == 1)[0]
        preds01[idx] = 2
        tmp01 = confusion_matrix(y01_orig[test], preds01)
        cnf_matrix01 = cnf_matrix01 + tmp01
        classifier = create_classifier(classifier_name, input_shape,
                                       nb_classes, output_directory)

        class_weight = {0:15.0, 1:1.0}
        classifier.fit(x_train[train, ...], y12[train, ...],
                      x_train[test, ...], y12[test, ...],
                      class_weight=class_weight)

        scores12 = classifier.model.evaluate(x_train[test, ...],
                                          y12[test, ...], verbose=0)

        preds12 = classifier.model(x_train[test, ...], training=False)
        preds12 = np.argmax(preds12, axis=1)
        idx = np.where(preds12 == 1)[0]
        preds12[idx] = 2
        tmp12 = confusion_matrix(y12_orig[test], preds12)
        cnf_matrix12 = cnf_matrix12 + tmp12

        print(f'Score for fold {fold} with 0 and 1 grouped together: {classifier.model.metrics_names[0]} of {scores01[0]}; {classifier.model.metrics_names[1]} of {scores01[1]}')

        print(f'Score for fold {fold} with 1 and 1 grouped together: {classifier.model.metrics_names[0]} of {scores12[0]}; {classifier.model.metrics_names[1]} of {scores12[1]}')

        print('Confusion matrix with 0 and 1 grouped together:')
        print(tmp01)

        print('Confusion matrix with 1 and 2 grouped together:')
        print(tmp12)


        acc_per_fold01.append(scores01[1])
        loss_per_fold01.append(scores01[0])

        acc_per_fold12.append(scores12[1])
        loss_per_fold12.append(scores12[0])

        fold += 1

    print('------------------------------------------------------------------------')
    print('Average scores for all folds with 0 and 1 grouped together:')
    print(f'> Accuracy: {np.mean(acc_per_fold01)} (+- {np.std(acc_per_fold01)})')
    print(f'> Loss: {np.mean(loss_per_fold01)}')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds with 1 and 2 grouped together:')
    print(f'> Accuracy: {np.mean(acc_per_fold12)} (+- {np.std(acc_per_fold12)})')
    print(f'> Loss: {np.mean(loss_per_fold12)}')
    print('------------------------------------------------------------------------')
    print('Confusion matrix for all folds with 0 and 1 grouped together:')
    print(cnf_matrix01)
    print('Confusion matrix for all folds with 1 and 2 grouped together:')
    print(cnf_matrix12)

    print(acc_per_fold01)
    print(acc_per_fold12)

    ifile = open(os.path.join(output_directory, 'x-val.txt'), 'w')
    ifile.write('Average scores for all folds with 0 and 1 grouped together: \n')
    ifile.write(f'> Accuracy: {np.mean(acc_per_fold01)} (+- {np.std(acc_per_fold01)}) \n')
    ifile.write(f'> Loss: {np.mean(loss_per_fold01)} \n \n')
    ifile.write('Average scores for all folds with 1 and 2 grouped together: \n')
    ifile.write(f'> Accuracy: {np.mean(acc_per_fold12)} (+- {np.std(acc_per_fold12)}) \n')
    ifile.write(f'> Loss: {np.mean(loss_per_fold12)} \n \n')
    ifile.write('Confusion matrix for all folds with 0 and 1 grouped together: \n')
    ifile.write(cnf_matrix01)
    ifile.write('\nConfusion matrix for all folds with 1 and 2 grouped together: \n')
    ifile.write(cnf_matrix12)
    # ifile.write(classifier.model.summary())
    ifile.close()


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
        return masked_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=3, nb_filters=64, kernel_size=51, nb_epochs=5000, bottleneck_size=32, use_residual=True)
    if classifier_name == 'xcm':
        from classifiers import xcm
        return xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose)
    if classifier_name == 'masked-xcm':
        from classifiers import masked_xcm
        return masked_xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51,31,11], decay=False)
    if classifier_name == 'masked-xcm-mod':
        from classifiers import masked_xcm_mod
        return masked_xcm_mod.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51,31,11], decay=False)
    if classifier_name == 'net1d':
        from classifiers import net1d
        return net1d.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16,32,64], depth=2, window=[51,31,11], decay=False)
    if classifier_name == 'net1d-v2':
        from classifiers import net1d_v2
        return net1d_v2.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=32, depth=2, window=41, decay=False)
    if classifier_name == 'cnn2d':
        from classifiers import cnn2d
        return cnn2d.Classifier_CNN2D(output_directory, input_shape, nb_classes, nb_epochs=8000, verbose=verbose, filters=4, depth=2, decay=False, window=121, batch_size=32)
    if classifier_name == 'net1d-mod':
        from classifiers import net1d_mod
        return net1d_mod.Classifier_NET1d(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16,32,64], depth=2, window=[51,31,11], decay=False)


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
    parser.add_argument('--idx', default='/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz')
    parser.add_argument('--itr', default='')
    parser.add_argument('--merge_class', type=str2bool,
                        nargs='?', default=False)
    parser.add_argument('--gen_idx', type=str2bool, nargs='?', default=False)
    parser.add_argument('--train_idx', default='')
    parser.add_argument('--test_idx', default='')
    parser.add_argument('--archive', default='VA')
    args = parser.parse_args()
    main(args)
