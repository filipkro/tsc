
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets

# import matplotlib.pyplot as plt


def fit_classifier():
    # x_train = datasets_dict[dataset_name][0]
    # y_train = datasets_dict[dataset_name][1]
    # x_test = datasets_dict[dataset_name][2]
    # y_test = datasets_dict[dataset_name][3]

    x = dataset['mts']
    y = dataset['labels']

    print('x shape::', x.shape)

    train_size = int(np.round(0.9 * len(y)))

    train_idx = np.random.choice(len(y), train_size, replace=False)
    train_idx
    x_train = x[train_idx, ...]
    x_test = np.delete(x, train_idx, axis=0)
    y_train = y[train_idx]
    y_test = np.delete(y, train_idx)


    # plt.plot(x_train[33, :, 0])
    # plt.plot(x_train[33, :, 1])
    # print(y_train[33])
    # plt.show()

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
        return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
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
        return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, nb_filters=16)
    if classifier_name == 'inception_simple':
        from classifiers import inception_simple
        return inception_simple.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


root_dir = sys.argv[1]
archive_name = sys.argv[2]
dataset_path = sys.argv[3]
train_fp = sys.argv[4]
classifier_name = sys.argv[5]

rate = dataset_path.split('-')[-1].split('.')[0]
itr = sys.argv[5]

if itr == '_itr_0':
    itr = ''

output_directory = root_dir + '/' + classifier_name + '/' + rate + '-' + archive_name + itr + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', archive_name, dataset_path, classifier_name, itr)

if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:

    create_directory(output_directory)
    # datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
    dataset = np.load(dataset_path)
    fit_classifier()

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
