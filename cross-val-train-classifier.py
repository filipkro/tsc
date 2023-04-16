from utils.utils import create_directory
import os
import numpy as np
from sklearn.model_selection import KFold
from argparse import ArgumentParser, ArgumentTypeError
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import coral_ordinal as coral
import pandas as pd
from keras.utils import to_categorical

def idx_same_subject(meta, subject):
    indices = np.where(meta[:, 1] == subject)[0]
    return indices


def get_max_subj(info_file):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    data = np.array(meta_data.values[first_data:, 1], dtype=int)

    return max(data)


def read_meta_data(info_file):
    meta_data = pd.read_csv(info_file, delimiter=',')
    first_data = np.where(meta_data.values[:, 0] == 'index')[0][0] + 1
    meta_data = np.array(meta_data.values[first_data:, :4], dtype=int)
    return meta_data


def fit_classifier(dp, classifier_name, output_directory):

    dataset = np.load(dp)
    info_file = dp.split('.npz')[0] + '-info.txt'
    meta_data = read_meta_data(info_file)

    x = dataset['mts']
    y = dataset['labels']
    merge_class = False 
    if merge_class:
        idx = np.where(y == 2)[0]
        y[idx] = 1

    nb_classes = len(np.unique(y))
    print(f'Number of classes: {nb_classes}')

    y_oh = to_categorical(y)
    print(f'y shape: {y_oh.shape}')
    if len(x.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x = x.reshape((x.shape[0], x.shape[1], 1))

    input_shape = x.shape[1:]

    num_folds = 10 
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
    fold = 1
    acc_per_fold = []
    loss_per_fold = []
    abs_err = []

    if 'coral' in classifier_name or 'focal' in classifier_name or 'reg' in classifier_name:
        # these models should not have one hot encoded y
        y_oh = y

    print(x.shape[0])

    # assert False

    cnf_matrix = np.zeros((nb_classes, nb_classes))

    for train, val in kfold.split(np.arange(get_max_subj(info_file) + 1)):
        print(f'Fold number {fold} out of {num_folds}')

        # make sure videos by the same subject not in both train and val data 
        train_idx = np.concatenate([idx_same_subject(meta_data, subj)
                                    for subj in train])
        val_idx = np.concatenate([idx_same_subject(meta_data, subj)
                                  for subj in val])

        print(f'train data len: {len(train_idx)}')
        print(f'val data len: {len(val_idx)}')

        class_weight = {0: 2, 1: 1, 2: 1.5}
        class_weight = None
        if 'coral' in classifier_name:
            n0 = (y[train_idx, ...] == 0).sum()
            n1 = (y[train_idx, ...] == 1).sum()
            n2 = (y[train_idx, ...] == 2).sum()
            class_weight = [np.max((n0, n1 + n2)) / n0,
                            np.max((n0 + n1, n2)) / (n1 + n2)]
            class_weight = [2,1]
            class_weight = None
            classifier = create_classifier(classifier_name, input_shape,
                                           nb_classes, output_directory,
                                           class_weight=class_weight)
            print(f'class weight: {class_weight}')
            print(f'n0: {n0}, n1: {n1}, n2: {n2}')
        else:
            #class_weight = {0:1, 1:3, 2:5}
            classifier = create_classifier(classifier_name, input_shape,
                                           nb_classes, output_directory)
            print(f'class weight: {class_weight}')

        if fold == 1:
            # print and save some info about model to file for first fold
            if 'conf' in classifier_name:
                print(f"U: \n{classifier.model.loss.get_config()['U']}")

            ifile = open(os.path.join(output_directory, 'class_weights.txt'), 'w')
            ifile.write(f'model type: {classifier_name}\n')
            U = classifier.model.loss.get_config()['U'] if 'conf' in classifier_name else class_weight
            ifile.write(f'Training weight:\n')
            ifile.write(str(U))
            ifile.close()
            print('file done')
            print(classifier.model.summary())

        print(x[train_idx, ...].shape)
        print(y_oh[train_idx, ...].shape)

        classifier.fit(x[train_idx, ...], y_oh[train_idx, ...],
                       x[val_idx, ...], y_oh[val_idx, ...])

        scores = classifier.model.evaluate(x[val_idx, ...],
                                           y_oh[val_idx, ...], verbose=0)

        probs = classifier.model(x[val_idx, ...], training=False)

        if 'reg' in classifier_name:
            # ?
            preds = np.zeros(probs.shape)
            for i, pred in enumerate(probs):
                pred = int(np.round(pred))
                if pred > 6:
                    pred = 6
                elif pred < 0:
                    pred = 0
                preds[i] = pred

            acc = accuracy_score(y[val_idx], preds)
            print(f'Score for fold {fold}: Loss: {scores}; Accuracy: {acc}')
            acc_per_fold.append(acc)
            loss_per_fold.append(scores)

        elif 'coral' in classifier_name:
            probs = coral.ordinal_softmax(probs)
            preds = np.argmax(probs, axis=1)
            acc = accuracy_score(y[val_idx], preds)
            print(f'Score for fold {fold}: {classifier.model.metrics_names[0]}'
                  ' of {scores[0]}; {classifier.model.metrics_names[1]} of '
                  '{scores[1]}; Accuracy of {acc}')
            abs_err.append(scores[1])
            acc_per_fold.append(acc)
            loss_per_fold.append(scores[0])
        else:
            preds = np.argmax(probs, axis=1)
            print(f'Score for fold {fold}: {classifier.model.metrics_names[0]}'
                  ' of {scores[0]}; {classifier.model.metrics_names[1]} of '
                  '{scores[1]}')
            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])

        cm_tmp = confusion_matrix(y[val_idx], preds, labels=range(nb_classes))
        cnf_matrix = cnf_matrix + cm_tmp
        print(cm_tmp)

        model_name = os.path.join(output_directory, f'model_fold_{fold}.hdf5')
        dataset_name = os.path.join(output_directory, f'idx_{fold}.npz')
        classifier.model.save(model_name)
        np.savez(dataset_name, train_idx=train_idx, val_idx=val_idx)

        fold += 1

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(
        f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold) * 2 / np.sqrt(num_folds)})')
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
    ifile.close()
    print(acc_per_fold)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=2, class_weight=None):
    if classifier_name == 'masked-fcn':
        from classifiers import masked_fcn
        return masked_fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=5000, kernel_size=101, filters=8, batch_size=16, depth=3)
    if classifier_name == 'masked-resnet':
        from classifiers import masked_resnet
        return masked_resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose, nb_epochs=5000)
    if classifier_name == 'masked-resnet-mod':
        from classifiers import masked_resnet_mod
        return masked_resnet_mod.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose, nb_epochs=5000, batch_size=32, n_feature_maps=64, depth=4)
    if classifier_name == 'masked-inception':
        from classifiers import masked_inception
        return masked_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=5000, bottleneck_size=32, use_residual=True)
    if classifier_name == 'masked-inception-mod':
        from classifiers import masked_inception_mod
        return masked_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=False, lr=0.005)
    if classifier_name == 'inception-conf':
        from classifiers import inception_conf
        return inception_conf.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=False, lr=0.005)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=False, lr=0.005)
    if classifier_name == 'inception-coral':
        from classifiers import inception_coral
        return inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'inception-reg':
        from classifiers import inception_reg
        return inception_reg.Classifier_REGRESSION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'bayes-inception-coral':
        from classifiers import bayes_inception_coral
        return bayes_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'simple-bayes-coral':
        from classifiers import simple_bays_coral
        return simple_bays_coral.Classifier_SIMPLE_BAYS_CORAL(output_directory, input_shape, nb_classes, verbose, depth=5, nb_filters=128, kernel_size=41, nb_epochs=2000, lr=0.005)
    if classifier_name == 'inception-focal':
        from classifiers import inception_focal
        return inception_focal.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=8, use_residual=True, lr=0.005)
    if classifier_name == 'x-inception':
        from classifiers import x_inception
        return x_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception':
        from classifiers import xx_inception
        # return xx_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=8, kernel_size=21, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.01, use_bottleneck=False)
        return xx_inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-focal':
        from classifiers import xx_inception_focal
        return xx_inception_focal.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=600, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-conf':
        from classifiers import xx_inception_conf
        return xx_inception_conf.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-coral':
        from classifiers import xx_inception_coral
        # return xx_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=64, kernel_size=51, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.01, use_bottleneck=False)
        return xx_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=True, class_weight=class_weight)
        #return xx_inception_coral.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=64, kernel_size=15, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False, class_weight=class_weight)
    if classifier_name == 'xx-inception-reg':
        from classifiers import xx_inception_reg
        return xx_inception_reg.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, use_bottleneck=False)
    if classifier_name == 'xx-inception-evidence':
        from classifiers import xx_inception_evidence
        # return xx_inception_evidence.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=64, kernel_size=21, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.01, use_bottleneck=False)
        return xx_inception_evidence.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.05, use_bottleneck=False, batch_size=16)
    if classifier_name == 'xx-inception-coral-ext':
        from classifiers import xx_inception_coral_ext
        return xx_inception_coral_ext.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=64, kernel_size=21, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.01, use_bottleneck=False)
    if classifier_name == 'coral-inception-mod':
        from classifiers import coral_inception_mod
        return coral_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, batch_size=16)
        #return coral_inception_mod.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=2, nb_filters=128,     kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.005, batch_size=4)
    if classifier_name == 'malstm-fcn':
        from classifiers import malstm_fcn
        return malstm_fcn.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, depth=1, nb_filters=32, kernel_size=31, nb_epochs=2000, bottleneck_size=32, use_residual=False, lr=0.001, use_bottleneck=False)
    if classifier_name == 'xcm':
        from classifiers import xcm
        return xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose)
    if classifier_name == 'masked-xcm':
        from classifiers import masked_xcm
        return masked_xcm.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[16, 32, 64], depth=2, window=[51, 31, 11], decay=False)
    if classifier_name == 'masked-xcm-mod':
        from classifiers import masked_xcm_mod
        return masked_xcm_mod.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=5000, verbose=verbose, filters=[128, 128, 64], depth=2, window=[41, 31, 21], decay=False, batch_size=32)
    if classifier_name == 'masked-xcm-2d':
        from classifiers import masked_xcm_2d
        return masked_xcm_2d.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=2000, verbose=verbose, filters=256, depth=1, window=41, decay=False, batch_size=32)
    if classifier_name == 'xcm-coral':
        from classifiers import xcm_coral
        return xcm_coral.Classifier_XCM(output_directory, input_shape, nb_classes, nb_epochs=2000, verbose=verbose, filters=32, depth=2, window=31, decay=False, batch_size=32, use_bottleneck=True, use_1d=False)
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
        fit_classifier(args.dataset, classifier_name, output_directory)
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
