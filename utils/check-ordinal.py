import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt

import coral_ordinal as coral
from sklearn import discriminant_analysis
from sklearn import decomposition


def check_grad(input, model, return_prediction=False):

    concat_layer = 'global_average_pooling1d'
    output_layer = 'coral_ordinal'

    gradModel = keras.models.Model(inputs=[model.inputs],
                                   outputs=[model.get_layer(concat_layer).output,
                                   model.get_layer(output_layer).output])

    # print(gradModel.summary())


    # input = np.expand_dims(input[np.where(input[:,0] > -900)[0], ...], axis=0)
    inp = tf.convert_to_tensor(input)
    # print(inp)

    # print(model.predict(inp))
    with tf.GradientTape() as tape:
        tape.watch(inp)
        concat_out, preds = gradModel(inp)
        pred_output = preds[:, 0]
        # pred2 = preds[:, 1]

    # print(concat_out.shape)
    # print(preds.shape)
    # assert False

    concat_grad1d = tape.gradient(pred_output, concat_out)
    concat_out = concat_out.numpy()#[0, ...]
    concat_grad1d = concat_grad1d.numpy()#[0, ...]
    # print(concat_out)
    # print(concat_grad1d)
    # print(concat_out * concat_grad1d)
    if return_prediction:
        return concat_out * concat_grad1d, preds

    return concat_out * concat_grad1d

    # print(concat_out)
    # print(concat_grad1d)
    # print(concat_out * concat_grad1d)




def main(args):
    idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    lit = os.path.basename(args.root).split('_')[0]
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    dataset = np.load(dp) if args.dataset == '' else np.load(args.dataset)

    x_all = dataset['mts']
    y_all = dataset['labels']

    num_folds = 0
    for file in os.listdir(args.root):
        if 'idx' in file:
            fold = int(file.split('.')[0].split('_')[1])
            num_folds = np.max((num_folds, fold))

    # channels = np.zeros((num_folds, x.shape[-1]))

    for fold in range(1, num_folds + 1):

        # idx = np.load(os.path.join(args.root, f'idx_{fold}.npz'))
        # idx = [117, 118]
        output = 0

        model_path = os.path.join(args.root, f'model_fold_{fold}.hdf5')
        model = keras.models.load_model(model_path, custom_objects={'CoralOrdinal': coral.CoralOrdinal, 'OrdinalCrossEntropy': coral.OrdinalCrossEntropy, 'MeanAbsoluteErrorLabels': coral.MeanAbsoluteErrorLabels})
        # model.summary()
        # assert False
        # out_layer = 'global_average_pooling1d'
        out_layer = 'dense_1'
        # out_layer = 'leaky_re_lu_2'
        feats_model = keras.models.Model(inputs=[model.inputs],
                                       outputs=[model.get_layer(out_layer).output])

        idxs = np.load(os.path.join(args.root, f'idx_{fold}.npz'))
        val_idx = idxs['val_idx']
        train_idx = idxs['train_idx']
        # x = x_all[val_idx, ...]
        # y = y_all[val_idx, ...]
        # x_train = x_all[train_idx, ...]
        # y_train = y_all[train_idx, ...]
        x = x_all
        y = y_all
        feats = np.array(feats_model(x))

        print(feats.shape)
        print(y.shape)
        # print(feats[train_idx, ...])

        lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
        transformed = lda.fit_transform(feats,y)
        # # pca = decomposition.PCA(n_components=2)
        # # transformed = pca.fit_transform(feats)
        # lda.fit(feats[train_idx, ...],y[train_idx,...])
        # transformed = lda.transform(feats[val_idx, ...])
        # print(transformed.shape)
        for i in range(3):
            idx = np.where(y==i)[0]
            label = f'class {i}'
            plt.scatter(transformed[idx,0], transformed[idx,1], label=label)

        plt.legend()

        lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
        transformed = lda.fit_transform(feats,y)
        # pca = decomposition.PCA(n_components=1)
        # transformed = pca.fit_transform(feats)
        # lda.fit(feats[train_idx, ...],y[train_idx,...])
        # transformed = lda.transform(feats[val_idx, ...])

        fig, axs = plt.subplots(3)
        for i in range(3):
            idx = np.where(y==i)[0]
            axs[i].hist(transformed[idx],range=(np.min(transformed), np.max(transformed)), bins=30)

        plt.show()
        # assert False

    #
    #     start = 0
    #     end = 53
    #     for i in range(10):
    #         xx = x[start:end, ...]
    #         yy = y[start:end, ...][:,0]
    #         result = np.sum(np.abs(check_grad(xx, model, output)), axis=0)
    #         # print(result)
    #         # result, preds = check_grad(xx, model, return_prediction=True)
    #         # preds = np.argmax(coral.ordinal_softmax(preds), axis=1)
    #         # print(f'preds shape {preds.shape}')
    #         # print(f'y shape {yy.shape}')
    #         # incorr = preds != yy
    #         # print(f'incorr shape {incorr}')
    #         # print(f'result shape {result}')
    #         # result[incorr, ...] = -result[incorr, ...]
    #         # print(f'result shape {result}')
    #         # result = np.sum(result, axis=0)
    #         # corr = ) != yy
    #         # print(corr)
    #         # print(corr.shape)
    #         # corr = preds == yy
    #         # print(result.shape)
    #         # assert False
    #         channels[fold-1, ...] = channels[fold-1, ...] + result / np.linalg.norm(result) if np.linalg.norm(result) != 0 else channels[fold-1, ...] + result
    #         print(f'{i} of 9 done in fold {fold}')
    #         start += 53
    #         end += 53
    #
    #
    # print(channels)
    # ch = np.mean(channels, axis=0)
    # std = np.std(channels, axis=0)
    # for i in range(channels.shape[1]):
    #     print(f'{ch[i]} +- {std[i]}')
    # # print(np.sum(channels, axis=0))
    # # print(np.sum(channels, axis=0) / (530 * 5))
    #     # print(channels)
    #     # print(mean)
    #     # for
    #     # check_grad(x, model, output)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
