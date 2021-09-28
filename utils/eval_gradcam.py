import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt

import coral_ordinal as coral


def check_grad(input, model, output=0):
    '''performs forward pass with model on input,
    calculates and returns grad cam between concat_layer and output_layer'''

    concat_layer = 'concat'
    output_layer = 'coral_ordinal'

    gradModel = keras.models.Model(inputs=[model.inputs],
                                   outputs=[model.get_layer(concat_layer).output,
                                   model.get_layer(output_layer).output])

    input = np.expand_dims(input[np.where(input[:, 0] > -900)[0], ...], axis=0)
    inp = tf.convert_to_tensor(input)

    with tf.GradientTape() as tape:
        tape.watch(inp)
        concat_out, preds = gradModel(inp)
        pred_output = preds[:, 0]

    concat_grad1d = tape.gradient(pred_output, concat_out)

    # see report and XCM paper for details, basically L1d is activation map
    # (output of concat_layer) weighted with gradient between this and
    # output_layer
    qk1d = tf.reduce_mean(concat_grad1d, axis=(0, 1))
    concat_out = concat_out.numpy()[0, ...]
    qk = qk1d.numpy()
    L1d = np.zeros((concat_out.shape[0], concat_out.shape[1]))

    for i in range(qk.shape[0]):
        # iterate over inputs
        L1d[:, i] = concat_out[:,i] * qk[i]

    return L1d


def plotgradcam_input(input, heatmap, subplots_per_fig=5, title=''):
    '''plots input time series colored by grad cam (heatmap)'''
    nb_inputs = input.shape[-1]
    max_idx = np.where(input[..., 0] < -900)[0]
    max_idx = max_idx[0] if len(max_idx) > 0 else input.shape[0]

    if nb_inputs > subplots_per_fig:
        raise NotImplementedError

    fig, axs = plt.subplots(nb_inputs, gridspec_kw={'hspace': 0.05})
    fig.suptitle(title, fontsize=18)
    for i in range(nb_inputs):
        sc = axs[i].scatter(np.linspace(0, max_idx - 1, max_idx),
                            input[:max_idx, i],
                            c=heatmap[:max_idx, i], cmap='cool',
                            vmin=np.min(heatmap), vmax=np.max(heatmap))
        if i < 3:
            axs[i].axes.xaxis.set_ticklabels([])
    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)


def main(args):
    '''runs forward passes for the data specified in idxs, using the specified
    model and plots the input time series colored with the grad cam weights for
    each time step'''
    # idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    lit = os.path.basename(args.root).split('_')[0]
    # lit = 'Olga-Tokarczuk_len100'
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    # info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    dataset = np.load(dp)

    x = dataset['mts']
    y = dataset['labels']

    fold = 1
    idxs = [5, 66, 68, 500, 100, 72]
    output = 0

    model_path = os.path.join(args.root, f'model_fold_{fold}.hdf5')
    model = keras.models.load_model(model_path,
                                    custom_objects={'CoralOrdinal':
                                                    coral.CoralOrdinal,
                                                    'OrdinalCrossEntropy':
                                                    coral.OrdinalCrossEntropy,
                                                    'MeanAbsoluteErrorLabels':
                                                    coral.MeanAbsoluteErrorLabels})

    for idx in idxs:
        prd = model(np.expand_dims(x[idx, ...], axis=0))
        prd = coral.ordinal_softmax(prd)
        print(np.argmax(prd))
        print(y[idx])
        title = f'Correct: {int(y[idx][0])}, Predicted: {np.argmax(prd)}'
        heatmap = check_grad(x[idx, ...], model, output)
        plotgradcam_input(x[idx, ...], heatmap, title=title)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    # parser.add_argument('data')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--test_idx', default='')
    args = parser.parse_args()
    main(args)
