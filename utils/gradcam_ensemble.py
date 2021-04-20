import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt

import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy


def check_grad(input, model, output):

    last_conv1d_layer_name = 'bn1d_last'
    last_conv2d_layer_name = 'bn2d_last'
    concat_layer = 'concat'

    output_layer = 'coral_ordinal'


    # output_layer = model.outputs
    # all_layers = [layer.output for layer in model.layers]
    # grad_model = keras.models.Model(inputs=model.inputs, outputs=all_layers)
    #
    # with tf.GradientTape() as tape:    #     output_of_all_layers = grad_model(input)
    #     preds = output_layer[-1]  # last layer is output layer
    #     # take gradients of last layer with respect to all layers in the model
    #     grads = tape.gradient(preds, output_of_all_layers)
    #     # note: grads[-1] should be all 1, since it it d(output)/d(output)
    # print(grads)
    layer2d_name = last_conv1d_layer_name

    gradModel = keras.models.Model(inputs=[model.inputs],
                                   outputs=[model.get_layer(concat_layer).output,
                                   model.get_layer(output_layer).output])

    gradModel.summary()

    # print(input)
    # print(input.shape)
    # print(np.where(input[:,0] > -900)[0])
    # assert False
    input = np.expand_dims(input[np.where(input[:,0] > -900)[0], ...], axis=0)
    inp = tf.convert_to_tensor(input)

    print(model.predict(inp))
    with tf.GradientTape() as tape:
        tape.watch(inp)
        concat_out, preds = gradModel(inp)
        pred_output = preds[:, 0]
        # pred2 = preds[:, 1]



    concat_grad1d = tape.gradient(pred_output, concat_out)



    print(preds)
    print(pred_output)
    # print(pred2)

    qk1d = tf.reduce_mean(concat_grad1d, axis=(0,1))
    concat_out = concat_out.numpy()[0, ...]
    qk = qk1d.numpy()
    L1d = np.zeros((concat_out.shape[0], concat_out.shape[1]))

    for i in range(qk.shape[0]):
        #iterate over inputs
        L1d[:,i] = concat_out[:,i] * qk[i]
        # last_conv_layer_output[:, i] *= pooled_grads[i]
    print(L1d.shape)
    print('gradient shape: {}'.format(concat_grad1d.shape))
    print(f'shape reduced mean: {qk1d.shape}')


    print('**************************************')
    # print(layer2d_out)
    # print(layer2d_out)
    print('**************************************')
    # print(last_conv_layer_output)


    # print(layer1d_out.shape)
    # print(layer2d_out.shape)


    print('conv layer shape {}'.format(concat_out.shape))
    print('weight shape: {}'.format(qk.shape))
    print(input.shape)

    # if input.shape[-1] < 5:
    #
    # else:

    # for i in range(L1d.shape[1]):
    #     plt.figure(1)
    #     plt.plot(L1d[:,i], label=f'{i}')
    #     plt.figure(2)
    #     plt.plot(input[0,:,i], label=f'{i}')
    #
    # # plt.plot(L1d)
    #
    # plt.figure(1)
    # plt.legend()
    # plt.figure(2)
    # plt.legend()

    # plt.figure()
    # plt.plot(concat_out)
    #
    # plt.show()

    # L1d = (L1d)/(np.max(L1d) - np.min(L1d))

    # print(L1d.shape)
    # print(L2d.shape)

    return L1d

def plotgradcam_input(input, heatmap, subplots_per_fig=5):
    nb_inputs = input.shape[-1]
    max_idx = np.where(input[..., 0] < -900)[0]
    max_idx = max_idx[0] if len(max_idx) > 0 else input.shape[0]

    if nb_inputs > subplots_per_fig:
        raise NotImplementedError

    fig, axs = plt.subplots(nb_inputs)
    for i in range(nb_inputs):
        sc = axs[i].scatter(np.linspace(0, max_idx - 1, max_idx),
                            input[:max_idx, i],
                            c=heatmap[:max_idx,i], cmap='cool',
                            vmin=np.min(heatmap), vmax=np.max(heatmap))
    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)





def main(args):
    idx_path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'
    lit = os.path.basename(args.root).split('_')[0]
    if lit[-2].isdigit():
        assert lit[-1].isdigit()
        num_folds = int(lit[-2:])
        lit = lit[:-2]
        print(num_folds)
        print(lit)
    dp = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '.npz'
    dp100 = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '_len100' + '.npz'
    info_file = '/home/filipkr/Documents/xjob/data/datasets/data_' + lit + '-info.txt'
    dataset = np.load(dp)
    dataset100 = np.load(dp100)

    x = dataset['mts']
    x100 = dataset100['mts']
    y = dataset['labels']

    fold = 1
    idxs = [500, 100, 479]
    output = 0

    models = ['coral-100-7000', 'xx-coral-100-7000','xx-conf-100-7000', 'xx-conf-9000','xx-conf-100-11000']
    weights = np.array([[1/3,1.25/3,1/3],[1/3,1.25/3,1/3],[1/3,0,0], [0,1.25/3,0],[0,0,1/3]])

    ensembles = [os.path.join(args.root, i) for i in models]
    paths = [os.path.join(
        root, f'model_fold_{fold}.hdf5') for root in ensembles]

    for model_i, model_path in enumerate(paths):
        model = keras.models.load_model(model_path, custom_objects={
                                        'CoralOrdinal': coral.CoralOrdinal,
                                        'OrdinalCrossEntropy':
                                        coral.OrdinalCrossEntropy,
                                        'MeanAbsoluteErrorLabels':
                                        coral.MeanAbsoluteErrorLabels,
                                        'ConfusionCrossEntropy':
                                        ConfusionCrossEntropy})
        input = x100 if '-100-' in model_path else x
        for i in idxs:
            inp = np.expand_dims(input[i,...], axis=0)
            print(inp.shape)
            pred = model(inp)
            if 'coral' in model_path:
                pred = coral.ordinal_softmax(pred)
            print(pred)




    assert False

    model_path = os.path.join(args.root, f'model_fold_{fold}.hdf5')
    model = keras.models.load_model(model_path, custom_objects={'CoralOrdinal': coral.CoralOrdinal, 'OrdinalCrossEntropy': coral.OrdinalCrossEntropy, 'MeanAbsoluteErrorLabels': coral.MeanAbsoluteErrorLabels})

    for idx in idxs:
        heatmap = check_grad(x[idx, ...], model, output)
        plotgradcam_input(x[idx, ...], heatmap)
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
