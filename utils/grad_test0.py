
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


def check_grad(input, model):

    last_conv1d_layer_name = 'bn1d_last'
    last_conv2d_layer_name = 'bn2d_last'
    classifier1d_layer_names = ['conv1d-1x1', 'lambda1d-final', 'conv-final',
                                'bn-final', 'relu-final', 'gap', 'result']
    classifier2d_layer_names = ['conv2d-1x1', 'lambda2d-final', 'conv-final',
                                'bn-final', 'relu-final', 'gap', 'result']

    layer_name = last_conv2d_layer_name

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
                                   outputs=[model.get_layer('conv1d_last').output,
                                   model.get_layer('conv2d_last').output,
                                            model.get_layer('sm').output])

    inp = tf.convert_to_tensor(input)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        layer1d_out, _, preds = gradModel(inp)
        top_pred_index = tf.argmax(preds[0])
        top1d = preds[:, top_pred_index]

    grad1d = tape.gradient(top1d, layer1d_out)

    with tf.GradientTape() as tape:
        tape.watch(inp)
        _, layer2d_out, preds = gradModel(inp)
        # print(preds)
        top_pred_index = tf.argmax(preds[0])
        # print(top_pred_index)
        top2d = preds[:, top_pred_index]
        # print(top_class_channel)

    grad2d = tape.gradient(top2d, layer2d_out)
    # print(grad2d)
    # print(np.max(grad2d), np.min(grad2d))
    # assert False
    print('grad 1d shape: {}'.format(grad1d.shape))
    qk1d = tf.reduce_mean(grad1d, axis=(0,1))
    # print(qk1d.shape)
    # assert False
    wk2d = tf.reduce_mean(grad2d, axis=(0, 1))
    # print(qk1d.shape)
    # print(wk2d)

    layer1d_out = layer1d_out.numpy()[0, ...]
    print('**************************************')
    # print(layer2d_out)
    layer2d_out = layer2d_out.numpy()[0, ...]
    # print(layer2d_out)
    print('**************************************')
    # print(last_conv_layer_output)
    qk = qk1d.numpy()
    wk = wk2d.numpy()
    # print(qk.shape)
    # print(wk.shape)

    # print(layer1d_out.shape)
    # print(layer2d_out.shape)

    L1d = np.zeros((layer1d_out.shape[0]))
    print('conv layer shape {}'.format(layer1d_out.shape))
    print('weight shape: {}'.format(qk.shape))
    for i in range(qk.shape[0]):
        #iterate over inputs
        L1d = L1d + layer1d_out[:,i] * qk[i]
        # last_conv_layer_output[:, i] *= pooled_grads[i]
    # print(L1d)

    L2d = np.zeros((layer2d_out.shape[0], layer2d_out.shape[1]))
    print('conv layer shape {}'.format(layer2d_out.shape))
    for i in range(wk.shape[0]):
        #iterate over inputs
        for j in range(wk.shape[1]):
            #iterate over feature maps
            L2d[:,i] = L2d[:,i] + layer2d_out[:,i,j] * wk[i,j]
            # last_conv_layer_output[:, i] *= pooled_grads[i]
    # print(L2d)


    L1d = (L1d)/(np.max(L1d) - np.min(L1d))
    L2d = (L2d)/(np.max(L2d) - np.min(L2d))

    # print(L1d.shape)
    # print(L2d.shape)

    return L1d, L2d

    assert False

    # L1d = np.zeros((layer1d_out.shape[0]))#, layer1d_out.shape[1]))
    # print(L1d.shape)
    # for i in range(len(qk)):
    #     print('in loop', layer1d_out[..., i].shape)
    #     L1d = L1d + qk[i] * layer1d_out[..., i]
    # print(L1d.shape)
    # L1d = np.maximum(L1d, 0)
    #
    # L2d = np.zeros((layer2d_out.shape[0], layer2d_out.shape[1]))
    # for i in range(len(wk)):
    #     L2d = L2d + wk[i] * layer2d_out[..., i]
    # # L2d = np.maximum(L2d, 0)
    # print(L2d)
    # print(L1d)
    print('grad: {}'.format(grad2d))
    print('feats: {}'.format(layer2d_out))
    # print(L1d.shape)
    print(L2d.shape)
    assert False
    # for i in range(pooled_grads.shape[-1]):
    #     layer1d_out[:, i] *= pooled_grads[i]
    #
    # heatmap = np.mean(last_conv_layer_output, axis=-1)
    # print(heatmap)

    # with tf.GradientTape() as tape:
    #     tape.watch(inp)
    #     _, layer2d_out, preds = gradModel(inp)

    # grad2d = tape.gradient(preds, layer2d_out)
    p = preds[:, 0]
    print(preds)
    # , shape=(1, 3)) #, dtype=float32)

    # castConvOutputs = tf.cast(layer_out > 0, "float32")
    # castGrads = tf.cast(grads > 0, "float32")
    # guidedGrads = castConvOutputs * castGrads * grads
    print(layer1d_out)
    print(preds)
    # print(grads)
    print(np.max(grad1d), np.min(grad1d))
    print(grad1d)
    # print(grad2d)
    # print(model.summary())
    # print(model.output)
    # print(guidedGrads)
    # print(gradModel.summary())
    print(pooled_grads)
    print(pooled_grads.shape)
