
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
    # with tf.GradientTape() as tape:
    #     output_of_all_layers = grad_model(input)
    #     preds = output_layer[-1]  # last layer is output layer
    #     # take gradients of last layer with respect to all layers in the model
    #     grads = tape.gradient(preds, output_of_all_layers)
    #     # note: grads[-1] should be all 1, since it it d(output)/d(output)
    # print(grads)
    layer2d_name = last_conv1d_layer_name

    gradModel = keras.models.Model(inputs=[model.inputs],
                                   outputs=[model.get_layer('conv2d_last').output, model.get_layer('sm').output])

    inp = tf.convert_to_tensor(input)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        layer1d_out, preds = gradModel(inp)
        # tape.watch(layer1d_out)
        # tape.watch(layer2d_out)
        # tape.watch(preds)

    grad1d = tape.gradient(preds, layer1d_out)
    pooled_grads = tf.reduce_mean(grad1d, axis=(0, 1))

    layer1d_out = layer1d_out.numpy()[0]
    # print(last_conv_layer_output)
    pooled_grads = pooled_grads.numpy()
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
