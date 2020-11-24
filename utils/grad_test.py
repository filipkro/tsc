
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


def check_grad(input, model):

    last_conv1d_layer_name = 'lambda1d_last'
    last_conv2d_layer_name = 'lambda2d_last'
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
    layer_name = last_conv2d_layer_name

    gradModel = keras.models.Model(inputs=[model.inputs],
                                   outputs=[model.get_layer(layer_name).output,
                                            model.output])

    inp = tf.convert_to_tensor(input)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        layer_out, preds = gradModel(inp)
        tape.watch(layer_out)
        tape.watch(preds)

    p = preds[:,0]
    print(preds)
    one_hot = tf.one_hot(tf.nn.top_k(preds).indices, tf.shape(preds)[0])#, shape=(1, 3)) #, dtype=float32)

    grads = tape.gradient(preds, layer_out)

    # castConvOutputs = tf.cast(layer_out > 0, "float32")
    # castGrads = tf.cast(grads > 0, "float32")
    # guidedGrads = castConvOutputs * castGrads * grads
    print(layer_out)
    print(preds)
    print(grads)
    print(np.max(grads))
    # print(model.output)
    # print(guidedGrads)
    # print(gradModel.summary())
