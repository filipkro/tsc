#
# @misc{fchollet2020-gradcam,
#     title = "Grad-CAM class activation visualization",
#     author = "Francois Chollet",
#     howpublished = "\url{https://keras.io/examples/vision/grad_cam/}",
#     year = 2020,
#     note = "Accessed: 2020-11-20"}

# https://keras.io/examples/vision/grad_cam/
# from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
# from tensorflow.python.framework import ops
# import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
# import sys
# import cv2
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def make_gradcam_heatmap(input, model):

    last_conv1d_layer_name = 'lambda1d_last'
    last_conv2d_layer_name = 'lambda2d_last'
    classifier1d_layer_names = ['conv1d-1x1', 'lambda1d-final', 'conv-final',
                                'bn-final', 'relu-final', 'gap', 'result']
    classifier2d_layer_names = ['conv2d-1x1', 'lambda2d-final', 'conv-final',
                                'bn-final', 'relu-final', 'gap', 'result']
    # First, we create a model that maps the input to the activations
    # of the last conv2d layer, i.e. x -> A
    last_conv2d_layer = model.get_layer(last_conv2d_layer_name)
    last_conv2d_layer_model = keras.Model(model.inputs,
                                          last_conv2d_layer.output)

    # Likewise create model that maps input to activations of the last conv1d
    # layer, i.e. x -> M
    last_conv1d_layer = model.get_layer(last_conv1d_layer_name)
    last_conv1d_layer_model = keras.Model(model.inputs,
                                          last_conv2d_layer.output)

    # Second, we create a model that maps the activations of the last conv2d
    # layer to the final class predictions
    classifier2d_input = keras.Input(shape=last_conv2d_layer.output.shape[1:])
    # print(classifier_input)
    x = classifier2d_input
    for layer_name in classifier2d_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier2d_model = keras.Model(classifier2d_input, x)
    # print(classifier_model.summary())

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(input)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # print('line 170:',top_class_channel)
    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    # print(grads)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    # print(pooled_grads)
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    # print(last_conv_layer_output)
    pooled_grads = pooled_grads.numpy()
    # print(pooled_grads.shape[-1])
    # print(last_conv_layer_output.shape)
    # print(pooled_grads.shape)
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def main(args):
    model = keras.models.load_model(args.model)
    data = np.load(args.dataset)['mts']
    # prediction = np.argmax(model(input)[0])
    input = np.expand_dims(data[1, ...], 0)

    pred = model(input)[0]
    print(pred)
    print(data[0, ...].shape)
    print(model.summary())
    # input = data

    hm = make_gradcam_heatmap(input, model, 'concatenate_2', [
                              'global_average_pooling1d', 'result'])

    print(hm)
    print(hm.shape)
    print(input.shape)
    plt.plot(hm)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('index', type=int)
    parser.add_argument('model')
    args = parser.parse_args()
    main(args)
