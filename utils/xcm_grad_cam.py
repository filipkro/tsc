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

    last_conv1d_layer_name = 'relu1d_last'
    last_conv2d_layer_name = 'relu2d_last'
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
    x = classifier2d_input
    for layer_name in classifier2d_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier2d_model = keras.Model(classifier2d_input, x)

    # Likewise from last conv1d to final class predictions
    classifier1d_input = keras.Input(shape=last_conv1d_layer.output.shape[1:])
    x = classifier1d_input
    for layer_name in classifier1d_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier1d_model = keras.Model(classifier1d_input, x)
    # print(classifier_model.summary())



    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv1d_layer_output = last_conv1d_layer_model(input)
        tape.watch(last_conv1d_layer_output)
        # Compute class predictions
        preds1d = classifier1d_model(last_conv1d_layer_output)
        top_pred1d_index = tf.argmax(preds1d[0])
        top_class1d_channel = preds[:, top_pred1d_index]

        last_conv2d_layer_output = last_conv2d_layer_model(input)
        tape.watch(last_conv2d_layer_output)
        # Compute class predictions
        preds2d = classifier2d_model(last_conv2d_layer_output)
        top_pred2d_index = tf.argmax(preds2d[0])
        top_class2d_channel = preds[:, top_pred2d_index]

    # print('line 170:',top_class_channel)
    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads1d = tape.gradient(top_class1d_channel, last_conv1d_layer_output)
    grads2d = tape.gradient(top_class2d_channel, last_conv2d_layer_output)
    # print(grads)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads1d = tf.reduce_mean(grads1d, axis=(0, 1))
    pooled_grads2d = tf.reduce_mean(grads2d, axis=(0, 1))
    # print(pooled_grads)
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv1d_layer_output = last_conv1d_layer_output.numpy()[0]
    last_conv2d_layer_output = last_conv2d_layer_output.numpy()[0]
    # print(last_conv_layer_output)
    pooled_grads1d = pooled1d_grads.numpy()
    pooled_grads2d = pooled2d_grads.numpy()
    # print(pooled_grads.shape[-1])
    # print(last_conv_layer_output.shape)
    # print(pooled_grads.shape)
    for i in range(pooled1d_grads.shape[-1]):
        last_conv1d_layer_output[:, i] *= pooled1d_grads[i]
    for i in range(pooled2d_grads.shape[-1]):
        last_conv2d_layer_output[:, i] *= pooled2d_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap1d = np.mean(last_conv1d_layer_output, axis=-1)
    heatmap2d = np.mean(last_conv2d_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap1d = np.maximum(heatmap1d, 0) / np.max(heatmap1d)
    heatmap2d = np.maximum(heatmap2d, 0) / np.max(heatmap2d)
    return heatmap1d, heatmap2d


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
