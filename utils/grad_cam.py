#
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
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    classifier_layer_names = ['bn2d_last', 'relu2d_last', 'conv2d-1x1',
                              'tf_op_layer_Squeeze', 'conv-final',
                              'gap', 'result', 'sm']
    last_conv_layer = model.get_layer('conv2d_last')
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    # print(last_conv_layer)
    print(last_conv_layer_model.summary())

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    print(last_conv_layer.output.shape[1:])
    classifier_input = keras.Input(batch_shape=last_conv_layer.output.shape)
    # print(classifier_input)
    x = classifier_input
    for layer_name in classifier_layer_names:
        print('layer: {}, shape: {}'.format(layer_name, x.shape))
        print(model.get_layer(layer_name))
        x = model.get_layer(layer_name)(x)
        # print(x.input.shape)
    classifier_model = keras.Model(classifier_input, x)
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
    # print(top_class_channel)
    # print(preds)
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    print(np.max(grads), np.min(grads))
    # print(grads)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    print(grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    print(pooled_grads)
    # print(pooled_grads)
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    # print(last_conv_layer_output)
    pooled_grads = pooled_grads.numpy()
    # print(pooled_grads.shape[-1])
    # print(last_conv_layer_output.shape)
    # print(pooled_grads.shape)
    L2d = np.zeros((last_conv_layer_output.shape[0], last_conv_layer_output.shape[1]))
    print('conv layer shape {}'.format(last_conv_layer_output.shape))
    for i in range(pooled_grads.shape[0]):
        #iterate over inputs
        for j in range(pooled_grads.shape[1]):
            #iterate over feature maps
            L2d[:,i] = L2d[:,i] + last_conv_layer_output[:,i,j] * pooled_grads[i,j]
            # last_conv_layer_output[:, i] *= pooled_grads[i]
    print(L2d)
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

    # cam, heatmap = grad_cam(model, input, prediction, "conv_2")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('index', type=int)
    parser.add_argument('model')
    args = parser.parse_args()
    main(args)

# preprocessed_input = load_image(sys.argv[1])
#
# model = VGG16(weights='imagenet')
#
# predictions = model.predict(preprocessed_input)
# top_1 = decode_predictions(predictions)[0][0]
# print('Predicted class:')
# print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
#
# predicted_class = np.argmax(predictions)
# cam, heatmap = grad_cam(model, preprocessed_input,
#                         predicted_class, "block5_conv3")
# cv2.imwrite("gradcam.jpg", cam)
#
# register_gradient()
# guided_model = modify_backprop(model, 'GuidedBackProp')
# saliency_fn = compile_saliency_function(guided_model)
# saliency = saliency_fn([preprocessed_input, 0])
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
