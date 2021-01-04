import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pickle
from kerastuner import HyperModel

import os
from keras.utils.layer_utils import count_params
from utils.lr_schedules import StepDecay

import coral_ordinal as coral


class HyperInceptionCoral(HyperModel):

    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def _inception_module(self, input_tensor, masked, hp, stride=1):
        activation = hp.Choice('activation', ('linear', 'relu'))
        input_tensor = keras.layers.Lambda((lambda x: x))(input_tensor,
                                                          mask=masked[:, :, 0])
        bottleneck_size = hp.Int('bottleneck_size', 8, 32, step=8)
        if hp.Boolean('bottleneck') and int(input_tensor.shape[-1]) > bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=bottleneck_size,
                                                  kernel_size=1, padding='same',
                                                  activation=activation,
                                                  use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [
            hp.Int('kernel_size', 8, 56, step=8) // (2 ** i) for i in range(3)]

        conv_l = []

        input_inception = keras.layers.Lambda((lambda x: x))(input_inception,
                                                             mask=masked[:, :, 0])
        filters = hp.Int('filters', 32, 138, step=32)
        for i in range(len(kernel_size_s)):
            conv_l.append(keras.layers.Conv1D(filters=filters,
                                              kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same',
                                              activation=activation,
                                              use_bias=False)(input_inception))

        max_pool_1 = keras.layers.MaxPool1D(
            pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=filters, kernel_size=1,
                                     padding='same', activation=activation,
                                     use_bias=False)(max_pool_1)

        conv_6 = keras.layers.Lambda((lambda x: x))(conv_6,
                                                    mask=masked[:, :, 0])

        conv_l.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_l)

        x = keras.layers.Lambda((lambda x: x))(x, mask=masked[:, :, 0])
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]),
                                         kernel_size=1, padding='same',
                                         use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build(self, hp):
        input_layer = keras.layers.Input(self.input_shape)
        masked_layer = keras.layers.Masking(mask_value=-1000,
                                            name='mask')(input_layer)
        x = masked_layer
        input_res = masked_layer

        for d in range(hp.Int('inception_modules', 1, 3)):

            x = self._inception_module(x, masked_layer, hp)

            if hp.Boolean('use_residual') and d % 3 == 2:
                input_res = keras.layers.Lambda((lambda x: x))(input_res,
                                                               mask=masked_layer[:, :, 0])
                x = self._shortcut_layer(input_res, x)
                input_res = x

        x = keras.layers.Dropout(hp.Float('dropout', 0.0, 0.4, step=0.1))(x)
        gap_layer = keras.layers.GlobalAveragePooling1D()(
            x, mask=masked_layer[:, :, 0])

        for i in range(hp.Int('nb_dense', 0, 2, step=1)):
            gap_layer = keras.layers.Dense(
                hp.Int(f"dense_{i}", 16, 64, step=16))(gap_layer)

        output_layer = coral.CoralOrdinal(self.num_classes)(gap_layer)
        # output_layer = keras.layers.Dense(self.num_classes, activation='softmax',
        #                                   name='result2')(gap_layer)

        # model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model = keras.models.Model(inputs=input_layer,
                                   outputs=output_layer)

        print('line 106')

        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
        #               metrics=['accuracy'])
        lr = hp.Float('learning_rate', 1e-5, 1e-2,
                      sampling='LOG', default=1e-3)
        model.compile(loss=coral.OrdinalCrossEntropy(num_classes=self.num_classes),
                      optimizer=keras.optimizers.Adam(lr),
                      metrics=[coral.MeanAbsoluteErrorLabels()])

        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
        #                                               actor=0.5, patience=50,
        #                                               min_lr=0.0001)

        # file_path = self.output_directory + 'best_model.hdf5'

        # model_checkpoint = keras.callbacks.ModelCheckpoint(
        #     filepath=file_path, monitor='val_accuracy',
        #     save_best_only=True, mode='max')

        # stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                            restore_best_weights=True,
        #                                            patience=300)

        schedule = StepDecay(initAlpha=lr, factor=hp.Float(
            'lr_factor', 0.7, 1.0, step=0.1), dropEvery=hp.Int('lr_dropstep', 10, 40, step=10))
        lr_decay = keras.callbacks.LearningRateScheduler(schedule)

        self.callbacks = [lr_decay]

        model.summary()
        return model
