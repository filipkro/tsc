import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pickle
from keras import backend as K

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration
import os
from keras.utils.layer_utils import count_params
from utils.lr_schedules import StepDecay

import coral_ordinal as coral


class Classifier_XCM:

    def __init__(self, output_directory, input_shape, nb_classes, lr=0.001,
                 batch_size=16, verbose=2, nb_epochs=2000, depth=1,
                 filters=16, window=21, decay=False):

        input_shape = (None, None, input_shape[-1])
        self.output_directory = output_directory

        self.batch_size = batch_size
        self.verbose = verbose
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.depth = depth
        self.filters = filters
        self.window = window
        self.decay = decay

        self.model = self.build_model()

        trainable_count = count_params(self.model.trainable_weights)
        model_hyper = {'model': 'masked-xcm', 'filters': filters,
                       'depth': depth, 'window_size': window, 'decay': decay,
                       'batch_size': batch_size, 'classes': nb_classes,
                       'input_shape': input_shape, 'epochs': nb_epochs,
                       'trainable_params': trainable_count}

        f = open(os.path.join(self.output_directory, 'hyperparams.txt'), "w")
        f.write(str(model_hyper))
        f.close()

        return

    def build_model(self):
        input_layer = keras.layers.Input(batch_shape=self.input_shape)
        masked = keras.layers.Masking(mask_value=-1000)(input_layer)

        conv_2d = tf.keras.backend.expand_dims(masked, axis=-1)
        conv_1d = masked

        for i in range(self.depth):
            conv_2d = keras.layers.Conv2D(self.filters, (self.window, 1),
                                          padding='same')(conv_2d)
            conv_2d = keras.layers.Lambda((lambda x: x))(conv_2d,
                                                         mask=masked[:, :, 0])
            # print('after conv2d: {}'.format(conv_2d))
            conv_2d = keras.layers.BatchNormalization()(conv_2d)
            conv_2d = keras.layers.Activation(activation='relu')(conv_2d)\

        conv_2d = keras.layers.Conv2D(1, (1, 1), padding='same',
                                      name='conv2d-1x1',
                                      activation='relu')(conv_2d)
        conv_2d = keras.layers.Lambda((lambda x: x),
                                      name='cam')(conv_2d,
                                                  mask=masked[:, :, 0])

        print('after 1x1 conv2d: {}'.format(conv_2d))

        feats = tf.keras.backend.squeeze(conv_2d, -1)


        feats = keras.layers.Conv1D(2 * self.filters, self.window,
                                    padding='same', name='conv-final')(feats)
        feats = keras.layers.Lambda((lambda x: x),
                                    name='lambda_final')(feats,
                                                         mask=masked[:, :, 0])
        print('after conv1d: {}'.format(feats))
        feats = keras.layers.BatchNormalization()(feats)
        feats = keras.layers.Activation(activation='relu')(feats)
        print('before gap: {}'.format(feats))
        gap_layer = keras.layers.GlobalAveragePooling1D()(feats,
                                                          mask=masked[:, :, 0])

        output_layer = keras.layers.Dense(self.filters, use_bias=False)(gap_layer)
        output_layer = coral.CoralOrdinal(self.nb_classes)(output_layer)
        # output_layer = keras.layers.Dense(self.nb_classes,
        #                                   activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=coral.OrdinalCrossEntropy(num_classes=self.nb_classes),
                      optimizer=keras.optimizers.Adam(self.lr),
                      metrics=[coral.MeanAbsoluteErrorLabels()])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=0.75, patience=50,
                                                      min_lr=0.0001)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor='val_mean_absolute_error_labels',
            save_best_only=True, mode='min')


        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   restore_best_weights=True,
                                                   patience=200)

        schedule = StepDecay(initAlpha=self.lr, factor=0.85, dropEvery=20)
        lr_decay = keras.callbacks.LearningRateScheduler(schedule)

        self.callbacks = [reduce_lr, model_checkpoint, stop_early, lr_decay]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true='', class_weight=None):

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size,
                              epochs=self.nb_epochs, verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks, class_weight=class_weight)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        keras.backend.clear_session()

        return hist


    def predict(self, x_test, y_true, x_train, y_train, y_test,
                return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)

        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory +
                               'test_duration.csv', test_duration)
            return y_pred
