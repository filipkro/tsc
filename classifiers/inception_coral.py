import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pickle

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration
import os
from keras.utils.layer_utils import count_params
from utils.lr_schedules import StepDecay

import coral_ordinal as coral


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes,
                 verbose=False, build=True, batch_size=64, lr=0.001,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, nb_epochs=2000, bottleneck_size=32,
                 class_weight=None):

        self.output_directory = output_directory

        self.loss = coral.OrdinalCrossEntropy(num_classes=nb_classes,
                                              importance_weights=class_weight)

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        trainable_count = count_params(self.model.trainable_weights)

        model_hyper = {'model': 'masked-inception', 'filters': nb_filters,
                       'residuals': use_residual, 'bottleneck': use_bottleneck,
                       'depth': depth, 'kernel_size': kernel_size,
                       'batch_size': batch_size, 'epochs': nb_epochs,
                       'bottleneck_size': self.bottleneck_size,
                       'classes': nb_classes, 'input_shape': input_shape,
                       'trainable_params': trainable_count}

        f = open(os.path.join(self.output_directory, 'hyperparams.txt'), "w")
        f.write(str(model_hyper))
        f.close()

    def _inception_module(self, input_tensor, masked, stride=1, activation='linear'):

        input_tensor = keras.layers.Lambda((lambda x: x))(input_tensor,
                                                          mask=masked[:, :, 0])
        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size,
                                                  kernel_size=1, padding='same',
                                                  activation=activation,
                                                  use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_l = []

        input_inception = keras.layers.Lambda((lambda x: x))(input_inception,
                                                             mask=masked[:, :, 0])

        for i in range(len(kernel_size_s)):
            conv_l.append(keras.layers.Conv1D(filters=self.nb_filters,
                                              kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same',
                                              activation=activation,
                                              use_bias=False)(input_inception))

        max_pool_1 = keras.layers.MaxPool1D(
            pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
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

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        masked_layer = keras.layers.Masking(mask_value=-1000,
                                            name='mask')(input_layer)
        x = masked_layer
        input_res = masked_layer

        for d in range(self.depth):

            x = self._inception_module(x, masked_layer)

            if self.use_residual and d % 3 == 2:
                input_res = keras.layers.Lambda((lambda x: x))(input_res,
                                                               mask=masked_layer[:, :, 0])
                x = self._shortcut_layer(input_res, x)
                input_res = x


        x = keras.layers.Dropout(0.2)(x)
        gap_layer = keras.layers.GlobalAveragePooling1D()(
            x, mask=masked_layer[:, :, 0])

        output_layer = keras.layers.Dense(self.nb_filters,
                                          name='result1')(gap_layer)
        output_layer = keras.layers.Dense(self.nb_filters, use_bias=False)(output_layer)
        output_layer = coral.CoralOrdinal(nb_classes)(output_layer)

        # model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model = keras.models.Model(inputs=input_layer,
                                   outputs=output_layer)

        model.compile(loss=self.loss,
                      optimizer=keras.optimizers.Adam(self.lr),
                      metrics=[coral.MeanAbsoluteErrorLabels()])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      actor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor='val_mean_absolute_error_labels',
            save_best_only=True, mode='min')

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   restore_best_weights=True,
                                                   patience=300)

        schedule = StepDecay(initAlpha=self.lr, factor=0.85, dropEvery=20)
        lr_decay = keras.callbacks.LearningRateScheduler(schedule)

        self.callbacks = [reduce_lr, model_checkpoint, stop_early, lr_decay]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true='', class_weight=None):
        if not tf.test.is_gpu_available:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size,
                              epochs=self.nb_epochs, verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks,
                              class_weight=class_weight)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        keras.backend.clear_session()

        return hist

        # return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test,
                return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred, cam = model.predict(x_test, batch_size=self.batch_size)

        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory +
                               'test_duration.csv', test_duration)
            return y_pred, cam
