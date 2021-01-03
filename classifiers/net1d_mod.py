import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from keras import backend as K

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration
import os
from keras.utils.layer_utils import count_params


class Classifier_NET1d:

    def __init__(self, output_directory, input_shape, nb_classes, lr=0.001,
                 batch_size=16, verbose=2, nb_epochs=10000, depth=2,
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
        model_hyper = {'model': 'net1d or whatever', 'filters': filters,
                       'depth': depth, 'window_size': window, 'decay': decay,
                       'batch_size': batch_size, 'classes': nb_classes,
                       'input_shape': input_shape, 'epochs': nb_epochs,
                       'trainable_params': trainable_count}

        f = open(os.path.join(self.output_directory, 'hyperparams.txt'), "w")
        f.write(str(model_hyper))
        f.close()

        return

    def build_model(self):
        filters = []
        windows = []

        if isinstance(self.filters, int):

            for i in range(self.depth):
                if True:  # i < self.depth - 1:
                    filters.append(int(self.filters / (self.depth - i))) if \
                        self.decay else filters.append(int(self.filters))
                else:
                    filters.append(64)
                windows.append(int(self.window / (i + 1))) if self.decay else \
                    windows.append(int(self.window))

        elif len(self.filters) > 1:
            filters = self.filters
            if isinstance(self.window, int):
                windows = []
                for a in filters:
                    windows.append(self.window)
            elif len(self.window) == len(self.filters):
                windows = self.window
            else:
                windows = []
                for a in filters:
                    windows.append(self.window[0])

        input_layer = keras.layers.Input(batch_shape=self.input_shape)
        masked = keras.layers.Masking(mask_value=-1000,
                                      name='mask')(input_layer)
        conv_1d = masked
        # filters = [16, 32, 64]
        # windows = [71, 41, 11]




        for i in range(len(filters)):
            f = filters[i]
            w = windows[i]

            name_ending = str(i) if i < len(filters) - 1 else 'last'

            conv_1d = keras.layers.Conv1D(f * self.input_shape[-1], w,
                                          padding='same',
                                          groups=self.input_shape[-1],
                                          name='conv1d_{}'.format(name_ending)
                                          )(conv_1d)
            conv_1d = keras.layers.Lambda((lambda x: x),
                                          name='lambda1d_{}'.format(name_ending)
                                          )(conv_1d, mask=masked[:, :, 0])

            conv_1d = keras.layers.BatchNormalization(
                name='bn1d_{}'.format(name_ending))(conv_1d)
            conv_1d = keras.layers.Activation(
                activation='relu',
                name='relu1d_{}'.format(name_ending))(conv_1d)

        conv_1d = keras.layers.Conv1D(filters[-1], windows[-1],
                                      padding='same', name='conv1d-merge')(conv_1d)
        conv_1d = keras.layers.Lambda(
            (lambda x: x), name='lambda-merge')(conv_1d, mask=masked[:, :, 0])
        conv_1d = keras.layers.BatchNormalization(name='bn1d_merge')(conv_1d)
        conv_1d = keras.layers.Activation(activation='relu',
                                          name='relu1d_merge')(conv_1d)
        gap_layer = keras.layers.GlobalAveragePooling1D(name='gap')(conv_1d,
                                                                    mask=masked[:, :, 0])

        #output_layer = keras.layers.Dense(self.nb_classes,
        #                                  name='result')(gap_layer)

        output_layer = keras.layers.Dense(filters[-1],
                                           name='result1')(gap_layer)
        output_layer = keras.layers.Dense(self.nb_classes,
                                           name='result2')(output_layer)
        output_layer = keras.layers.Activation(activation='softmax',
                                               name='sm')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=0.75, patience=50,
                                                      min_lr=0.0001)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor='val_loss',
            save_best_only=True, mode='min')

        check_folder = self.output_directory + \
            'checkpoints/cp-{epoch:04d}.hdf5'
        rec_checkpoints = keras.callbacks.ModelCheckpoint(
            filepath=check_folder, save_freq='epoch', period=10)

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   restore_best_weights=True,
                                                   patience=300)

        self.callbacks = [reduce_lr, model_checkpoint, stop_early,
                          rec_checkpoints]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true='', class_weight=None):
        # if not tf.test.is_gpu_available:
        #     print('error no gpu')
        #     exit()
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
