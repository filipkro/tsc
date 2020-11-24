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


class Classifier_XCM:

    def __init__(self, output_directory, input_shape, nb_classes, lr=0.001,
                 batch_size=16, verbose=2, nb_epochs=2000, depth=1,
                 filters=16, window=21, decay=False):

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
        filters = []
        windows = []

        for i in range(self.depth):
            filters.append(int(self.filters / (depth - i))) if self.decay else \
                filters.append(int(self.filters))
            windows.append(int(self.window / (i + 1))) if self.decay else \
                windows.append(int(self.window))

        input_layer = keras.layers.Input(self.input_shape)
        masked = keras.layers.Masking(mask_value=-1000,
                                      name='mask')(input_layer)

        conv_2d = tf.keras.backend.expand_dims(masked, axis=-1)
        conv_1d = masked

        # for f, w in zip(filters, windows):
        for i in range(len(filters)):
            f = filters[i]
            w = windows[i]
            name_ending = str(i) if i < len(filters) - 1 else 'last'
            conv_2d = keras.layers.Conv2D(f, (w, 1), padding='same',
                                          name='conv2d_{}'.format(name_ending))(conv_2d)
            print('after conv2d: {}'.format(conv_2d))
            conv_2d = keras.layers.BatchNormalization(
                name='bn2d_{}'.format(name_ending))(conv_2d)
            conv_2d = keras.layers.Activation(activation='relu',
                                              name='relu2d_{}'.format(name_ending))(conv_2d)

            conv_1d = keras.layers.Conv1D(f, w, padding='same',
                                          name='conv1d_{}'.format(name_ending))(conv_1d)
            print('after conv1d: {}'.format(conv_1d))
            conv_1d = keras.layers.BatchNormalization(
                name='bn1d_{}'.format(name_ending))(conv_1d)
            conv_1d = keras.layers.Activation(activation='relu',
                                              name='relu1d_{}'.format(name_ending))(conv_1d)

            conv_2d = keras.layers.Lambda((lambda x: x), name='lambda2d_{}'.format(
                name_ending))(conv_2d, mask=masked[:, :, 0])
            conv_1d = keras.layers.Lambda((lambda x: x), name='lambda1d_{}'.format(
                name_ending))(conv_1d, mask=masked[:, :, 0])

        conv_2d = keras.layers.Conv2D(1, (1, 1), padding='same',
                                      name='conv2d-1x1',
                                      activation='relu')(conv_2d)
        print('after 1x1 conv2d: {}'.format(conv_2d))

        conv_1d = keras.layers.Conv1D(1, 1, padding='same',
                                      name='conv1d-1x1',
                                      activation='relu')(conv_1d)
        # conv_1d = tf.keras.backend.expand_dims(conv_1d, axis=-1,
        #                                        name='exp_dims1d')
        conv_2d = tf.keras.backend.squeeze(conv_2d, -1)  # , name='squeeze2d')
        print('after 1x1 conv1d: {}'.format(conv_1d))

        conv_2d = keras.layers.Lambda((lambda x: x),
                                      name='lambda2d-final')(conv_2d,
                                                             mask=masked[:, :, 0])
        conv_1d = keras.layers.Lambda((lambda x: x),
                                      name='lambda1d-final')(conv_1d,
                                                             mask=masked[:, :, 0])
        feats = keras.layers.Concatenate(axis=2,
                                         name='concat')([conv_2d, conv_1d])
        # feats = tf.keras.backend.squeeze(feats, -1)
        feats = keras.layers.Conv1D(self.filters, self.window,
                                    padding='same', name='conv-final')(feats)
        print('after conv1d: {}'.format(feats))
        feats = keras.layers.BatchNormalization(name='bn-final')(feats)
        feats = keras.layers.Activation(activation='relu',
                                        name='relu-final')(feats)

        print('before gap: {}'.format(feats))
        gap_layer = keras.layers.GlobalAveragePooling1D(name='gap')(feats,
                                                                    mask=masked[:, :, 0])
        output_layer = keras.layers.Dense(self.nb_classes,
                                          name='result')(gap_layer)
        output_layer = keras.layers.Activation(activation='softmax',
                                               name='sm')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      actor=0.5, patience=50,
                                                      min_lr=0.0001)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor='val_accuracy',
            save_best_only=True, mode='max')

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
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
                              callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)
        # np.save(self.output_directory + 'cam.npy', cam)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory,
                               hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

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
