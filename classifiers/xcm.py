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
                 batch_size=16, verbose=2, nb_epochs=2000):

        self.output_directory = output_directory

        self.batch_size = batch_size
        self.verbose = verbose
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.input_shape = input_shape
        self.nb_classes = nb_classes

        self.model = self.build_model()

    def build_model(self):
        F = 16
        W = 21
        input_layer = keras.layers.Input(self.input_shape)
        input_2d = tf.keras.backend.expand_dims(input_layer, axis=-1)

        time_feats = keras.layers.Conv2D(F, (W, 1), padding='same')(input_2d)
        print('after first conv2d: {}'.format(time_feats))
        time_feats = keras.layers.BatchNormalization()(time_feats)
        time_feats = keras.layers.Activation(activation='relu')(time_feats)
        time_feats = keras.layers.Conv2D(1, (1, 1), padding='same')(time_feats)
        print('after 1x1 conv2d: {}'.format(time_feats))

        inp_feats = keras.layers.Conv1D(F, W, padding='same')(input_layer)
        print('after first conv1d: {}'.format(inp_feats))
        inp_feats = keras.layers.BatchNormalization()(inp_feats)
        inp_feats = keras.layers.Activation(activation='relu')(inp_feats)
        inp_feats = keras.layers.Conv1D(1, 1, padding='same')(inp_feats)
        print('after 1x1 conv1d: {}'.format(inp_feats))
        inp_feats = tf.keras.backend.expand_dims(inp_feats, axis=-1)

        feats = keras.layers.Concatenate(axis=2, name='last_feat')([time_feats,
                                                                    inp_feats])
        print('after concat: {}'.format(feats))
        feats = tf.keras.backend.squeeze(feats, -1)
        print('after squeeze: {}'.format(feats))

        feats = keras.layers.Conv1D(F, W, padding='same')(feats)
        print('after conv1d: {}'.format(feats))
        feats = keras.layers.BatchNormalization()(feats)
        feats = keras.layers.Activation(activation='relu')(feats)

        print('before gap: {}'.format(feats))
        gap_layer = keras.layers.GlobalAveragePooling1D()(
            feats)  # , mask=masked_layer[:,:,0])
        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax',
                                          name='result')(gap_layer)

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
