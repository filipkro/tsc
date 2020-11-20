# resnet model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
from utils.utils import calculate_metrics
from utils.utils import save_logs
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pickle
import os

import matplotlib
from utils.utils import save_test_duration
from keras.utils.layer_utils import count_params

matplotlib.use('agg')


class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes,
                 verbose=False, build=True, load_weights=False,
                 n_feature_maps=64, depth=3, nb_epochs=1500, batch_size=64):
        self.output_directory = output_directory
        self.n_feature_maps = n_feature_maps
        self.depth = depth
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(
                    self.output_directory + 'model_init.hdf5')

        trainable_count = count_params(self.model.trainable_weights)
        model_hyper = {'model': 'masked-resnet', 'classes': nb_classes,
                       'input_shape': input_shape, 'depth': depth,
                       'feature_maps': n_feature_maps, 'epochs': nb_epochs,
                       'trainable_params': trainable_count,
                       'batch_size': batch_size}

        f = open(os.path.join(self.output_directory, 'hyperparams.txt'), "w")
        f.write(str(model_hyper))
        f.close()

        return

    def build_model(self, input_shape, nb_classes):

        input_layer = keras.layers.Input(input_shape)
        masked_layer = keras.layers.Masking(mask_value=-1000,
                                            name='mask')(input_layer)
        x = masked_layer
        # BLOCK 1
        for i in range(self.depth):
            conv_x = keras.layers.Conv1D(filters=self.n_feature_maps,
                                         kernel_size=8,
                                         padding='same')(x)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)

            conv_y = keras.layers.Conv1D(filters=self.n_feature_maps,
                                         kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)

            conv_z = keras.layers.Conv1D(filters=self.n_feature_maps,
                                         kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=self.n_feature_maps,
                                             kernel_size=1,
                                             padding='same')(input_layer)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_1 = keras.layers.add([shortcut_y, conv_z])
            x = keras.layers.Activation('relu')(output_block_1)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(x, mask=masked_layer[:,:,0])
        cam = keras.layers.GlobalAveragePooling1D(data_format='channels_first',
                                                  name='cam')(x)

        output_layer = keras.layers.Dense(nb_classes, name='result',
                                          activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer,
                                   outputs=[output_layer, cam, masked_layer])

        model.compile(loss=['categorical_crossentropy', None, None],
                      optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor='val_result_accuracy',
            save_best_only=True, mode='max')

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        mini_batch_size = int(min(x_train.shape[0] / 10, self.batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred, cam, mask = self.model.predict(x_val)

        # y_pred, cam, mask = self.predict(x_val, y_true, x_train, y_train, y_val,
        #                       return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory,
                               hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred, cam, mask = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory +
                               'test_duration.csv', test_duration)
            return y_pred, cam, mask
