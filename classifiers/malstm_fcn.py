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

# from utils.layer_utils import AttentionLSTM


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes,
                 verbose=False, build=True, batch_size=64, lr=0.001,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, nb_epochs=2000, bottleneck_size=32):

        input_shape = (None, input_shape[-1])

        self.output_directory = output_directory

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


    def build_model(self, input_shape, nb_classes):

        print('building')
        ip = keras.layers.Input(input_shape)
        print('input', ip)
        mask = keras.layers.Masking(mask_value=-1000)(ip)
        print(mask)
        # # x = AttentionLSTM(8)(mask)
        # x = keras.layers.LSTM(8)(mask)#, mask=mask[:, :, 0])
        # print('lstm', x)
        # x = keras.layers.Attention()(x)
        # x = keras.layers.Dropout(0.8)(x)
        #
        # print(x)

        y = keras.layers.Permute((2, 1))(mask)
        y = keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        print(y)

        y = keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)

        y = keras.layers.GlobalAveragePooling1D()(y)

        # x = keras.layers.concatenate([x, y])
        x = y
        out = keras.layers.Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(ip, out)
        model.summary()

        # add load model code here to fine-tune

        optm = keras.optimizers.Adam(self.lr)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      actor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor='val_accuracy',
            save_best_only=True, mode='max')

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   restore_best_weights=True,
                                                   patience=300)

        schedule = StepDecay(initAlpha=self.lr, factor=0.75, dropEvery=20)
        lr_decay = keras.callbacks.LearningRateScheduler(schedule)

        self.callbacks = [reduce_lr, model_checkpoint, stop_early, lr_decay]

        return model

    def squeeze_excite_block(self, input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor

        Returns: a keras tensor
        '''
        filters = input._keras_shape[-1] # channel_axis = -1 for TF

        se = keras.layers.GlobalAveragePooling1D()(input)
        se = keras.layers.Reshape((1, filters))(se)
        se = keras.layers.Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = keras.layers.multiply([input, se])
        return se


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
