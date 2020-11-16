# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics


class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, nb_layers=4,
                 kernel_size=21, filters=64, verbose=2, build=True):
        self.output_directory = output_directory
        self.nb_layers = nb_layers
        self.kernel_size = kernel_size
        self.filters = filters

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes,
                    channel_order='channels_first'):

        kernel_size_s = [int(self.kernel_size // i) for i
                         in np.linspace(self.kernel_size, 1, self.nb_layers)]
        print('building model')
        input_layer = keras.layers.Input(input_shape)

        x = keras.layers.Masking(mask_value=-1000)(input_layer)

        for kern in kernel_size_s:
            x = keras.layers.Conv1D(filters=self.filters, kernel_size=kern,
                                    padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        cam = keras.layers.GlobalAveragePooling1D(data_format='channels_first', name='cam')(x)

        output_layer = keras.layers.Dense(nb_classes,
                                          activation='softmax', name='result')(gap_layer)

        # model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        model = keras.models.Model(inputs=input_layer,
                                   outputs=[output_layer, cam])

        model.compile(loss=['categorical_crossentropy', None],
                      optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        print('TRAINING')
        if not tf.test.is_gpu_available:
            print('error')
            # exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 256
        nb_epochs = 2000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        # hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size,
        #                       epochs=nb_epochs, verbose=self.verbose,
        #                       validation_data=(x_val, y_val),
        #                       callbacks=self.callbacks)

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size,
                              epochs=nb_epochs, verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(
            self.output_directory + 'best_model.hdf5')

        y_pred, gap = model.predict(x_val)

        print(gap.shape)
        # print(hist.accuracy)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)
        print(y_pred)
        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

    def predict(self, x_test, y_true, x_train, y_train, y_test,
                return_df_metrics=True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)

        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
