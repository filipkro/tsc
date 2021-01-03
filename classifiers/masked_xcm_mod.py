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


class Classifier_XCM:

    def __init__(self, output_directory, input_shape, nb_classes, lr=0.001,
                 batch_size=16, verbose=2, nb_epochs=2000, depth=1, mask=True,
                 filters=16, window=21, decay=False, metric='val_accuracy',
                 reduce_lr_metric='train_loss'):

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
        self.mask = mask

        if metric == 'train_loss':
            self.metric = 0
        elif metric == 'train_accuracy':
            self.metric = 1
        elif metric == 'val_loss':
            self.metric = 2
        else:
            self.metric = 3

        if reduce_lr_metric == 'train_accuracy':
            self.reduce_lr_metric = 1
        elif reduce_lr_metric == 'val_loss':
            self.reduce_lr_metric = 2
        elif reduce_lr_metric == 'val_accuracy':
            self.reduce_lr_metric = 3
        else:
            self.reduce_lr_metric = 0

        self.model = self.build_model()

        trainable_count = count_params(self.model.trainable_weights)
        model_hyper = {'model': 'masked-xcm', 'filters': filters, 'mask': mask,
                       'depth': depth, 'window_size': window, 'decay': decay,
                       'batch_size': batch_size, 'classes': nb_classes,
                       'input_shape': input_shape, 'epochs': nb_epochs,
                       'trainable_params': trainable_count, 'metric': metric,
                       'lr_metric': reduce_lr_metric}

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
                    filters.append(int(self.filters / (self.depth - i))) if self.decay else \
                        filters.append(int(self.filters))
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

        conv_2d = tf.keras.backend.expand_dims(masked, axis=-1)
        conv_1d = masked
        #conv_2d = tf.keras.backend.expand_dims(input_layer, axis=-1)
        #conv_1d = input_layer

        # for f, w in zip(filters, windows):
        # filters = [16, 32, 64]
        # windows = [71, 41, 11]
        for i in range(len(filters)):
            f = filters[i]
            w = windows[i]
            name_ending = str(i) if i < len(filters) - 1 else 'last'
            conv_2d = keras.layers.Conv2D(f, (w, 1), padding='same',
                                          name='conv2d_{}'.format(name_ending))(conv_2d)
            conv_2d = keras.layers.Lambda((lambda x: x), name='lambda2d_{}'.format(
                name_ending))(conv_2d, mask=masked[:, :, 0])
            print('after conv2d: {}'.format(conv_2d))
            conv_2d = keras.layers.BatchNormalization(
                name='bn2d_{}'.format(name_ending))(conv_2d)
            conv_2d = keras.layers.Activation(activation='relu',
                                              name='relu2d_{}'.format(name_ending))(conv_2d)

            conv_1d = keras.layers.Conv1D(f, w, padding='same',
                                          name='conv1d_{}'.format(name_ending))(conv_1d)
            conv_1d = keras.layers.Lambda((lambda x: x), name='lambda1d_{}'.format(
                name_ending))(conv_1d, mask=masked[:, :, 0])
            print('after conv1d: {}'.format(conv_1d))
            conv_1d = keras.layers.BatchNormalization(
                name='bn1d_{}'.format(name_ending))(conv_1d)
            conv_1d = keras.layers.Activation(activation='relu',
                                              name='relu1d_{}'.format(name_ending))(conv_1d)

            # conv_2d = keras.layers.Lambda((lambda x: x), name='lambda2d_{}'.format(
            #     name_ending))(conv_2d, mask=masked[:, :, 0])
            # conv_1d = keras.layers.Lambda((lambda x: x), name='lambda1d_{}'.format(
            #     name_ending))(conv_1d, mask=masked[:, :, 0])

        conv_2d = keras.layers.Conv2D(1, (1, 1), padding='same',
                                      name='conv2d-1x1',
                                      activation='relu')(conv_2d)
        conv_2d = keras.layers.Lambda((lambda x: x),
                                      name='lambda2d_1x1')(conv_2d,
                                                           mask=masked[:, :, 0])

        print('after 1x1 conv2d: {}'.format(conv_2d))

        conv_1d = keras.layers.Conv1D(1, 1, padding='same',
                                      name='conv1d-1x1',
                                      activation='relu')(conv_1d)
        conv_1d = keras.layers.Lambda((lambda x: x),
                                      name='lambda1d_1x1')(conv_1d,
                                                           mask=masked[:, :, 0])
        #conv_1d = tf.keras.backend.expand_dims(conv_1d, axis=-1, name='exp_dims1d')
        conv_2d = tf.keras.backend.squeeze(conv_2d, -1)  # , name='squeeze2d')
        print('after 1x1 conv1d: {}'.format(conv_1d))

        # conv_2d = keras.layers.Lambda((lambda x: x),
        #                               name='lambda2d-final')(conv_2d,
        #                                                      mask=masked[:, :, 0])
        # conv_1d = keras.layers.Lambda((lambda x: x),
        #                               name='lambda1d-final')(conv_1d,
        #                                                      mask=masked[:, :, 0])
        feats = keras.layers.Concatenate(axis=2,
                                         name='concat')([conv_2d, conv_1d])
        # feats = tf.keras.backend.squeeze(feats, -1)
        #feats = keras.layers.Conv1D(filters[-1], self.window, padding='same', name='conv-final')(conv_2d)

        feats = keras.layers.Conv1D(256, windows[-1],
                                    padding='same', name='conv-final')(feats)
        feats = keras.layers.Lambda((lambda x: x),
                                    name='lambda_final')(feats,
                                                         mask=masked[:, :, 0])
        print('after conv1d: {}'.format(feats))
        feats = keras.layers.BatchNormalization(name='bn-final')(feats)
        feats = keras.layers.Activation(activation='relu',
                                        name='relu-final')(feats)
        print('before gap: {}'.format(feats))
        gap_layer = keras.layers.GlobalAveragePooling1D(name='gap')(feats,
                                                                    mask=masked[:, :, 0])
        #gap_layer = keras.layers.GlobalAveragePooling1D(name='gap')(feats)
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

        if not self.mask:
            self.fit_wo_mask(x_train, y_train, x_val, y_val, y_true)
        else:
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

    def fit_wo_mask(self, x_train, y_train, x_val, y_val, y_true):
        print('WARNING: Will train with batch size = 1 to avoid masks')
        print(x_val.shape[0])
        self.old_metric = -100
        self.old_epoch = 0
        self.best_metric = -100

        best_acc = 0.0
        best_loss = 100
        history = np.zeros((self.nb_epochs, 4))
        filewrites = 0
        start_time = time.time()
        for epoch_nb in range(self.nb_epochs):
            train_accuracy = 0
            train_loss = 0

            for i in range(x_train.shape[0]):
                max_idx = np.where(x_train[i, :, 0] < -900)[0][0]
                x = np.expand_dims(x_train[i, :max_idx, ...], 0)
                y = np.expand_dims(y_train[i], 0)
                train = self.model.train_on_batch(x, y, reset_metrics=True)
                train_accuracy += train[1]
                train_loss += train[0]
                # print(train)
            train_accuracy /= x_train.shape[0]
            train_loss /= x_train.shape[0]
            val_accuracy = 0
            val_loss = 0

            for i in range(x_val.shape[0]):
                max_idx = np.where(x_val[i, :, 0] < -900)[0][0]
                x = np.expand_dims(x_val[i, :max_idx, ...], 0)
                y = np.expand_dims(y_val[i], 0)

                val = self.model.test_on_batch(x, y, reset_metrics=True)
                # print(val)
                val_accuracy += val[1]
                val_loss += val[0]

            val_accuracy /= x_val.shape[0]
            val_loss /= x_val.shape[0]

            history[epoch_nb, :] = np.array([-train_loss, train_accuracy,
                                             -val_loss, val_accuracy])
            self.reduce_lr(epoch_nb, history[epoch_nb, self.reduce_lr_metric])
            if self.eval_epoch(history[epoch_nb, self.metric]):
                best_metrics = np.array([[epoch_nb, train_loss, train_accuracy,
                                          val_loss, val_accuracy]])
                filewrites += 1

            print('*********************************')
            print('Epoch: {}/{}: Training, loss: {:.3f}, accuracy: {:.3f}\nValidation, loss: {:.3f}, accuracy: {:.3f}'.format(
                epoch_nb, self.nb_epochs, train_loss, train_accuracy, val_loss, val_accuracy))
            print('*********************************')

        duration = time.time() - start_time
        self.model.save(self.output_directory + 'last_model.hdf5')
        self.best_model.save(self.output_directory + 'best_model.hdf5')
        history[:, [0, 2]] = -history[:, [0, 2]]
        np.savetxt(self.output_directory + 'history.csv', history,
                   header='Train loss,Train acc,Val loss,Val acc',
                   delimiter=',')
        np.savetxt(self.output_directory + 'best_model.csv', best_metrics,
                   header='Epoch,Train loss,Train acc,Val loss,Val acc',
                   delimiter=',')
        print('Number of file writes due to new best model: {}'.format(filewrites))
        print('Best model: {}'.format(best_metrics))
        print('Training time: {:.3f}'.format(duration))
        print('DONE')

    def reduce_lr(self, epoch, new_metric):
        min_lr = 0.0001
        patience = 50
        factor = 0.75
        if K.eval(self.model.optimizer.lr) > min_lr:
            if new_metric == self.old_metric:
                if self.old_epoch + patience < epoch:
                    K.set_value(self.model.optimizer.learning_rate,
                                K.eval(self.model.optimizer.lr * factor))
                    self.old_metric = new_metric
                    self.old_epoch = epoch
            else:
                self.old_metric = new_metric
                self.old_epoch = epoch

    def eval_epoch(self, new_metric):
        if new_metric > self.best_metric:
            self.best_metric = new_metric
            # self.model.save(self.output_directory + 'best_model.hdf5')
            self.best_model = keras.models.clone_model(self.model)
            self.best_model.set_weights(self.model.get_weights())
            return True
        else:
            return False

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
