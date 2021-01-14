import tensorflow as tf
import numpy as np
from keras import backend as K


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    """
    A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
    @url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    @author: wassname
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def train_loop(classifier, x_train, y_train, x_val, y_val, class_weight=None):
    expand = False
    if hasattr(classifier, 'loss'):
        loss_fn = classifier.loss
        # elif class_weight is not None:
        #     loss_fn = weighted_categorical_crossentropy(class_weight)
        #     expand = True
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        expand = True

    batch = 8#classifier.batch_size

    for epoch in range(classifier.nb_epochs):
        print("\nStart of epoch %d" % (epoch + 1,))

        train_vars = classifier.model.trainable_variables
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
        total_loss = 0
        last_opt = 0
        corr = 0

        # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        for step, (x, y) in enumerate(zip(x_train, y_train)):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # print(x)
                # print(y)
                # print(x.shape)
                x = x[np.where(x[:,0] > -900), :]
                logits = classifier.model(x, training=True)  # Logits for this minibatch
                if expand:
                    y = np.expand_dims(y, 0)
                loss_value = loss_fn(y, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, classifier.model.trainable_weights)
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads)]
            total_loss += loss_value
            corr += np.argmax(y) == np.argmax(logits)
            # print(grads.shape)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            if step % batch == 0:
                # accum_gradient = [this_grad/num_samples for this_grad in accum_gradient]
                accum_gradient = [this_grad/batch for this_grad in accum_gradient]
                classifier.model.optimizer.apply_gradients(zip(accum_gradient, classifier.model.trainable_weights))
                last_opt = step
                accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
                # batch_loss = total_loss / batch

                # print(f"Batch training loss: {batch_loss}")
        since_last = step - last_opt
        if since_last > 0:
            accum_gradient = [this_grad/since_last for this_grad in accum_gradient]
            classifier.model.optimizer.apply_gradients(zip(accum_gradient, classifier.model.trainable_weights))

        print(f"Traning loss for epoch {epoch + 1}: {total_loss/step}")
        print(f"Training accuracy after epoch {epoch + 1}: {corr/step}")
        total_loss = 0
        corr = 0
        for x, y in zip(x_val, y_val):
            x = x[np.where(x[:,0] > -900), :]
            logits = classifier.model(x, training=True)
            if expand:
                y = np.expand_dims(y, 0)

            total_loss += loss_fn(y, logits)
            corr += np.argmax(y) == np.argmax(logits)
            # print(y)
            # print(logits)

        print(f"Validation loss after epoch {epoch + 1}: {total_loss/y_val.shape[0]}")
        print(f"Validation accuracy after epoch {epoch + 1}: {corr/y_val.shape[0]}")

        #
        # # Log every 200 batches.
        # if step % 2 == 0:
        #     print(
        #         "Training loss (for one batch) at step %d: %.4f"
        #         % (step, float(loss_value))
        #     )
        #     print("Seen so far: %s samples" % ((step + 1) * 64))
