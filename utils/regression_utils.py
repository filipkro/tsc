import tensorflow as tf


class OneSidedMSE(tf.keras.losses.Loss):
    def __init__(self, min_val, max_val, name='one_side_MSE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def call(self, y_t, y_p):
        # loss = tf.zeros(y_t.shape)
        mask_lower = tf.math.equal(y_t, self.min_val)
        mask_greater = tf.math.equal(y_t, self.max_val)
        mask_mid = tf.math.logical_not(tf.math.logical_or(mask_lower, mask_greater))

        squared = tf.math.squared_difference(y_t, y_p)

        lower_loss = tf.cast(tf.math.logical_and(mask_lower, tf.math.greater(y_p, self.min_val)), tf.float32) * squared
        mid_loss = tf.cast(mask_mid, tf.float32) * squared
        greater_loss = tf.cast(tf.math.logical_and(mask_lower, tf.math.less(y_p, self.max_val)), tf.float32) * squared
        return tf.reduce_mean(lower_loss + mid_loss + greater_loss)
