import tensorflow as tf

class ConfusionCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, U, name='confidence_crossent', **kwargs):
        super().__init__(name=name, **kwargs)
        # self.U = tf.convert_to_tensor(U, dtype=tf.float32)
        self.U = U

    def confusion_cross_entropy(self, y_t, y_p):
        # print(self.U)
        # tf.print(f'y_t: {y_t}')
        # tf.print('yt')
        # tf.print(y_t)
        # tf.print('yp')
        # tf.print(y_p)
        # tf.print(-tf.reduce_sum(self.U * tf.math.log(tf.matmul(y_t+0.001, y_p, True, False))))
        return -tf.reduce_sum(self.U * tf.math.log(tf.matmul(y_t+0.0001, y_p, True, False)))

    def confusion_square(self, y_t, y_p):
        return tf.reduce_sum(tf.math.square(self.U - tf.matmul(y_t, y_p, True, False)))

    def call(self, y_t, y_p):
        # tf.print(f'y_t: {y_t}')
        # tf.print(f'yp: {y_p}')
        # return self.confusion_square(y_t, y_p)
        return self.confusion_cross_entropy(y_t, y_p)

    def get_config(self):
        config = {
            "U": self.U,
        }
        base_config = super().get_config()
        return {**base_config, **config}
