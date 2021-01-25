import tensorflow as tf

class ConfusionCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, U, name='confidence_crossent'):
        super().__init__(name=name)
        # self.U = tf.convert_to_tensor(U, dtype=tf.float32)
        self.U = U

    def confusion_cross_entropy(self, y_t, y_p):
        return -tf.reduce_sum(self.U * tf.math.log(tf.matmul(y_t, y_p, True, False)))

    def call(self, y_t, y_p):
        return self.confusion_cross_entropy(y_t, y_p)

    def get_config(self):
        config = {
            "U": self.U,
        }
        base_config = super().get_config()
        return {**base_config, **config}