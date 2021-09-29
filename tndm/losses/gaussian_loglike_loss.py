from typing import List
import tensorflow as tf
import math as m


def gaussian_loglike_loss(arg_idx: List[int]):
    means_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in arg_idx]))
    @tf.function
    def loss_fun(y_true, y_pred):
        # GAUSSIAN LOG-LIKELIHOOD
        b = means_getter(y_pred)
        targets = tf.cast(y_true, dtype=tf.float32)
        mse = 0.5 * \
            tf.reduce_sum(tf.keras.backend.square(
                (targets - b)))
        constant = tf.reduce_sum(tf.ones_like(
            b) * (0.5 * tf.math.log(2 * m.pi)))
        return mse + constant
    return loss_fun