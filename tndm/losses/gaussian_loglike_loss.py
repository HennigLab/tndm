import tensorflow as tf
import math as m


def gaussian_loglike_loss(sigma=1.):
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    @tf.function
    def loss_fun(y_true, y_pred):
        # GAUSSIAN LOG-LIKELIHOOD
        beh = tf.cast(y_pred, dtype=tf.float32)
        targets = tf.cast(y_true, dtype=tf.float32)
        mse = 0.5 * \
            tf.reduce_sum(tf.keras.backend.square(
                (targets - beh)))
#         constant = tf.reduce_sum(tf.ones_like(
#             targets) * (sigma * tf.math.log(2 * m.pi)))
        return mse# + constant # -loglik
    return loss_fun