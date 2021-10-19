import math as m
import tensorflow as tf
from tndm.losses import mse_loss
from numpy import random
import pytest

@pytest.mark.unit
def test_mse_loss():
    
    def old_gaussian_loglike_loss():
        @tf.function
        def loss_fun(y_true, y_pred):
            # GAUSSIAN LOG-LIKELIHOOD
            b = tf.cast(y_pred, dtype=tf.float32)
            targets = tf.cast(y_true, dtype=tf.float32)
            mse = 0.5 * \
                tf.reduce_sum(tf.keras.backend.square(
                    (targets - b)))
            return mse
        return loss_fun

    ll_new = mse_loss()
    ll_old = old_gaussian_loglike_loss()
    random.seed(42)
    y_true = tf.convert_to_tensor(random.randn(50,100,20))
    y_pred = tf.convert_to_tensor(random.randn(50,100,20))
    tf.debugging.assert_near(ll_new(y_true,y_pred),ll_old(y_true,y_pred),atol=1e-2)