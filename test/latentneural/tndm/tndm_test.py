import tensorflow as tf
import numpy as np

from latentneural import TNDM


def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    model = TNDM(neural_space=50)
    model.build(input_shape=[None] + list(input_data.shape[1:]))
    
    b, f, (r_mean, r_logvar), (i_mean, i_logvar) = model.call(input_data)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 1]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))