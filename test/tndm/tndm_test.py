import tensorflow as tf
import numpy as np
import pytest

from latentneural import TNDM


@pytest.mark.unit
def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    model = TNDM(neural_space=50)
    model.build(input_shape=[None] + list(input_data.shape[1:]))
    
    f, b, (g0_r, r_mean, r_logvar), (g0_i, i_mean, i_logvar) = model.call(input_data, training=True)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 1]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_i.shape, tf.TensorShape([10, 64]))

    f, b, (g0_r, r_mean, r_logvar), (g0_i, i_mean, i_logvar) = model.call(input_data, training=False)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 1]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_i.shape, tf.TensorShape([10, 64]))