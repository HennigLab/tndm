import tensorflow as tf
import numpy as np
import pytest
from tensorflow.python.ops.numpy_ops.np_array_ops import asarray

from latentneural import LFADS
from latentneural.lfads.adaptive_weights import AdaptiveWeights


@pytest.mark.unit
def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    model = LFADS(neural_space=50)
    model.build(input_shape=[None] + list(input_data.shape[1:]))
    
    log_f, (g0_r, r_mean, r_logvar), z, inputs = model.call(input_data, training=True)

    tf.debugging.assert_equal(log_f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))

    log_f, (g0_r, r_mean, r_logvar), z, inputs = model.call(input_data, training=False)

    tf.debugging.assert_equal(log_f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))

@pytest.mark.unit
def test_adaptive_weights():

    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons

    adaptive_weights = AdaptiveWeights(
        initial=[0.5, 1, 1],
        min_weight=[0., 0., 0.],
        max_weight=[1., 1., 1.],
        update_steps=[1, 2, 1],
        update_starts=[2, 1, 1],
        update_rates=[-0.05, -0.1, -0.01]
    )

    model = LFADS(neural_space=50, max_grad_norm=200)

    model.build(input_shape=[None] + list(input_data.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(x=input_data, y=None, callbacks=[adaptive_weights], shuffle=True, epochs=4)

    tf.debugging.assert_equal(adaptive_weights.w[0], 0.45)
    tf.debugging.assert_equal(adaptive_weights.w[1], 0.9)
