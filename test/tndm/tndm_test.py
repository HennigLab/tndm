import tensorflow as tf
import numpy as np
import pytest

from latentneural import TNDM
from latentneural.tndm.adaptive_weights import AdaptiveWeights


@pytest.mark.unit
def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    model = TNDM(neural_space=50, behavioural_space=2)
    model.build(input_shape=[None] + list(input_data.shape[1:]))
    
    f, b, (g0_r, r_mean, r_logvar), (g0_i, i_mean, i_logvar), (z_r, z_i), inputs = model.call(input_data, training=True)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 2]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_i.shape, tf.TensorShape([10, 64]))

    f, b, (g0_r, r_mean, r_logvar), (g0_i, i_mean, i_logvar), (z_r, z_i), inputs = model.call(input_data, training=False)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 2]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_i.shape, tf.TensorShape([10, 64]))

@pytest.mark.unit
def test_adaptive_weights():
    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    input_n_data = np.exp(np.random.randn(10, 100, 10)) # trials X time X behaviour

    adaptive_weights = AdaptiveWeights(
        initial=[0.5, 1, 1, 0, 0],
        min_weight=[0., 0., 0., 0, 0],
        max_weight=[1., 1., 1., 0, 0],
        update_steps=[1, 2, 1, 1, 1],
        update_starts=[2, 1, 1, 0, 0],
        update_rates=[-0.05, -0.1, -0.01, 0, 0]
    )

    model = TNDM(neural_space=50, behavioural_space=10, max_grad_norm=200)

    model.build(input_shape=[None] + list(input_data.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(x=input_data, y=input_n_data, callbacks=[adaptive_weights], shuffle=True, epochs=4)

    tf.debugging.assert_equal(adaptive_weights.w[0], 0.45)
    tf.debugging.assert_equal(adaptive_weights.w[1], 0.9)
