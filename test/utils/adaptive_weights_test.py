import tensorflow as tf
import numpy as np
import pytest

from tndm import TNDM, LFADS
from tndm.utils import AdaptiveWeights


@pytest.mark.unit
def test_adaptive_weights():
    input_data = np.exp(np.random.randn(10, 100, 50)
                        )  # trials X time X neurons
    input_n_data = np.exp(np.random.randn(10, 100, 10)
                          )  # trials X time X behaviour

    adaptive_weights = AdaptiveWeights(
        initial=[0.5, 1, 1, 0, 1, 0],
        min_weight=[0., 0., 0., 0, 0, 0],
        max_weight=[1., 1., 1., 0, 0, 0],
        update_step=[1, 2, 1, 1, 1, 1],
        update_start=[2, 1, 1, 0, 0, 0],
        update_rate=[-0.05, -0.1, -0.01, 0, 0, 0]
    )

    model = TNDM(neural_dim=50, behaviour_dim=10, max_grad_norm=200)

    model.build(input_shape=[None] + list(input_data.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(
        x=input_data,
        y=input_n_data,
        callbacks=[adaptive_weights],
        shuffle=True,
        epochs=4)

    tf.debugging.assert_equal(adaptive_weights.w[0], 0.45)
    tf.debugging.assert_equal(adaptive_weights.w[1], 0.9)


@pytest.mark.unit
def test_adaptive_weights():

    input_data = np.exp(np.random.randn(10, 100, 50)
                        )  # trials X time X neurons

    adaptive_weights = AdaptiveWeights(
        initial=[0.5, 1, 1],
        min_weight=[0., 0., 0.],
        max_weight=[1.],
        update_step=[1, 2, 1],
        update_start=[2, 1, 1],
        update_rate=[-0.05, -0.1, -0.01]
    )

    model = LFADS(neural_dim=50, max_grad_norm=200)

    model.build(input_shape=[None] + list(input_data.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(
        x=input_data,
        y=None,
        callbacks=[adaptive_weights],
        shuffle=True,
        epochs=4)

    tf.debugging.assert_equal(adaptive_weights.w[0], 0.45)
    tf.debugging.assert_equal(adaptive_weights.w[1], 0.9)
