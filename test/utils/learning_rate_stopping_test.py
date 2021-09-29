import tensorflow as tf
import numpy as np
import pytest

from tndm import TNDM, LFADS
from tndm.utils import LearningRateStopping, AdaptiveWeights


@pytest.mark.unit
def test_learning_rate_stopping():
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

    lr = LearningRateStopping(0.001)

    model = TNDM(neural_dim=50, behaviour_dim=10, max_grad_norm=200)

    model.build(input_shape=[None] + list(input_data.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.1, 2, 0.1)),
        loss_weights=adaptive_weights.w
    )

    history = model.fit(
        x=input_data,
        y=input_n_data,
        callbacks=[adaptive_weights, lr],
        shuffle=True,
        epochs=10)

    assert (history.history['learning_rate'][-2] > 0.001)
    assert (history.history['learning_rate'][-1] <= 0.001)
