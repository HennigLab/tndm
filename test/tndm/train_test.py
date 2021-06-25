import tensorflow as tf
import numpy as np
import pytest
import os

from latentneural import TNDM
from latentneural.tndm.train import train
from latentneural.tndm.adaptive_weights import AdaptiveWeights


@pytest.mark.unit
def test_train_model_quick():
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(float) # test_trials X time X neurons
    behaviour_data_train = np.exp(np.random.randn(10, 100, 2)) # test_trials X time X behaviour
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(float) # val_trials X time X neurons
    behaviour_data_val = np.exp(np.random.randn(2, 100, 2)) # val_trials X time X behaviour
    
    adaptive_weights = AdaptiveWeights(
        initial=[1.0, .0, .0, .0, .0],
        update_starts=[0, 0, 1000, 1000, 0],
        update_rates=[0., 0., 0.0005, 0.0005, 0.0005],
        min_weight=[1.0, 0.0, 0.0, 0.0, 0.0]
    )

    model = TNDM(neural_space=50, behaviour_space=2)
    model.build(input_shape=[None] + list(neural_data_train.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(x=neural_data_train, y=behaviour_data_train, callbacks=[adaptive_weights], shuffle=True, epochs=4, validation_data=(neural_data_val, behaviour_data_val))

@pytest.mark.unit
def test_train_wrap():
    train(
        model_settings={}, 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=(np.random.binomial(1, 0.5, (100, 100, 50)).astype(float), np.exp(np.random.randn(100, 100, 4))), 
        val_dataset=(np.random.binomial(1, 0.5, (2, 100, 50)).astype(float), np.exp(np.random.randn(2, 100, 4))), 
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0, 0, 0],
            update_rates=[0, 0.002, 0.002, 0, 0],
        ),
        batch_size=20
    )

@pytest.mark.unit
def test_train_wrap_different_specs():
    train(
        model_settings=dict(
            encoded_var_max=0.1,
            original_generator=False
        ), 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=(np.random.binomial(1, 0.5, (100, 100, 50)).astype(float), np.exp(np.random.randn(100, 100, 4))), 
        val_dataset=None,
        adaptive_lr=dict(factor=0.95, patience=10, min_lr=1e-5),
        logdir=os.path.join('.','latentneural','data','storage'),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0, 0, 0],
            update_rates=[0, 0.002, 0.002, 0, 0],
        ),
        batch_size=20
    )