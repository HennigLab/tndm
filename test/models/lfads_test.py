import tensorflow as tf
import numpy as np
import pytest
import os

from tndm import LFADS
from tndm.utils import AdaptiveWeights, upsert_empty_folder, remove_folder


@pytest.fixture(scope='module', autouse=True)
def cleanup(request):
    def remove_test_dir():
        folder = os.path.join(
            'test', 'models', 'lfads_tmp')
        upsert_empty_folder(folder)
        remove_folder(folder)
    request.addfinalizer(remove_test_dir)


@pytest.fixture(scope='function')
def save_location():
    folder = os.path.join('test', 'models', 'lfads_tmp')
    upsert_empty_folder(folder)
    return folder

@pytest.mark.unit
def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)
                        )  # trials X time X neurons
    model = LFADS(neural_dim=50, layers={'decoder': {'original_cell': True}})
    model.build(input_shape=[None] + list(input_data.shape[1:]))

    log_f, (g0_r, r_mean, r_logvar), z, inputs = model.call(
        input_data, training=True)

    tf.debugging.assert_equal(log_f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))

    log_f, (g0_r, r_mean, r_logvar), z, inputs = model.call(
        input_data, training=False)

    tf.debugging.assert_equal(log_f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))


@pytest.mark.unit
def test_train_model_quick(save_location):
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(
        float)  # test_trials X time X neurons
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(
        float)  # val_trials X time X neurons

    adaptive_weights = AdaptiveWeights(
        initial=[0.5, 1, 1],
        min_weight=[0., 0., 0.],
        max_weight=[1., 1., 1.],
        update_step=[1, 2, 1],
        update_start=[2, 1, 1],
        update_rate=[-0.05, -0.1, -0.01]
    )

    model = LFADS(neural_dim=50, max_grad_norm=200)

    model.build(input_shape=[None] + list(neural_data_train.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(
        x=neural_data_train,
        y=None,
        callbacks=[adaptive_weights],
        shuffle=True,
        epochs=4,
        validation_data=(
            neural_data_val,
            None))

    before = model(neural_data_train, training=False)[0]
    model.save(save_location)
    model_new = LFADS.load(save_location)
    after = model_new(neural_data_train, training=False)[0]

    tf.debugging.assert_equal(
        before, after
    )

@pytest.mark.regression
@pytest.mark.slow
def test_training_regression():
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(
        float)  # test_trials X time X neurons
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(
        float)  # val_trials X time X neurons

    adaptive_weights = AdaptiveWeights(
        initial=[1, 0, 0],
        update_rate=[0, 0.002, 0],
    )

    model = LFADS(neural_dim=50, max_grad_norm=200)

    model.build(input_shape=[None] + list(neural_data_train.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(
        x=neural_data_train,
        y=None,
        callbacks=[adaptive_weights],
        shuffle=True,
        epochs=10,
        validation_data=(
            neural_data_val,
            None))

    log_f, _, _, _ = model.call(neural_data_train, training=False)

    probs = 1 / (1 + np.exp(-log_f.numpy()))

    assert np.corrcoef(probs.flatten(), neural_data_train.flatten())[
        0, 1] > 0  # Rates are correlated with actual spikes
