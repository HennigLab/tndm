import tensorflow as tf
import numpy as np
import pytest
import os

from tndm import TNDM
from tndm.utils import AdaptiveWeights, upsert_empty_folder, remove_folder


@pytest.fixture(scope='module', autouse=True)
def cleanup(request):
    def remove_test_dir():
        folder = os.path.join(
            'test', 'models', 'tndm_tmp')
        upsert_empty_folder(folder)
        remove_folder(folder)
    request.addfinalizer(remove_test_dir)

@pytest.fixture(scope='function')
def save_location():
    folder = os.path.join('test', 'models', 'tndm_tmp')
    upsert_empty_folder(folder)
    return folder

@pytest.mark.unit
def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)
                        )  # trials X time X neurons
    model = TNDM(neural_dim=50, behaviour_dim=2, layers={'irrelevant_decoder': {'original_cell': True}, 'relevant_decoder': {'original_cell': True}})
    model.build(input_shape=[None] + list(input_data.shape[1:]))

    f, b, (g0_r, r_mean, r_logvar), (g0_i, i_mean, i_logvar), (z_r,
                                                               z_i), inputs = model.call(input_data, training=True)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 2]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_i.shape, tf.TensorShape([10, 64]))

    f, b, (g0_r, r_mean, r_logvar), (g0_i, i_mean, i_logvar), (z_r,
                                                               z_i), inputs = model.call(input_data, training=False)

    tf.debugging.assert_equal(b.shape, tf.TensorShape([10, 100, 2]))
    tf.debugging.assert_equal(f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(i_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_i.shape, tf.TensorShape([10, 64]))


@pytest.mark.unit
def test_train_model_quick(save_location):
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(
        float)  # test_trials X time X neurons
    behaviour_data_train = np.exp(np.random.randn(
        10, 100, 2))  # test_trials X time X behaviour
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(
        float)  # val_trials X time X neurons
    # val_trials X time X behaviour
    behaviour_data_val = np.exp(np.random.randn(2, 100, 2))

    adaptive_weights = AdaptiveWeights(
        initial=[1.0, .0, .0, .0, 1.0, .0],
        update_start=[0, 0, 1000, 1000, 0, 0],
        update_rate=[0., 0., 0.0005, 0.0005, 0.0, 0.0005],
        min_weight=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )

    model = TNDM(neural_dim=50, behaviour_dim=2)
    model.build(input_shape=[None] + list(neural_data_train.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(
        x=neural_data_train,
        y=behaviour_data_train,
        callbacks=[adaptive_weights],
        shuffle=True,
        epochs=4,
        validation_data=(
            neural_data_val,
            behaviour_data_val))

    before = model(neural_data_train, training=False)[0]
    model.save(save_location)
    model_new = TNDM.load(save_location)
    after = model_new(neural_data_train, training=False)[0]

    tf.debugging.assert_equal(
        before, after
    )