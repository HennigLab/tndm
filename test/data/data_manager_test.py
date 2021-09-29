import numpy as np
import pytest
import os

from tndm.data import DataManager
from tndm.utils import remove_folder, upsert_empty_folder


@pytest.fixture(scope='module')
def tmp_folder():
    dir_name = os.path.join('.', 'test', 'data', 'tmp')
    upsert_empty_folder(dir_name)
    return dir_name


@pytest.fixture(scope='module', autouse=True)
def cleanup(request, tmp_folder):
    def remove_test_dir():
        remove_folder(tmp_folder)
    request.addfinalizer(remove_test_dir)


@pytest.mark.unit
def test_train_validation_test_split():
    data = np.random.randn(100, 50, 10, 20)

    train, validation, test, _, _, _ = DataManager.split_dataset(
        data,
        train_pct=0.7,
        valid_pct=0.1,
        test_pct=0.2)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test, _, _, _ = DataManager.split_dataset(
        data,
        train_pct=70,
        valid_pct=10,
        test_pct=20)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test, _, _, _ = DataManager.split_dataset(
        data,
        train_pct=0.7,
        valid_pct=0.1,
        test_pct=0.2)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test, _, _, _ = DataManager.split_dataset(
        data,
        train_pct=0.7,
        valid_pct=0.1)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test, _, _, _ = DataManager.split_dataset(
        data,
        train_pct=70,
        valid_pct=10)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))


@pytest.mark.unit
def test_train_validation_test_split_failures():
    data = np.random.randn(100, 50, 10, 20)

    with pytest.raises(ValueError):
        DataManager.split_dataset(
            data,
            train_pct=70)

    with pytest.raises(ValueError):
        DataManager.split_dataset(
            data,
            train_pct=0.7)


@pytest.mark.unit
def test_build_dataset():
    k = 10  # trials
    t = 100  # timesteps
    n = 50  # neurons
    y = 4  # behavioural dimension
    b = 2  # behaviour relevant latent variables
    # neural latent variables (behaviour relevant + behaviour irrelevant)
    l = 3

    data_dict, settings = DataManager.build_dataset(
        neural_data=np.random.randn(k, t, n),
        behaviour_data=np.random.randn(k, t, y),
        noisless_behaviour_data=np.random.randn(k, t, y),
        rates_data=np.random.randn(k, t, n),
        settings={},
        latent_data=np.random.randn(k, t, 3),
        time_data=np.random.randn(t),
        behaviour_weights=np.random.randn(b, y),
        neural_weights=np.random.randn(l, n),
        train_pct=0.7,
        test_pct=0.1
    )

    assert all([
        x in data_dict.keys()
        for x in [
            'train_data',
            'valid_data',
            'test_data',
            'train_behaviours',
            'valid_behaviours',
            'test_behaviours',
            'train_behaviours_noiseless',
            'valid_behaviours_noiseless',
            'test_behaviours_noiseless',
            'train_rates',
            'valid_rates',
            'test_rates',
            'train_latent',
            'valid_latent',
            'test_latent',
            'time_data',
            'relevant_dims',
            'behaviour_weights',
            'neural_weights'
        ]
    ])

    assert 'created' in list(settings.keys())


@pytest.mark.unit
def test_store_load_dataset(tmp_folder):
    data = {
        'nums': np.random.randn(100, 100),
    }
    DataManager.store_dataset(
        data,
        {'hello': 'world'},
        tmp_folder
    )
    data_dict, settings = DataManager.load_dataset(
        tmp_folder
    )

    np.testing.assert_equal(data_dict['nums'], data['nums'])
    assert settings['hello'] == 'world'
