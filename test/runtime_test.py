import tensorflow as tf
import numpy as np
import pytest
import os
import runpy
from copy import deepcopy


from tndm.runtime import Runtime, ModelType
from tndm.utils import AdaptiveWeights, remove_folder, upsert_empty_folder


@pytest.fixture(scope='module')
def json_filename():
    return os.path.join('.', 'test', 'mocks', 'runtime_test_settings.json')


@pytest.fixture(scope='module')
def yaml_filename():
    return os.path.join('.', 'test', 'mocks', 'runtime_test_settings.yaml')

@pytest.fixture(scope='module', autouse=True)
def cmd_line_args(request):
    import sys
    args = sys.argv
    sys.argv[0] = 'r'
    sys.argv[1] = json_filename

    def restore_cmd_args():
        sys.argv = args
    request.addfinalizer(restore_cmd_args)

    return True

@pytest.mark.unit
def test_train_wrap_tndm():
    Runtime.train(
        model_type=ModelType.TNDM,
        model_settings={},
        optimizer=tf.keras.optimizers.Adam(1e-3),
        epochs=2,
        train_dataset=(
            np.random.binomial(
                1, 0.5, (100, 100, 50)).astype(float), np.exp(
                np.random.randn(
                    100, 100, 4))),
        val_dataset=(
            np.random.binomial(
                1, 0.5, (2, 100, 50)).astype(float), np.exp(
                np.random.randn(
                    2, 100, 4))),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0, 0, 1, 0],
            update_rate=[0, 0.002, 0.002, 0, 0, 0],
        ),
        batch_size=20,
        layers_settings={}
    )

@pytest.mark.unit
def test_train_wrap_tndm_different_specs():
    Runtime.train(
        model_type='tndm',
        model_settings={},
        optimizer=tf.keras.optimizers.Adam(1e-3),
        epochs=2,
        train_dataset=(
            np.random.binomial(
                1, 0.5, (100, 100, 50)).astype(float), np.exp(
                np.random.randn(
                    100, 100, 4))),
        val_dataset=None,
        adaptive_lr=dict(factor=0.95, patience=10, min_lr=1e-5),
        logdir=os.path.join('.', 'tndm', 'data', 'storage'),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0, 0, 1, 0],
            update_rate=[0, 0.002, 0.002, 0, 0, 0],
        ),
        batch_size=20,
        layers_settings={
            'encoder': dict(var_min=0.1),
            'relevant_decoder': dict(original_cell=False),
            'irrelevant_decoder': dict(original_cell=False)
        }
    )


@pytest.mark.unit
def test_train_wrap_lfads():
    Runtime.train(
        model_type='lfads',
        model_settings={},
        optimizer=tf.keras.optimizers.Adam(1e-3),
        epochs=2,
        train_dataset=np.random.binomial(1, 0.5, (100, 100, 50)).astype(float),
        val_dataset=np.random.binomial(1, 0.5, (20, 100, 50)).astype(float),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0],
            update_rate=[0, 0.002, 0],
        ),
        batch_size=20,
        layers_settings=None
    )

@pytest.fixture(scope='module', autouse=True)
def cleanup(request):
    def remove_test_dir():
        folder = os.path.join(
            'test', 'mocks', 'runtime_test_outputs')
        upsert_empty_folder(folder)
        remove_folder(folder)
    request.addfinalizer(remove_test_dir)

@pytest.mark.unit
def test_train_wrap_lfads_different_specs():
    Runtime.train(
        model_type=ModelType.LFADS,
        model_settings={},
        optimizer=tf.keras.optimizers.Adam(1e-3),
        epochs=2,
        train_dataset=np.random.binomial(1, 0.5, (100, 100, 50)).astype(float),
        val_dataset=None,
        adaptive_lr=dict(factor=0.95, patience=10, min_lr=1e-5),
        logdir=os.path.join('.', 'tndm', 'data', 'storage'),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0],
            update_rate=[0, 0.002, 0.002],
        ),
        batch_size=20,
        layers_settings={
            'encoder': dict(var_min=0.1),
            'decoder': dict(original_cell=False)
        }
    )

@pytest.mark.unit
def test_running_tndm_from_json(json_filename):
    Runtime.train_from_file(json_filename)

@pytest.mark.unit
def test_running_lfads_from_yaml(yaml_filename):
    Runtime.train_from_file(yaml_filename)

@pytest.mark.unit
def test_running_from_command_line(json_filename, cmd_line_args):
    if cmd_line_args:
        runpy.run_module('tndm', run_name='__main__')
    