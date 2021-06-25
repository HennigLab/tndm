from __future__ import annotations

from collections import defaultdict
from latentneural.utils.args_parser import ArgsParser
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from collections.abc import Iterable
import numpy as np
import json
import yaml
import os
from copy import deepcopy

from latentneural.utils import AdaptiveWeights, logger
from latentneural.models import TNDM, LFADS
from latentneural.data import DataManager


tf.config.run_functions_eagerly(True)


class ModelType(Enum):
    TNDM = 'tndm'
    LFADS = 'lfads'

    @staticmethod
    def from_string(string: str) -> ModelType:
        if string.lower() == 'tndm':
            return ModelType.TNDM
        elif string.lower() == 'lfads':
            return ModelType.LFADS
        else:
            raise ValueError('Value not recognized.')

    @property
    def with_behaviour(self) -> bool:
        return self.name.lower() == 'tndm'

    @property
    def weights_num(self) -> bool:
        return 5 if self.name.lower() == 'tndm' else 3


class Runtime(object):

    @staticmethod
    def clean_datasets(
            train_dataset: Union[List[tf.Tensor], tf.Tensor],
            val_dataset: Optional[Union[List[tf.Tensor], tf.Tensor]] = None,
            with_behaviour: bool = False) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Optional[Tuple[tf.Tensor, tf.Tensor]]]:

        train_neural = None
        train_behaviour = None
        valid = None

        if tf.debugging.is_numeric_tensor(train_dataset) or (
                isinstance(train_dataset, np.ndarray)):
            train_dataset = (train_dataset,)
        elif isinstance(train_dataset, Iterable):
            train_dataset = tuple(train_dataset)

        if tf.debugging.is_numeric_tensor(val_dataset) or (
                isinstance(val_dataset, np.ndarray)):
            val_dataset = (val_dataset,)
        elif isinstance(val_dataset, Iterable):
            val_dataset = tuple(val_dataset)

        if with_behaviour:
            assert len(train_dataset) > 1, ValueError(
                'The train dataset must be a list containing two elements: neural activity and behaviour')
            neural_dims = train_dataset[0].shape[1:]
            behavioural_dims = train_dataset[1].shape[1:]
            train_neural, train_behaviour = train_dataset[:2]
            if val_dataset is not None:
                if len(val_dataset) > 1:
                    assert neural_dims == val_dataset[0].shape[1:], ValueError(
                        'Validation and training datasets must have coherent sizes')
                    assert behavioural_dims == val_dataset[1].shape[1:], ValueError(
                        'Validation and training datasets must have coherent sizes')
                    valid = val_dataset[:2]
        else:
            assert len(train_dataset) > 0, ValueError(
                'Please provide a non-empty train dataset')
            neural_dims = train_dataset[0].shape[1:]
            train_neural, train_behaviour = train_dataset[0], None
            if val_dataset is not None:
                assert neural_dims == val_dataset[0].shape[1:], ValueError(
                    'Validation and training datasets must have coherent sizes')
                valid = (val_dataset[0], None)

        return (train_neural, train_behaviour), valid

    @staticmethod
    def train(model_type: Union[str, ModelType], model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int,
              train_dataset: Tuple[tf.Tensor, tf.Tensor], adaptive_weights: AdaptiveWeights,
              val_dataset: Optional[Tuple[tf.Tensor, tf.Tensor]] = None, batch_size: Optional[int] = None, logdir: Optional[str] = None,
              adaptive_lr: Optional[Union[dict, tf.keras.callbacks.Callback]] = None, layers_settings: Dict[str, Any] = {}):

        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)

        if layers_settings is None:
            layers_settings = {}

        (x, y), validation_data = Runtime.clean_datasets(
            train_dataset, val_dataset, model_type.with_behaviour)

        if model_type == ModelType.TNDM:
            model = TNDM(
                neural_space=x.shape[-1],
                behavioural_space=y.shape[-1],
                **model_settings,
                layers=layers_settings
            )
        elif model_type == ModelType.LFADS:
            model = LFADS(
                neural_space=x.shape[-1],
                **model_settings,
                layers=layers_settings
            )
        else:
            raise NotImplementedError(
                'This model type has not been implemented yet')

        callbacks = [adaptive_weights]
        if logdir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))
        if adaptive_lr is not None:
            if isinstance(adaptive_lr, tf.keras.callbacks.Callback):
                callbacks.append(adaptive_lr)
            else:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', **adaptive_lr))

        model.build(input_shape=[None] + list(x.shape[1:]))

        model.compile(
            optimizer=optimizer,
            loss_weights=adaptive_weights.w
        )

        try:
            model.fit(
                x=x,
                y=y,
                callbacks=callbacks,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data
            )
        except KeyboardInterrupt:
            return model

        return model

    @staticmethod
    def train_from_file(settings_path: str):
        is_json = settings_path.split('.')[-1].lower() == 'json'
        is_yaml = settings_path.split('.')[-1].lower() in ['yaml', 'yml']
        if is_json:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        elif is_yaml:
            with open(settings_path, 'r') as f:
                try:
                    settings = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)
        logger.info('Loaded settings file:\n%s' % yaml.dump(
            settings, default_flow_style=None, default_style=''))

        model_type, model_settings, layers_settings, data, runtime_settings, logdir = Runtime.parse_settings(
            settings)
        train_dataset, val_dataset = data
        optimizer, adaptive_weights, adaptive_lr, epochs, batch_size = runtime_settings
        logger.info('Arguments parsed')

        model = Runtime.train(
            model_type=model_type,
            model_settings=model_settings,
            layers_settings=layers_settings,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            adaptive_weights=adaptive_weights,
            adaptive_lr=adaptive_lr,
            epochs=epochs,
            batch_size=batch_size,
            logdir=logdir)
        logger.info('Model training finished')

    @staticmethod
    def parse_settings(settings: Dict[str, Any]):
        # MODEL
        model = ArgsParser.get_or_error(settings, 'model')
        model_type: ModelType = ModelType.from_string(
            ArgsParser.get_or_error(model, 'type'))
        model_settings, layers_settings = Runtime.parse_model_settings(
            ArgsParser.get_or_error(model, 'settings'))

        # OUTPUT
        output = ArgsParser.get_or_error(settings, 'output')
        output_directory = ArgsParser.get_or_error(output, 'directory')
        logdir = os.path.join(output_directory, 'logs')

        # DATA
        data = ArgsParser.get_or_error(settings, 'data')
        data_directory = ArgsParser.get_or_error(data, 'directory')
        data_dataset_filename = ArgsParser.get_or_default(
            data, 'dataset_filename', 'dataset.h5')
        data_metadata_filename = ArgsParser.get_or_default(
            data, 'metadata_filename', 'metadata.json')
        dataset, dataset_settings = DataManager.load_dataset(
            directory=data_directory,
            filename=data_dataset_filename,
            metadata_filename=data_metadata_filename)
        neural_keys = ArgsParser.get_or_error(data, 'neural_keys')
        behavioural_keys = ArgsParser.get_or_default(
            data, 'behavioural_keys', {})
        latent_keys = ArgsParser.get_or_default(data, 'latent_keys', {})
        d_n_train, d_n_validation, d_n_test = Runtime.parse_data(
            dataset, neural_keys)
        d_b_train, d_b_validation, d_b_test = Runtime.parse_data(
            dataset, behavioural_keys)
        d_l_train, d_l_validation, d_l_test = Runtime.parse_data(
            dataset, latent_keys)
        valid_available = (d_b_validation is not None) and (
            d_n_validation is not None) if model_type.with_behaviour else (d_n_validation is not None)
        data = ((d_n_train, d_b_train), (d_n_validation, d_b_validation))

        # RUNTIME
        runtime = ArgsParser.get_or_default(settings, 'runtime', {})
        initial_lr, lr_callback = Runtime.parse_learning_rate(
            ArgsParser.get_or_default(runtime, 'learning_rate', {}), valid_available)
        optimizer = Runtime.parse_optimizer(
            ArgsParser.get_or_default(runtime, 'optimizer', {}), initial_lr)
        weights = Runtime.parse_weights(
            ArgsParser.get_or_default(runtime, 'weights', {}), model_type)
        epochs = ArgsParser.get_or_default(runtime, 'epochs', 1000)
        batch_size = ArgsParser.get_or_default(runtime, 'batch_size', 8)
        runtime_settings = optimizer, weights, lr_callback, epochs, batch_size

        return model_type, model_settings, layers_settings, data, runtime_settings, logdir

    @staticmethod
    def parse_model_settings(model_settings):
        # DEFAULTS
        default_layer = ArgsParser.get_or_default_and_remove(
            model_settings, 'default_layer_settings', {})
        default_init = Runtime.parse_initializer(
            ArgsParser.get_or_default(default_layer, 'kernel_initializer', {}))
        default_reg = Runtime.parse_regularizer(
            ArgsParser.get_or_default(default_layer, 'kernel_regularizer', {}))

        # ALL OTHER LAYERS
        layers = defaultdict(
            lambda: {'kernel_regularizer': default_reg, 'kernel_initializer': default_init})
        custom_layers = ArgsParser.get_or_default_and_remove(
            model_settings, 'layers', {})
        for layer_name, settings in custom_layers.items():
            tmp_layer = deepcopy(layers.default_factory())
            for key, value in settings.items():
                if 'initializer' in key.lower():
                    tmp_layer[key] = Runtime.parse_initializer(value)
                elif 'regularizer' in key.lower():
                    tmp_layer[key] = Runtime.parse_regularizer(value)
                else:
                    tmp_layer[key] = value
            layers[layer_name] = tmp_layer

        return model_settings, layers

    @staticmethod
    def parse_initializer(settings):
        kernel_init_type = ArgsParser.get_or_default(settings, 'type', {})
        kernel_init_kwargs = ArgsParser.get_or_default(
            settings, 'arguments', {})
        if kernel_init_type.lower() == 'variance_scaling':
            i = tf.keras.initializers.VarianceScaling
        else:
            raise NotImplementedError(
                'Only variance_scaling has been implemented')
        return i(**kernel_init_kwargs)

    @staticmethod
    def parse_regularizer(settings):
        kernel_reg_type = ArgsParser.get_or_default(settings, 'type', {})
        kernel_reg_kwargs = ArgsParser.get_or_default(
            settings, 'arguments', {})
        if kernel_reg_type.lower() == 'l2':
            r = tf.keras.regularizers.L2
        else:
            raise NotImplementedError('Only l2 has been implemented')
        return r(**kernel_reg_kwargs)

    @staticmethod
    def parse_optimizer(optimizer_settings, initial_lr: float):
        type = ArgsParser.get_or_default(optimizer_settings, 'type', 'adam')
        kwargs = ArgsParser.get_or_default(optimizer_settings, 'arguments', {})
        if type.lower() == 'adam':
            opt = tf.keras.optimizers.Adam
        else:
            raise NotImplementedError(
                'Only Adam opimizer has been implemented')
        return opt(learning_rate=initial_lr, **kwargs)

    @staticmethod
    def parse_weights(weights_settings, model_type: ModelType):
        if weights_settings:
            initial = ArgsParser.get_or_default_and_remove(
                weights_settings, 'initial', [1.0])
            if not (len(initial) == model_type.weights_num):
                initial = [initial[0] for x in range(model_type.weights_num)]
            w = AdaptiveWeights(initial=initial, **weights_settings)
        else:
            w = AdaptiveWeights(
                initial=[1 for x in range(model_type.weights_num)])
        return w

    @staticmethod
    def parse_learning_rate(learning_rate_settings, valid_available: bool):
        initial_lr = ArgsParser.get_or_default_and_remove(
            learning_rate_settings, 'initial', 1e-2)
        monitor = 'val_loss' if valid_available else 'train_loss'
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, **learning_rate_settings)
        return initial_lr, lr_callback

    @staticmethod
    def parse_data(dataset: Dict[str, Any], keys: Dict[str, str]):
        fields = ['train', 'validation', 'test']
        out = []

        for field in fields:
            key = ArgsParser.get_or_default(keys, field, None)
            if (key is not None) and (key in (dataset.keys())):
                out.append(dataset[key].astype('float'))
            else:
                out.append(None)

        return out
