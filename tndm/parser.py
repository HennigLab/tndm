from __future__ import annotations

from collections import defaultdict
from tndm.utils.args_parser import ArgsParser
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from collections.abc import Iterable
import numpy as np
import json
import yaml
import os
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import Ridge
from datetime import datetime
import getpass
import socket

from tndm.utils import AdaptiveWeights, logger, CustomEncoder
from tndm.data import DataManager
import tndm.losses as lnl


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
        return 6 if self.name.lower() == 'tndm' else 3

class Parser(object):

    @staticmethod
    def parse_settings(settings: Dict[str, Any]):
        # MODEL
        model = ArgsParser.get_or_error(settings, 'model')
        model_type: ModelType = ModelType.from_string(
            ArgsParser.get_or_error(model, 'type'))
        model_settings, layers_settings = Parser.parse_model_settings(
            ArgsParser.get_or_error(model, 'settings'))

        # OUTPUT
        output = ArgsParser.get_or_error(settings, 'output')
        output_directory = ArgsParser.get_or_error(output, 'directory')

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
        d_n_train, d_n_validation, d_n_test = Parser.parse_data(
            dataset, neural_keys)
        d_b_train, d_b_validation, d_b_test = Parser.parse_data(
            dataset, behavioural_keys)
        d_l_train, d_l_validation, d_l_test = Parser.parse_data(
            dataset, latent_keys)
        valid_available = (d_b_validation is not None) and (
            d_n_validation is not None) if model_type.with_behaviour else (d_n_validation is not None)
        data = (
            (d_n_train, d_n_validation, d_n_test),
            (d_b_train, d_b_validation, d_b_test),
            (d_l_train, d_l_validation, d_l_test)
        )

        # RUNTIME
        runtime = ArgsParser.get_or_default(settings, 'runtime', {})
        initial_lr, lr_callback, terminating_lr = Parser.parse_learning_rate(
            ArgsParser.get_or_default(runtime, 'learning_rate', {}), valid_available)
        optimizer = Parser.parse_optimizer(
            ArgsParser.get_or_default(runtime, 'optimizer', {}), initial_lr)
        weights = Parser.parse_weights(
            ArgsParser.get_or_default(runtime, 'weights', {}), model_type)
        epochs = ArgsParser.get_or_default(runtime, 'epochs', 1000)
        batch_size = ArgsParser.get_or_default(runtime, 'batch_size', 8)
        runtime_settings = (optimizer, weights, lr_callback, epochs, batch_size, terminating_lr)

        return model_type, model_settings, layers_settings, data, dataset_settings, runtime_settings, output_directory

    @staticmethod
    def parse_model_settings(model_settings):
        # DEFAULTS
        default_layer = ArgsParser.get_or_default_and_remove(
            model_settings, 'default_layer_settings', {})
        default_init = Parser.parse_initializer(
            ArgsParser.get_or_default(default_layer, 'kernel_initializer', {}))
        default_reg = Parser.parse_regularizer(
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
                    tmp_layer[key] = Parser.parse_initializer(value)
                elif 'regularizer' in key.lower():
                    tmp_layer[key] = Parser.parse_regularizer(value)
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
        terminating_lr = ArgsParser.get_or_default_and_remove(learning_rate_settings, 'terminating', None)
        initial_lr = ArgsParser.get_or_default_and_remove(
            learning_rate_settings, 'initial', 1e-2)
        monitor = 'val_loss/reconstruction' if valid_available else 'train_loss/reconstruction'
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, **learning_rate_settings)
        return initial_lr, lr_callback, terminating_lr

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
