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

from tndm import TNDM, LFADS
from tndm.utils import AdaptiveWeights, logger, CustomEncoder, LearningRateStopping
from tndm.data import DataManager
import tndm.losses as lnl
from .parser import Parser, ModelType


tf.config.run_functions_eagerly(True)


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
              adaptive_lr: Optional[Union[dict, tf.keras.callbacks.Callback]] = None, layers_settings: Dict[str, Any] = {},
              terminating_lr: Optional[float]=None):

        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)

        if layers_settings is None:
            layers_settings = {}

        (x, y), validation_data = Runtime.clean_datasets(
            train_dataset, val_dataset, model_type.with_behaviour)

        if model_type == ModelType.TNDM:
            model_settings.update(
                neural_dim=x.shape[-1],
                behaviour_dim=y.shape[-1],
            )
            model = TNDM(
                **model_settings,
                layers=layers_settings
            )
        elif model_type == ModelType.LFADS:
            model_settings.update(
                neural_dim=x.shape[-1],
            )
            model = LFADS(
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
                    monitor='val_loss/reconstruction', **adaptive_lr))
        if terminating_lr is not None:
            callbacks.append(LearningRateStopping(terminating_lr))

        model.build(input_shape=[None] + list(x.shape[1:]))

        model.compile(
            optimizer=optimizer,
            loss_weights=adaptive_weights.w
        )

        try:
            history = model.fit(
                x=x,
                y=y,
                callbacks=callbacks,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data
            )
        except KeyboardInterrupt:
            return model, None

        return model, history

    @staticmethod
    def train_from_file(settings_path: str):
        start_time=datetime.utcnow()

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

        if 'seed' in settings.keys():
            seed = settings['seed']
        else:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info('Seed was set to %d' % (seed))

        model_type, model_settings, layers_settings, data, dataset_settings, runtime_settings, output_directory = Parser.parse_settings(
            settings)
        (d_n_train, d_n_validation, _), (d_b_train, d_b_validation, _), _ = data
        optimizer, adaptive_weights, adaptive_lr, epochs, batch_size, terminating_lr = runtime_settings
        logger.info('Arguments parsed')

        model, history = Runtime.train(
            model_type=model_type,
            model_settings=model_settings,
            layers_settings=layers_settings,
            train_dataset=(d_n_train, d_b_train),
            val_dataset=(d_n_validation, d_b_validation),
            optimizer=optimizer,
            adaptive_weights=adaptive_weights,
            adaptive_lr=adaptive_lr,
            epochs=epochs,
            batch_size=batch_size,
            logdir=os.path.join(output_directory, 'logs'),
            terminating_lr=terminating_lr)
        logger.info('Model training finished, now saving weights')

        model.save(os.path.join(output_directory, 'saved_model'))
        logger.info('Weights saved, now saving metrics history')

        if history is not None:
            pd.DataFrame(history.history).to_csv(os.path.join(output_directory, 'history.csv'))
            logger.info('Metrics history saved, now evaluating the model')

        stats = Runtime.evaluate_model(data, model)
        with open(os.path.join(output_directory, 'performance.json'), 'w') as fp:
            json.dump(stats, fp, cls=CustomEncoder, indent=2)
        logger.info('Model evaluated, now saving settings')

        end_time=datetime.utcnow()
        settings = dict(
            model=model_settings,
            dataset=dataset_settings,
            runtime_settings=dict(
                optimizer=optimizer, 
                adaptive_weights=adaptive_weights,
                adaptive_lr=adaptive_lr,
                epochs=epochs,
                batch_size=batch_size,
                terminating_lr=terminating_lr
            ),
            default_layers_settings=layers_settings.default_factory(),
            layers_settings=layers_settings,
            seed=seed,
            commit_hash=os.popen('git rev-parse HEAD').read().rstrip(),
            start_time=start_time.strftime('%Y-%m-%d %H:%M:%S.%f%Z'),
            end_time=end_time.strftime('%Y-%m-%d %H:%M:%S.%f%Z'),
            elapsed_time=str(end_time-start_time),
            author=getpass.getuser(),
            machine=socket.gethostname(),
            cpu_only_flag=ArgsParser.get_or_default(dict(os.environ), 'CPU_ONLY', 'FALSE') == 'TRUE',
            visible_devices=tf.config.get_visible_devices()
        )
        with open(os.path.join(output_directory, 'metadata.json'), 'w') as fp:
            json.dump(settings, fp, cls=CustomEncoder, indent=2)
        logger.info('Settings saved, execution terminated')

    @staticmethod
    def evaluate_model(data, model: tf.keras.Model):
        (d_n_train, d_n_validation, d_n_test), (d_b_train, d_b_validation, d_b_test), (d_l_train, d_l_validation, d_l_test) = data
        train_stats, ridge_model = Runtime.evaluate_performance(model, d_n_train, d_b_train, d_l_train)
        validation_stats, _ = Runtime.evaluate_performance(model, d_n_validation, d_b_validation, d_l_validation, ridge_model)
        test_stats, _ = Runtime.evaluate_performance(model, d_n_test, d_b_test, d_l_test, ridge_model)
        return dict(
            train=train_stats,
            validation=validation_stats,
            test=test_stats
        )
    
    def evaluate_performance(model: tf.keras.Model, neural: tf.Tensor, behaviour: tf.Tensor, latent: tf.Tensor, ridge_model=None):
        if isinstance(model, TNDM):
            log_f, b, (g0_r, mean_r, logvar_r), (g0_r, mean_i, logvar_i), (z_r, z_i), inputs = model(neural, training=False)
            z = np.concatenate([z_r.numpy().T, z_i.numpy().T], axis=0).T
        elif isinstance(model, LFADS):
            log_f, (g0, mean, logvar), z, inputs = model(neural, training=False)
            z = z.numpy()
        else:
            raise ValueError('Model not recognized')

        # Behaviour likelihood
        if model.with_behaviour:
            loss_fun = lnl.gaussian_loglike_loss([])
            b_like = loss_fun(behaviour, b).numpy() / behaviour.shape[0]
        else:
            b_like = None

        # Neural likelihood
        loss_fun = lnl.poisson_loglike_loss(model.timestep, ([0], [1]))
        n_like = loss_fun(None, (log_f, inputs)).numpy() / inputs.shape[0]

        # Behaviour R2
        if model.with_behaviour:
            unexplained_error = tf.reduce_sum(tf.square(behaviour - b)).numpy()
            total_error = tf.reduce_sum(tf.square(behaviour - tf.reduce_mean(behaviour, axis=[0,1]))).numpy()
            b_r2 = 1 - (unexplained_error / (total_error + 1e-10))
        else:
            b_r2 = None

        # Latent R2
        if latent is not None:
            z_unsrt = z.T.reshape(z.T.shape[0], z.T.shape[1] * z.T.shape[2]).T
            l = latent.T.reshape(latent.T.shape[0], latent.T.shape[1] * latent.T.shape[2]).T
            if ridge_model is None:
                ridge_model = Ridge(alpha=1.0)
                ridge_model.fit(z_unsrt, l)
            z_srt = ridge_model.predict(z_unsrt)
            unexplained_error = tf.reduce_sum(tf.square(l - z_srt)).numpy()
            total_error = tf.reduce_sum(tf.square(l - tf.reduce_mean(l, axis=[0,1]))).numpy()
            l_r2 = 1 - (unexplained_error / (total_error + 1e-10))
        else:
            l_r2 = None
        return dict(
            behaviour_likelihood=b_like, 
            neural_likelihood=n_like,
            behaviour_r2=b_r2,
            latent_r2=l_r2), ridge_model
