from __future__ import annotations
from copy import deepcopy
import tensorflow as tf
from typing import Dict, Any
from collections import defaultdict
import json

from tndm.utils import ArgsParser, clean_layer_name, logger
from tndm.layers import GaussianSampling, GeneratorGRU
from tndm.losses import gaussian_kldiv_loss, poisson_loglike_loss, regularization_loss
from .model_loader import ModelLoader


tf.config.run_functions_eagerly(True)


class LFADS(ModelLoader, tf.keras.Model):

    def __init__(self, **kwargs: Dict[str, Any]):
        tf.keras.Model.__init__(self)

        self.full_logs: bool = bool(ArgsParser.get_or_default(
            kwargs, 'full_logs', False))
        self.encoder_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'encoder_dim', 64))
        self.initial_condition_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'initial_condition_dim', 64))
        self.decoder_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'decoder_dim', 64))
        self.factors: int = int(ArgsParser.get_or_default(kwargs, 'factors', 3))
        self.neural_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'neural_dim', 50))
        self.max_grad_norm: float = float(ArgsParser.get_or_default(
            kwargs, 'max_grad_norm', 200))
        self.timestep: float = float(ArgsParser.get_or_default(
            kwargs, 'timestep', 0.01))
        self.prior_variance: float = float(ArgsParser.get_or_default(
            kwargs, 'prior_variance', 0.1))
        self.dropout: float = float(ArgsParser.get_or_default(
            kwargs, 'dropout', 0.05))
        self.with_behaviour = False

        layers = ArgsParser.get_or_default(kwargs, 'layers', {})
        if not isinstance(layers, defaultdict):
            layers: Dict[str, Any] = defaultdict(
                lambda: dict(
                    kernel_regularizer=tf.keras.regularizers.L2(l=1),
                    kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='normal')),
                layers
            )
        self.layers_settings = deepcopy(layers)

        # METRICS
        self.tracker_loss = tf.keras.metrics.Sum(name="loss")
        self.tracker_loss_loglike = tf.keras.metrics.Sum(name="loss_loglike")
        self.tracker_loss_kldiv = tf.keras.metrics.Sum(name="loss_kldiv")
        self.tracker_loss_reg = tf.keras.metrics.Sum(name="loss_reg")
        self.tracker_loss_count = tf.keras.metrics.Sum(name="loss_count")
        self.tracker_loss_w_loglike = tf.keras.metrics.Mean(
            name="loss_w_loglike")
        self.tracker_loss_w_kldiv = tf.keras.metrics.Mean(name="loss_w_kldiv")
        self.tracker_loss_w_reg = tf.keras.metrics.Mean(name="loss_w_reg")
        self.tracker_lr = tf.keras.metrics.Mean(name="lr")

        # ENCODER
        self.initial_dropout = tf.keras.layers.Dropout(self.dropout)
        encoder_args: Dict[str, Any] = layers['encoder']
        self.encoded_var_min: float = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_min', 0.1)
        self.encoded_var_trainable: bool = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_trainable', True)

        forward_layer = tf.keras.layers.GRU(
            self.encoder_dim, time_major=False, name="EncoderGRUForward", return_state=True, **encoder_args)
        backward_layer = tf.keras.layers.GRU(
            self.encoder_dim, time_major=False, name="EncoderGRUBackward", return_state=True, go_backwards=True, **encoder_args)
        self.encoder = tf.keras.layers.Bidirectional(
            forward_layer, backward_layer=backward_layer, name='EncoderRNN', merge_mode='concat')
        self.dropout_post_encoder = tf.keras.layers.Dropout(self.dropout)
        
        # DISTRIBUTION
        self.dense_mean = tf.keras.layers.Dense(
            self.initial_condition_dim, name="DenseMean", **layers['dense_mean'])
        self.dense_logvar = tf.keras.layers.Dense(
            self.initial_condition_dim, name="DenseLogVar", **layers['dense_logvar'])

        # SAMPLING
        self.sampling = GaussianSampling(name="GaussianSampling")

        # DECODERS
        if self.decoder_dim != self.initial_condition_dim:
            self.dense_pre_decoder = tf.keras.layers.Dense(
                self.decoder_dim, name="DensePreDecoder", **layers['dense_pre_decoder'])
        self.pre_decoder_activation = tf.keras.layers.Activation('tanh')
        decoder_args: Dict[str, Any] = layers['decoder']
        self.original_generator: float = ArgsParser.get_or_default_and_remove(
            decoder_args, 'original_cell', False)
        decoder_dropout = ArgsParser.get_or_default_and_remove(
            decoder_args, 'dropout', self.dropout)
        if self.original_generator:
            decoder_cell = GeneratorGRU(self.decoder_dim, **decoder_args)
            self.decoder = tf.keras.layers.RNN(
                decoder_cell, return_sequences=True, time_major=False, name='DecoderGRU')
            logger.warning('Dropout not implemented for the original decoder')
        else:
            self.decoder = tf.keras.layers.GRU(
                self.decoder_dim, return_sequences=True, time_major=False, dropout=decoder_dropout, name='DecoderGRU', **decoder_args)

        # DIMENSIONALITY REDUCTION
        self.dense = tf.keras.layers.Dense(
            self.factors, name="Dense", **layers['dense'])

        # NEURAL
        self.neural_dense = tf.keras.layers.Dense(
            self.neural_dim, name="NeuralDense", **layers['neural_dense'])

    @staticmethod
    def load(filename) -> LFADS:
        return ModelLoader.load(filename, LFADS)

    def get_settings(self):
        return dict(        
            encoder_dim=self.encoder_dim,
            factors=self.factors,
            neural_dim=self.neural_dim,
            max_grad_norm=self.max_grad_norm,
            timestep=self.timestep,
            prior_variance=self.prior_variance,
            layers=self.layers_settings,
            default_layer_settings=self.layers_settings.default_factory(),
            full_logs=self.full_logs
        )

    @tf.function
    def call(self, inputs, training: bool = True):
        g0, mean, logvar = self.encode(inputs, training=training)
        log_f, z = self.decode(g0, inputs, training=training)
        return log_f, (g0, mean, logvar), z, inputs

    @tf.function
    def decode(self, g0, inputs, training: bool = True):
        # Assuming inputs are zero and everything comes from the GRU
        u = tf.stack([tf.zeros_like(inputs)[:, :, -1]
                     for i in range(self.decoder.cell.units)], axis=-1)

        if self.decoder_dim != self.initial_condition_dim:
            g0 = self.dense_pre_decoder(g0, training=training)
        g0_activated = self.pre_decoder_activation(g0)
        g = self.decoder(u, initial_state=g0_activated, training=training)

        z = self.dense(g, training=training)

        # soft-clipping the log-firingrate log(self.timestep) so that the
        # log-likelihood does not return NaN
        # (https://github.com/tensorflow/tensorflow/issues/47019)
        log_f = tf.tanh(self.neural_dense(z, training=training) / 100) * 100

        # In order to be able to auto-encode, the dimensions should be the same
        if not self.built:
            assert all([f_i == i_i for f_i, i_i in zip(
                list(log_f.shape), list(inputs.shape))])

        return log_f, z

    @tf.function
    def encode(self, inputs, training: bool = True):
        dropped_neural = self.initial_dropout(inputs, training=training)
        encoded = self.encoder(dropped_neural, training=training)[0]
        dropped_encoded = self.dropout_post_encoder(encoded, training=training)

        mean = self.dense_mean(dropped_encoded, training=training)

        if self.encoded_var_trainable:
            logvar = tf.math.log(tf.exp(self.dense_logvar(
                dropped_encoded, training=training)) + self.encoded_var_min)
        else:
            logvar = tf.zeros_like(mean) + tf.math.log(self.encoded_var_min)

        g0 = self.sampling(
            tf.stack([mean, logvar], axis=-1), training=training)
        return g0, mean, logvar

    def compile(self, optimizer, loss_weights, *args, **kwargs):
        super(LFADS, self).compile(
            loss=[
                poisson_loglike_loss(self.timestep, args_idx=([0,0], [0,3])),
                gaussian_kldiv_loss(self.prior_variance, args_idx=([0,1,1], [0,1,2])),
                regularization_loss(arg_idx=[1])],
            optimizer=optimizer,
        )
        self.loss_weights = loss_weights

        if self.full_logs:
            self.tracker_gradient_dict = {'grads/' + clean_layer_name(x.name):
                                        tf.keras.metrics.Sum(name=clean_layer_name(x.name)) for x in
                                        self.trainable_variables if 'bias' not in x.name.lower()}
            self.tracker_norms_dict = {'norms/' + clean_layer_name(x.name):
                                    tf.keras.metrics.Sum(name=clean_layer_name(x.name)) for x in
                                    self.trainable_variables if 'bias' not in x.name.lower()}
            self.tracker_batch_count = tf.keras.metrics.Sum(name="batch_count")

    @tf.function
    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = (self(x, training=True), self.losses)
            loss_loglike, loss_kldiv, loss_reg = [
                func(None, y_pred) for func in self.compiled_loss._losses]
            loss = self.loss_weights[0] * loss_loglike + \
                self.loss_weights[1] * loss_kldiv + \
                self.loss_weights[2] * loss_reg
            unclipped_grads = tape.gradient(loss, self.trainable_variables)

        # For numerical stability (clip_by_global_norm returns NaNs for large
        # grads, becaues grad_global_norms goes to Inf)
        value_clipped_grads = [tf.clip_by_value(
            x, -1e16, 1e16) if x is not None else x for x in unclipped_grads]
        grads, grad_global_norm = tf.clip_by_global_norm(
            value_clipped_grads, self.max_grad_norm)
        # Run backwards pass.

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(grads, self.trainable_variables)
            if grad is not None
        )

        # Compute our own metrics
        self.tracker_loss.update_state(loss)
        self.tracker_loss_loglike.update_state(loss_loglike)
        self.tracker_loss_kldiv.update_state(loss_kldiv)
        self.tracker_loss_reg.update_state(loss_reg)
        self.tracker_loss_w_loglike.update_state(self.loss_weights[0])
        self.tracker_loss_w_kldiv.update_state(self.loss_weights[1])
        self.tracker_loss_w_reg.update_state(self.loss_weights[2])
        self.tracker_lr.update_state(
            self.optimizer._decayed_lr('float32').numpy())
        self.tracker_loss_count.update_state(x.shape[0])

        core_logs = {'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
                'loss/loglike': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
                'loss/kldiv': self.tracker_loss_kldiv.result() / self.tracker_loss_count.result(),
                'loss/reg': self.tracker_loss_reg.result(),
                'loss/reconstruction': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
                'weights/loglike': self.tracker_loss_w_loglike.result(),
                'weights/kldiv': self.tracker_loss_w_kldiv.result(),
                'weights/reg': self.tracker_loss_w_reg.result(),
                'learning_rate': self.tracker_lr.result()}

        if self.full_logs:
            self.tracker_batch_count.update_state(1)

            for grad, var in zip(grads, self.trainable_variables):
                if 'bias' not in var.name.lower():
                    cleaned_name = clean_layer_name(var.name)
                    self.tracker_gradient_dict['grads/' +
                                            cleaned_name].update_state(tf.norm(grad, 1))
                    self.tracker_norms_dict['norms/' +
                                            cleaned_name].update_state(tf.norm(var, 1))

            return {
                **core_logs
                **{k: v.result() / self.tracker_batch_count.result() for k, v in self.tracker_gradient_dict.items()},
                **{k: v.result() / self.tracker_batch_count.result() for k, v in self.tracker_norms_dict.items()}
            }
        else:
            return core_logs

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        core_losses = [
            self.tracker_loss,
            self.tracker_loss_loglike,
            self.tracker_loss_kldiv,
            self.tracker_loss_reg,
            self.tracker_loss_w_loglike,
            self.tracker_loss_w_kldiv,
            self.tracker_loss_w_reg,
            self.tracker_lr,
            self.tracker_loss_count,
        ]
        if self.full_logs:
            return core_losses + [self.tracker_batch_count] + list(self.tracker_norms_dict.values()) + list(self.tracker_gradient_dict.values())
        else:
            return core_losses

    @tf.function
    def test_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        y_pred = (self(x, training=False), self.losses)
        loss_loglike, loss_kldiv, loss_reg = [
            func(None, y_pred) for func in self.compiled_loss._losses]
        loss = self.loss_weights[0] * loss_loglike + \
            self.loss_weights[1] * loss_kldiv

        # Update the metrics.
        self.tracker_loss.update_state(loss)
        self.tracker_loss_loglike.update_state(loss_loglike)
        self.tracker_loss_kldiv.update_state(loss_kldiv)
        self.tracker_loss_count.update_state(x.shape[0])

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/loglike': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
            'loss/kldiv': self.tracker_loss_kldiv.result() / self.tracker_loss_count.result(),
            'loss/reconstruction': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
        }
