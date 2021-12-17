from __future__ import annotations
import tensorflow as tf
from copy import deepcopy
from typing import Dict, Any
from collections import defaultdict

from tndm.utils import ArgsParser, clean_layer_name, logger
from tndm.layers import GaussianSampling, GeneratorGRU, MaskedDense
from tndm.losses import gaussian_kldiv_loss, poisson_loglike_loss, regularization_loss, \
                        gaussian_loglike_loss, covariance_loss, mse_loss
from .model_loader import ModelLoader


tf.config.run_functions_eagerly(True)

@tf.custom_gradient
def grad_reverse(x, weight):
    y = tf.identity(x)
    def custom_grad(dy):
        return -weight * dy, -weight * dy #lower is better
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, weight):
        return grad_reverse(x, weight)

class DA(ModelLoader, tf.keras.Model):

    def __init__(self, **kwargs: Dict[str, Any]):
        tf.keras.Model.__init__(self)

        self.full_logs: bool = bool(ArgsParser.get_or_default(
            kwargs, 'full_logs', False))
        self.encoder_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'encoder_dim', 64))
        self.rel_initial_condition_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'rel_initial_condition_dim', 64))
        self.rel_decoder_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'rel_decoder_dim', 64))
        self.rel_factors: int = int(ArgsParser.get_or_default(
            kwargs, 'rel_factors', 3))
        self.neural_dim: int = int(ArgsParser.get_or_error(
            kwargs, 'neural_dim'))
        self.behaviour_dim: int = int(ArgsParser.get_or_error(
            kwargs, 'behaviour_dim'))
        self.max_grad_norm: float = float(ArgsParser.get_or_default(
            kwargs, 'max_grad_norm', 200))
        self.prior_variance: float = float(ArgsParser.get_or_default(
            kwargs, 'prior_variance', 0.1)) # TODO: check in original
        self.disentanglement_batches: int = int(ArgsParser.get_or_default(
            kwargs, 'disentanglement_batches', 10))
        self.dropout: float = float(ArgsParser.get_or_default(
            kwargs, 'dropout', 0.15))
        self.timestep: float = float(ArgsParser.get_or_default(
            kwargs, 'timestep', 0.01))
        self.with_behaviour = True
        self.neural_lik_type: str = str(ArgsParser.get_or_default(
            kwargs, 'neural_lik_type','poisson'))
        self.behavior_lik_type: str = str(ArgsParser.get_or_default(
            kwargs, 'behavior_lik_type','MSE'))
        self.behavior_scale: float = float(ArgsParser.get_or_default(
            kwargs, 'behavior_scale',1.0))
        self.threshold_poisson_log_firing_rate: float = float(ArgsParser.get_or_default(
            kwargs, 'threshold_poisson_log_firing_rate', 100.0))
        self.GRU_pre_activation: bool = bool(ArgsParser.get_or_default(
            kwargs, 'GRU_pre_activation', False))

        # convert likelihood types (str) to functions
        self.neural_loglike_loss = self.str2likelihood(self.neural_lik_type)
        self.behavior_loglike_loss = self.str2likelihood(self.behavior_lik_type,
                                                    scale=self.behavior_scale)

        layers = ArgsParser.get_or_default(kwargs, 'layers', {})
        if not isinstance(layers, defaultdict):
            print('Setting default regularizers...')
            layers: Dict[str, Any] = defaultdict(
                lambda: dict(
                    kernel_regularizer=tf.keras.regularizers.L2(l=1),
                    kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='normal')),
                layers
            )
        self.layers_settings = deepcopy(layers)

        # METRICS
        self.tracker_loss = tf.keras.metrics.Sum(name="loss")
        self.tracker_loss_neural_loglike = tf.keras.metrics.Sum(
            name="loss_neural_loglike")
        self.tracker_loss_behavioural_loglike = tf.keras.metrics.Sum(
            name="loss_behavioural_loglike")
        self.tracker_loss_relevant_kldiv = tf.keras.metrics.Sum(
            name="loss_relevant_kldiv")
        self.tracker_loss_reg = tf.keras.metrics.Sum(name="loss_reg")
        self.tracker_loss_count = tf.keras.metrics.Sum(name="loss_count")
        self.tracker_loss_w_neural_loglike = tf.keras.metrics.Mean(
            name="loss_w_neural_loglike")
        self.tracker_loss_w_behavioural_loglike = tf.keras.metrics.Mean(
            name="loss_w_behavioural_loglike")
        self.tracker_loss_w_relevant_kldiv = tf.keras.metrics.Mean(
            name="loss_w_relevant_kldiv")
        self.tracker_loss_w_reg = tf.keras.metrics.Mean(name="loss_w_reg")
        self.tracker_lr = tf.keras.metrics.Mean(name="lr")

        # ENCODER
        self.initial_dropout = tf.keras.layers.Dropout(self.dropout)
        encoder_args: Dict[str, Any] = layers['encoder']
        self.encoded_var_min: float = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_min', .0001)
        self.encoded_var_trainable: bool = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_trainable', True)

        forward_layer = tf.keras.layers.GRU(
            self.encoder_dim, time_major=False, name="EncoderGRUForward", return_state=True, **encoder_args)
        backward_layer = tf.keras.layers.GRU(
            self.encoder_dim, time_major=False, name="EncoderGRUBackward", return_state=True, go_backwards=True, **encoder_args)
        self.encoder = tf.keras.layers.Bidirectional(
            forward_layer, backward_layer=backward_layer, name='EncoderRNN', merge_mode='concat')
        self.dropout_post_encoder = tf.keras.layers.Dropout(self.dropout)
        self.dropout_post_rel_decoder = tf.keras.layers.Dropout(self.dropout)
        # DISTRIBUTION
        # Relevant
        self.relevant_dense_mean = tf.keras.layers.Dense(
            self.rel_initial_condition_dim, name="RelevantDenseMean", **layers['relevant_dense_mean'])
        self.relevant_dense_logvar = tf.keras.layers.Dense(
            self.rel_initial_condition_dim, name="RelevantDenseLogVar", **layers['relevant_dense_logvar'])

        # SAMPLING
        self.relevant_sampling = GaussianSampling(
            name="RelevantGaussianSampling")

        # DECODERS
        # Relevant
        if self.rel_decoder_dim != self.rel_initial_condition_dim:
            self.relevant_dense_pre_decoder = tf.keras.layers.Dense(
                self.rel_decoder_dim, name="RelevantDensePreDecoder", **layers['relevant_dense_pre_decoder'])
        self.relevant_pre_decoder_activation = tf.keras.layers.Activation('tanh')
        relevant_decoder_args: Dict[str, Any] = layers['relevant_decoder']
        self.relevant_decoder_original_cell: float = ArgsParser.get_or_default_and_remove(
            relevant_decoder_args, 'original_cell', False)
        
        if self.relevant_decoder_original_cell:
            relevant_decoder_cell = GeneratorGRU(
                self.rel_decoder_dim, **relevant_decoder_args)
            self.relevant_decoder = tf.keras.layers.RNN(
                relevant_decoder_cell, return_sequences=True, time_major=False, name='RelevantDecoderGRU')
        else:
            self.relevant_decoder = tf.keras.layers.GRU(
                self.rel_decoder_dim, return_sequences=True, time_major=False, name='RelevantDecoderGRU', **relevant_decoder_args)

        # DIMENSIONALITY REDUCTION
        self.rel_factors_dense = tf.keras.layers.Dense(
            self.rel_factors, use_bias=False, name="RelevantFactorsDense", **layers['rel_factors_dense'])

        # BEHAVIOURAL
        behavioural_dense_args: Dict[str, Any] = layers['behavioural_dense']
        self.behaviour_type: str = str(ArgsParser.get_or_default_and_remove(
            behavioural_dense_args, 'behaviour_type', 'causal'))
        if self.behaviour_type == 'causal' or self.behaviour_type == 'full':
            self.behavioural_dense = MaskedDense(
                self.behaviour_dim, name="BehaviouralDense", 
                mask_type=self.behaviour_type, 
                **behavioural_dense_args)
        elif self.behaviour_type == 'synchronous':
            self.behavioural_dense = tf.keras.layers.Dense(
                self.behaviour_dim, name="SynchronousBehaviouralDense", **behavioural_dense_args)
        else:
            raise NotImplementedError(
                'Behaviour type %s not implemented' % (self.behaviour_type))

        # NEURAL
        self.factors_concatenation = tf.keras.layers.Concatenate(
            name="FactorsConcat")
        self.neural_dense = tf.keras.layers.Dense(
            self.neural_dim, name="NeuralDense", **layers['neural_dense'])
        
        # reverse grad
        self.grad_rev = GradReverse()

    def str2likelihood(self, lik_str, scale=1.):
        if lik_str=='poisson':
            ll = poisson_loglike_loss(self.timestep)
        elif lik_str=='gaussian':
            ll = gaussian_loglike_loss(sigma=scale)
        elif lik_str=='MSE':
            ll = mse_loss()
        else:
            raise NotImplementedError(
                f'Likelihood type {lik_str} not implemented')
        return ll
            

    @staticmethod
    def load(filename) -> DA:
        return ModelLoader.load(filename, DA)

    def get_settings(self):
        return dict(
            neural_lik_type=self.neural_lik_type,
            behavior_lik_type=self.behavior_lik_type,  
            behavior_scale=self.behavior_scale,  
            encoder_dim=self.encoder_dim,
            rel_decoder_dim=self.rel_decoder_dim,
            rel_initial_condition_dim=self.rel_initial_condition_dim,
            rel_factors=self.rel_factors,
            neural_dim=self.neural_dim,
            behaviour_dim=self.behaviour_dim,
            timestep=self.timestep,
            max_grad_norm=self.max_grad_norm,
            prior_variance=self.prior_variance,
            layers=self.layers_settings,
            default_layer_settings=self.layers_settings.default_factory(),
            full_logs=self.full_logs
        )

    @tf.function
    def call(self, inputs, training: bool = True, test_sample_mode: str='mean', reverse_gradient: bool=False, weight=None):
        (g0_r, mean_r, logvar_r) = self.encode(inputs, training=training, test_sample_mode=test_sample_mode, reverse_gradient=reverse_gradient, weight=weight)
        log_f, b, z_r = self.decode(
            g0_r, inputs, training=training)
        return log_f, b, (g0_r, mean_r, logvar_r), z_r

    @tf.function
    def decode(self, g0_r, neural, training: bool = True):
        # Assuming inputs are zero and everything comes from the GRU
        u_r = tf.stack([tf.zeros_like(neural)[:, :, -1]
                       for i in range(self.relevant_decoder.cell.units)], axis=-1)
        # Relevant
        if self.rel_decoder_dim != self.rel_initial_condition_dim:
            g0_r = self.relevant_dense_pre_decoder(g0_r, training=training)
        if self.GRU_pre_activation:
            g0_r_pre_decoder = self.relevant_pre_decoder_activation(g0_r) # Not in the original
        else:
            g0_r_pre_decoder = g0_r
        g_r = self.relevant_decoder(u_r, initial_state=g0_r_pre_decoder, training=training)
        dropped_g_r = self.dropout_post_rel_decoder(g_r, training=training) #dropout after GRU
        z_r = self.rel_factors_dense(dropped_g_r, training=training)

        # Behaviour
        b = self.behavioural_dense(z_r, training=training)

        # clipping the log-firingrate log(self.timestep) so that the
        # log-likelihood does not return NaN
        # (https://github.com/tensorflow/tensorflow/issues/47019)
        if self.neural_lik_type == 'poisson':
            log_f = tf.clip_by_value(self.neural_dense(z_r, training=training), 
                                     clip_value_min=-self.threshold_poisson_log_firing_rate,
                                     clip_value_max=self.threshold_poisson_log_firing_rate)
        else:
            log_f = self.neural_dense(z_r, training=training)

        return log_f, b, z_r

    @tf.function
    def encode(self, neural, training: bool=True, test_sample_mode: str='mean', reverse_gradient: bool=False, weight=None):
        dropped_neural = self.initial_dropout(neural, training=training)
        if reverse_gradient:
            encoded = self.grad_rev(self.encoder(dropped_neural, training=training)[0], weight=weight)
        else:
            encoded = self.encoder(dropped_neural, training=training)[0]
        dropped_encoded = self.dropout_post_encoder(encoded, training=training)

        # Relevant
        mean_r = self.relevant_dense_mean(dropped_encoded, training=training)
        if self.encoded_var_trainable:
            logvar_r = tf.math.log(tf.exp(self.relevant_dense_logvar(
                dropped_encoded, training=training)) + self.encoded_var_min)
        else:
            logvar_r = tf.zeros_like(mean_r) + tf.math.log(self.encoded_var_min)
        g0_r = self.relevant_sampling(
            tf.stack([mean_r, logvar_r], axis=-1), training=training, test_sample_mode=test_sample_mode)
        
        return (g0_r, mean_r, logvar_r)

    def compile(self, optimizer, loss_weights: tf.Variable, *args, **kwargs):

        super(DA, self).compile(
            loss=[
                self.neural_loglike_loss,
                self.behavior_loglike_loss,
                gaussian_kldiv_loss(self.prior_variance),
                regularization_loss()],
            optimizer=optimizer,
        )
        self.loss_weights = loss_weights
        assert (loss_weights.shape == (6,)), ValueError(
            'The adaptive weights must have size 6 for DA')

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
            log_f, b, g_r, _ = self(x, training=True)
            behavioural_loglike_loss    = self.compiled_loss._losses[1](y,b)
            relevant_kldiv_loss         = self.compiled_loss._losses[2](g_r)
            reg_loss                    = self.compiled_loss._losses[3](self.losses)
                                        # self.losses contains L2 losses
            log_f, b, g_r, _ = self(x, training=True, reverse_gradient=True, weight=1)
            neural_loglike_loss         = self.compiled_loss._losses[0](log_f,x)

            loss = self.loss_weights[0] * neural_loglike_loss + \
                self.loss_weights[1] * behavioural_loglike_loss + \
                self.loss_weights[2] * relevant_kldiv_loss + \
                self.loss_weights[5] * reg_loss
            unclipped_grads = tape.gradient(loss, self.trainable_variables)

        # For numerical stability (clip_by_global_norm returns NaNs for large
        # grads, becaues grad_global_norms goes to Inf)
        value_clipped_grads = [tf.clip_by_value(
            x, -1e10, 1e10) if x is not None else x for x in unclipped_grads]
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
        self.tracker_loss_neural_loglike.update_state(neural_loglike_loss)
        self.tracker_loss_behavioural_loglike.update_state(
            behavioural_loglike_loss)
        self.tracker_loss_relevant_kldiv.update_state(relevant_kldiv_loss)
        self.tracker_loss_reg.update_state(reg_loss)
        self.tracker_loss_w_neural_loglike.update_state(self.loss_weights[0])
        self.tracker_loss_w_behavioural_loglike.update_state(
            self.loss_weights[1])
        self.tracker_loss_w_relevant_kldiv.update_state(self.loss_weights[2])
        self.tracker_loss_w_reg.update_state(self.loss_weights[5])
        self.tracker_lr.update_state(
            self.optimizer._decayed_lr('float32').numpy())
        self.tracker_loss_count.update_state(x.shape[0])

        core_logs = {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/neural': self.tracker_loss_neural_loglike.result() / self.tracker_loss_count.result(),
            'loss/behavioural': self.tracker_loss_behavioural_loglike.result() / self.tracker_loss_count.result(),
            'loss/relevant_kldiv': self.tracker_loss_relevant_kldiv.result() / self.tracker_loss_count.result(),
            'loss/reconstruction': (self.loss_weights[0] * self.tracker_loss_neural_loglike.result() + self.loss_weights[1] * self.tracker_loss_behavioural_loglike.result()) / self.tracker_loss_count.result(),
            'loss/reg': self.tracker_loss_reg.result(),
            'weights/neural_loglike': self.tracker_loss_w_neural_loglike.result(),
            'weights/behavioural_loglike': self.tracker_loss_w_behavioural_loglike.result(),
            'weights/relevant_kldiv': self.tracker_loss_w_relevant_kldiv.result(),
            'weights/reg': self.tracker_loss_w_reg.result(),
            'learning_rate': self.tracker_lr.result()
        }

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
                **core_logs,
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
            self.tracker_loss_neural_loglike,
            self.tracker_loss_behavioural_loglike,
            self.tracker_loss_relevant_kldiv,
            self.tracker_loss_reg,
            self.tracker_loss_w_neural_loglike,
            self.tracker_loss_w_behavioural_loglike,
            self.tracker_loss_w_relevant_kldiv,
            self.tracker_loss_w_reg,
            self.tracker_lr,
            self.tracker_loss_count
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
        log_f, b, g_r, _ = self(x, training=False)

        neural_loglike_loss         = self.compiled_loss._losses[0](log_f,x)
        behavioural_loglike_loss    = self.compiled_loss._losses[1](y, b)
        relevant_kldiv_loss         = self.compiled_loss._losses[2](g_r)
        reg_loss                    = self.compiled_loss._losses[3](self.losses)
                                    # self.losses contains L2 losses

        loss = self.loss_weights[0] * neural_loglike_loss + \
            self.loss_weights[1] * behavioural_loglike_loss + \
            self.loss_weights[2] * relevant_kldiv_loss + \
            self.loss_weights[5] * reg_loss

        # Update the metrics.
        self.tracker_loss.update_state(loss)
        self.tracker_loss_neural_loglike.update_state(neural_loglike_loss)
        self.tracker_loss_behavioural_loglike.update_state(
            behavioural_loglike_loss)
        self.tracker_loss_relevant_kldiv.update_state(relevant_kldiv_loss)
        self.tracker_loss_reg.update_state(reg_loss)
        self.tracker_loss_count.update_state(x.shape[0])

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/neural': self.tracker_loss_neural_loglike.result() / self.tracker_loss_count.result(),
            'loss/behavioural': self.tracker_loss_behavioural_loglike.result() / self.tracker_loss_count.result(),
            'loss/relevant_kldiv': self.tracker_loss_relevant_kldiv.result() / self.tracker_loss_count.result(),
            'loss/reconstruction': (self.loss_weights[0] * self.tracker_loss_neural_loglike.result() + self.loss_weights[1] * self.tracker_loss_behavioural_loglike.result()) / self.tracker_loss_count.result(),
        }
