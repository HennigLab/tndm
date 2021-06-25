import tensorflow as tf
from typing import Dict, Any
import tensorflow_probability as tfp
import math as m
from collections import defaultdict

from latentneural.utils import ArgsParser
from latentneural.layers import GaussianSampling, GeneratorGRU, MaskedDense


tf.config.run_functions_eagerly(True)


class TNDM(tf.keras.Model):

    _WEIGHTS_NUM = 5

    def __init__(self, **kwargs: Dict[str, Any]):
        super(TNDM, self).__init__()

        self.encoded_space: int = ArgsParser.get_or_default(
            kwargs, 'encoded_space', 64)
        self.irrelevant_factors: int = ArgsParser.get_or_default(
            kwargs, 'irrelevant_factors', 3)
        self.relevant_factors: int = ArgsParser.get_or_default(
            kwargs, 'relevant_factors', 3)
        self.neural_space: int = ArgsParser.get_or_default(
            kwargs, 'neural_space', 50)
        self.behavioural_space: int = ArgsParser.get_or_default(
            kwargs, 'behavioural_space', 1)
        self.max_grad_norm: float = ArgsParser.get_or_default(
            kwargs, 'max_grad_norm', 200)
        self.timestep: float = ArgsParser.get_or_default(
            kwargs, 'timestep', 0.01)
        self.prior_variance: float = ArgsParser.get_or_default(
            kwargs, 'prior_variance', 0.1)

        layers: Dict[str, Any] = defaultdict(
            lambda: dict(
                kernel_regularizer=tf.keras.regularizers.L2(l=0.1),
                kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='normal')),
            ArgsParser.get_or_default(kwargs, 'layers', {}))

        # METRICS
        self.tracker_loss = tf.keras.metrics.Sum(name="loss")
        self.tracker_loss_neural_loglike = tf.keras.metrics.Sum(
            name="loss_neural_loglike")
        self.tracker_loss_behavioural_loglike = tf.keras.metrics.Sum(
            name="loss_behavioural_loglike")
        self.tracker_loss_relevant_kldiv = tf.keras.metrics.Sum(
            name="loss_relevant_kldiv")
        self.tracker_loss_irrelevant_kldiv = tf.keras.metrics.Sum(
            name="loss_irrelevant_kldiv")
        self.tracker_loss_reg = tf.keras.metrics.Sum(name="loss_reg")
        self.tracker_loss_count = tf.keras.metrics.Sum(name="loss_count")
        self.tracker_loss_w_neural_loglike = tf.keras.metrics.Mean(
            name="loss_w_neural_loglike")
        self.tracker_loss_w_behavioural_loglike = tf.keras.metrics.Mean(
            name="loss_w_behavioural_loglike")
        self.tracker_loss_w_relevant_kldiv = tf.keras.metrics.Mean(
            name="loss_w_relevant_kldiv")
        self.tracker_loss_w_irrelevant_kldiv = tf.keras.metrics.Mean(
            name="loss_w_irrelevant_kldiv")
        self.tracker_loss_w_reg = tf.keras.metrics.Mean(name="loss_w_reg")
        self.tracker_lr = tf.keras.metrics.Mean(name="lr")

        # ENCODER
        encoder_args: Dict[str, Any] = layers['encoder']
        self.encoded_var_min: float = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_min', 0.1)
        self.encoded_var_max: float = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_max', 0.1)
        if self.encoded_var_min < self.encoded_var_max:
            self.encoded_var_trainable = True
        else:
            assert self.encoded_var_min == self.encoded_var_max, ValueError(
                'Max encoded var %.2f cannot be greater than min encoded var %.2f' % (
                    self.encoded_var_max,
                    self.encoded_var_min
                ))
            self.encoded_var_trainable = False

        forward_layer = tf.keras.layers.GRU(
            self.encoded_space, time_major=False, name="EncoderGRUForward", return_sequences=True, **encoder_args)
        backward_layer = tf.keras.layers.GRU(
            self.encoded_space, time_major=False, name="EncoderGRUBackward", return_sequences=True, go_backwards=True, **encoder_args)
        self.encoder = tf.keras.layers.Bidirectional(
            forward_layer, backward_layer=backward_layer, name='EncoderRNN')
        self.flatten_post_encoder = tf.keras.layers.Flatten()
        self.encoder_dense = tf.keras.layers.Dense(
            self.encoded_space, name="EncoderDense", **layers['encoder_dense'])

        # DISTRIBUTION
        # Relevant
        self.relevant_dense_mean = tf.keras.layers.Dense(
            self.encoded_space, name="RelevantDenseMean", **layers['relevant_dense_mean'])
        self.relevant_dense_logvar = tf.keras.layers.Dense(
            self.encoded_space, name="RelevantDenseLogVar", **layers['relevant_dense_logvar'])
        # Irrelevant
        self.irrelevant_dense_mean = tf.keras.layers.Dense(
            self.encoded_space, name="IrrelevantDenseMean", **layers['irrelevant_dense_mean'])
        self.irrelevant_dense_logvar = tf.keras.layers.Dense(
            self.encoded_space, name="IrrelevantDenseLogVar", **layers['irrelevant_dense_logvar'])

        # SAMPLING
        self.relevant_sampling = GaussianSampling(
            name="RelevantGaussianSampling")
        self.irrelevant_sampling = GaussianSampling(
            name="IrrelevantGaussianSampling")

        # DECODERS
        # Relevant
        self.relevant_pre_decoder_activation = tf.keras.layers.Activation(
            'tanh')
        relevant_decoder_args: Dict[str, Any] = layers['relevant_decoder']
        self.relevant_decoder_original_cell: float = ArgsParser.get_or_default_and_remove(
            relevant_decoder_args, 'original_cell', True)
        if self.relevant_decoder_original_cell:
            relevant_decoder_cell = GeneratorGRU(
                self.encoded_space, **relevant_decoder_args)
            self.relevant_decoder = tf.keras.layers.RNN(
                relevant_decoder_cell, return_sequences=True, time_major=False, name='RelevantDecoderGRU')
        else:
            self.relevant_decoder = tf.keras.layers.GRU(
                self.encoded_space, return_sequences=True, time_major=False, name='RelevantDecoderGRU', **relevant_decoder_args)
        # Irrelevant
        self.irrelevant_pre_decoder_activation = tf.keras.layers.Activation(
            'tanh')
        irrelevant_decoder_args: Dict[str, Any] = layers['irrelevant_decoder']
        self.irrelevant_decoder_original_cell: float = ArgsParser.get_or_default_and_remove(
            irrelevant_decoder_args, 'original_cell', True)
        if self.irrelevant_decoder_original_cell:
            irrelevant_decoder_cell = GeneratorGRU(
                self.encoded_space, **irrelevant_decoder_args)
            self.irrelevant_decoder = tf.keras.layers.RNN(
                irrelevant_decoder_cell, return_sequences=True, time_major=False, name='IrrelevantDecoderGRU')
        else:
            self.irrelevant_decoder = tf.keras.layers.GRU(
                self.encoded_space, return_sequences=True, time_major=False, name='IrrelevantDecoderGRU', **irrelevant_decoder_args)

        # DIMENSIONALITY REDUCTION
        self.relevant_factors_dense = tf.keras.layers.Dense(
            self.relevant_factors, name="RelevantFactorsDense", activation='tanh', **layers['relevant_factors_dense'])
        self.irrelevant_factors_dense = tf.keras.layers.Dense(
            self.irrelevant_factors, name="IrrelevantFactorsDense", activation='tanh', **layers['irrelevant_factors_dense'])

        # BEHAVIOURAL
        behavioural_dense_args: Dict[str, Any] = layers['behavioural_dense']
        self.behaviour_sigma: float = float(ArgsParser.get_or_default_and_remove(
            behavioural_dense_args, 'behaviour_sigma', 1.0))
        self.behaviour_type: str = str(ArgsParser.get_or_default_and_remove(
            behavioural_dense_args, 'behaviour_type', 'causal'))
        if self.behaviour_type == 'causal':
            self.behavioural_dense = MaskedDense(
                self.behavioural_space, name="CausalBehaviouralDense", **behavioural_dense_args)
        elif self.behaviour_type == 'synchronous':
            self.behavioural_dense = tf.keras.layers.Dense(
                self.behavioural_space, name="SynchronousBehaviouralDense", **behavioural_dense_args)
        else:
            raise NotImplementedError(
                'Behaviour type %s not implemented' % (self.behaviour_type))

        # NEURAL
        self.factors_concatenation = tf.keras.layers.Concatenate(
            name="FactorsConcat")
        self.neural_dense = tf.keras.layers.Dense(
            self.neural_space, name="NeuralDense", **layers['neural_dense'])

    @tf.function
    def call(self, inputs, training: bool = True):
        (g0_r, mean_r, logvar_r), (g0_i, mean_i,
                                   logvar_i) = self.encode(inputs, training=training)
        log_f, b, (z_r, z_i) = self.decode(
            g0_r, g0_i, inputs, training=training)
        return log_f, b, (g0_r, mean_r, logvar_r), (g0_r,
                                                    mean_i, logvar_i), (z_r, z_i), inputs

    @tf.function
    def decode(self, g0_r, g0_i, neural, training: bool = True):
        # Assuming inputs are zero and everything comes from the GRU
        u_r = tf.stack([tf.zeros_like(neural)[:, :, -1]
                       for i in range(self.relevant_decoder.cell.units)], axis=-1)
        u_i = tf.stack([tf.zeros_like(neural)[:, :, -1]
                       for i in range(self.irrelevant_decoder.cell.units)], axis=-1)

        # Relevant
        g0_r_activated = self.relevant_pre_decoder_activation(g0_r)
        g_r = self.relevant_decoder(
            u_r, initial_state=g0_r_activated, training=training)
        z_r = self.relevant_factors_dense(g_r, training=training)

        # Irrelevant
        g0_i_activated = self.irrelevant_pre_decoder_activation(g0_i)
        g_i = self.irrelevant_decoder(
            u_i, initial_state=g0_i_activated, training=training)
        z_i = self.irrelevant_factors_dense(g_i, training=training)

        # Behaviour
        b = self.behavioural_dense(z_r, training=training)

        # Neural
        z = self.factors_concatenation([z_r, z_i], training=training)
        # soft-clipping the log-firingrate log(self.timestep) so that the
        # log-likelihood does not return NaN
        # (https://github.com/tensorflow/tensorflow/issues/47019)
        log_f = tf.math.log(self.timestep) + \
            tf.tanh(self.neural_dense(z, training=training)) * 10

        # In order to be able to auto-encode, the dimensions should be the same
        if not self.built:
            assert all([f_i == i_i for f_i, i_i in zip(
                list(log_f.shape), list(neural.shape))])

        return log_f, b, (z_r, z_i)

    @tf.function
    def encode(self, neural, training: bool = True):
        encoded = self.encoder(neural, training=training)
        encoded_flattened = self.flatten_post_encoder(
            encoded, training=training)
        encoded_reduced = self.encoder_dense(
            encoded_flattened, training=training)

        # Relevant
        mean_r = self.relevant_dense_mean(encoded_reduced, training=training)
        if self.encoded_var_trainable:
            logit_var_r = tf.exp(self.relevant_dense_logvar(
                encoded_reduced, training=training))
            var_r = tf.nn.sigmoid(
                logit_var_r) * (self.encoded_var_max - self.encoded_var_min) + self.encoded_var_min
            logvar_r = tf.math.log(var_r)
        else:
            logvar_r = tf.zeros_like(mean_r) + self.encoded_var_min
        g0_r = self.relevant_sampling(
            tf.stack([mean_r, logvar_r], axis=-1), training=training)

        # Irrelevant
        mean_i = self.irrelevant_dense_mean(encoded_reduced, training=training)
        if self.encoded_var_trainable:
            logit_var_i = tf.exp(self.irrelevant_dense_logvar(
                encoded_reduced, training=training))
            var_i = tf.nn.sigmoid(
                logit_var_i) * (self.encoded_var_max - self.encoded_var_min) + self.encoded_var_min
            logvar_i = tf.math.log(var_i)
        else:
            logvar_i = tf.zeros_like(mean_i) + self.encoded_var_min
        g0_i = self.irrelevant_sampling(
            tf.stack([mean_i, logvar_i], axis=-1), training=training)

        return (g0_r, mean_r, logvar_r), (g0_i, mean_i, logvar_i)

    def compile(self, optimizer, loss_weights: tf.Variable, *args, **kwargs):
        super(TNDM, self).compile(
            loss=[
                TNDM.poisson_loglike_loss(self.timestep),
                TNDM.gaussian_loglike_loss(self.behaviour_sigma),
                TNDM.relevant_kldiv_loss(self.prior_variance),
                TNDM.irrelevant_kldiv_loss(self.prior_variance),
                TNDM.reg_loss()],
            optimizer=optimizer,
        )
        self.loss_weights = loss_weights
        assert (loss_weights.shape == (5,)), ValueError(
            'The adaptive weights must have size 5 for TNDM')

        self.tracker_gradient_dict = {'grads/' + TNDM.clean_layer_name(x.name):
                                      tf.keras.metrics.Sum(name=TNDM.clean_layer_name(x.name)) for x in
                                      self.trainable_variables if 'bias' not in x.name.lower()}
        self.tracker_norms_dict = {'norms/' + TNDM.clean_layer_name(x.name):
                                   tf.keras.metrics.Sum(name=TNDM.clean_layer_name(x.name)) for x in
                                   self.trainable_variables if 'bias' not in x.name.lower()}
        self.tracker_batch_count = tf.keras.metrics.Sum(name="batch_count")

    @staticmethod
    def gaussian_loglike_loss(behaviour_sigma):
        @tf.function
        def loss_fun(y_true, y_pred):
            # GAUSSIAN LOG-LIKELIHOOD
            behavioural = y_true
            (_, b, _, _, _, _), _ = y_pred
            targets = tf.cast(behavioural, dtype=tf.float32)
            mse = 0.5 * \
                tf.reduce_sum(tf.keras.backend.square(
                    (targets - b) / tf.math.square(behaviour_sigma)))
            constant = tf.reduce_sum(tf.ones_like(
                b) * (tf.math.log(behaviour_sigma) + 0.5 * tf.math.log(2 * m.pi)))
            return mse + constant
        return loss_fun

    @staticmethod
    def poisson_loglike_loss(timestep):
        @tf.function
        def loss_fun(y_true, y_pred):
            # POISSON LOG-LIKELIHOOD
            (log_f, _, _, _, _, neural), _ = y_pred
            targets = tf.cast(neural, dtype=tf.float32)
            log_f = tf.cast(tf.math.log(timestep) +
                            log_f, tf.float32)  # Timestep
            return tf.reduce_sum(tf.nn.log_poisson_loss(
                targets=targets,
                log_input=log_f, compute_full_loss=True
            ))
        return loss_fun

    @staticmethod
    def relevant_kldiv_loss(prior_variance):
        @tf.function
        def loss_fun(y_true, y_pred):
            # KL DIVERGENCE
            (_, _, (_, mean_r, logvar_r), _, _, _), _ = y_pred
            dist_prior = tfp.distributions.Normal(0., tf.sqrt(
                prior_variance), name='RelevantPriorNormal')  # PriorVariance
            dist_posterior = tfp.distributions.Normal(
                mean_r, tf.exp(0.5 * logvar_r), name='RelevantPosteriorNormal')
            return tf.reduce_sum(tfp.distributions.kl_divergence(
                dist_prior, dist_posterior, allow_nan_stats=False, name=None
            ))
        return loss_fun

    @staticmethod
    def irrelevant_kldiv_loss(prior_variance):
        @tf.function
        def loss_fun(y_true, y_pred):
            # KL DIVERGENCE
            (_, _, _, (_, mean_i, logvar_i), _, _), _ = y_pred
            dist_prior = tfp.distributions.Normal(0., tf.sqrt(
                prior_variance), name='IrrelevantPriorNormal')  # PriorVariance
            dist_posterior = tfp.distributions.Normal(mean_i, tf.exp(
                0.5 * logvar_i), name='IrrelevantPosteriorNormal')
            return tf.reduce_sum(tfp.distributions.kl_divergence(
                dist_prior, dist_posterior, allow_nan_stats=False, name=None
            ))
        return loss_fun

    @staticmethod
    def reg_loss():
        @tf.function
        def loss_fun(y_true, y_pred):
            # KL DIVERGENCE
            _, regularization = y_pred
            return tf.reduce_sum(regularization)
        return loss_fun

    @staticmethod
    def clean_layer_name(name: str) -> str:
        tks = name.split('/')[-3:-1]
        if len(tks) < 2:
            return tks[0].replace('_', '')
        else:
            return tks[0].split('_')[-1] + '_' + tks[1].replace('_', '')

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

            neural_loglike_loss, behavioural_loglike_loss, relevant_kldiv_loss, irrelevant_kldiv_loss, reg_loss = \
                [func(y, y_pred) for func in self.compiled_loss._losses]
            loss = self.loss_weights[0] * neural_loglike_loss + \
                self.loss_weights[1] * behavioural_loglike_loss + \
                self.loss_weights[2] * relevant_kldiv_loss + \
                self.loss_weights[3] * irrelevant_kldiv_loss + \
                self.loss_weights[4] * reg_loss
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
        self.tracker_loss_neural_loglike.update_state(neural_loglike_loss)
        self.tracker_loss_behavioural_loglike.update_state(
            behavioural_loglike_loss)
        self.tracker_loss_relevant_kldiv.update_state(relevant_kldiv_loss)
        self.tracker_loss_irrelevant_kldiv.update_state(irrelevant_kldiv_loss)
        self.tracker_loss_reg.update_state(reg_loss)
        self.tracker_loss_w_neural_loglike.update_state(self.loss_weights[0])
        self.tracker_loss_w_behavioural_loglike.update_state(
            self.loss_weights[1])
        self.tracker_loss_w_relevant_kldiv.update_state(self.loss_weights[2])
        self.tracker_loss_w_irrelevant_kldiv.update_state(self.loss_weights[3])
        self.tracker_loss_w_reg.update_state(self.loss_weights[4])
        self.tracker_lr.update_state(
            self.optimizer._decayed_lr('float32').numpy())
        self.tracker_loss_count.update_state(x.shape[0])
        self.tracker_batch_count.update_state(1)

        for grad, var in zip(grads, self.trainable_variables):
            if 'bias' not in var.name.lower():
                cleaned_name = TNDM.clean_layer_name(var.name)
                self.tracker_gradient_dict['grads/' +
                                           cleaned_name].update_state(tf.norm(grad, 1))
                self.tracker_norms_dict['norms/' +
                                        cleaned_name].update_state(tf.norm(var, 1))

        return {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/neural': self.tracker_loss_neural_loglike.result() / self.tracker_loss_count.result(),
            'loss/behavioural': self.tracker_loss_behavioural_loglike.result() / self.tracker_loss_count.result(),
            'loss/relevant_kldiv': self.tracker_loss_relevant_kldiv.result() / self.tracker_loss_count.result(),
            'loss/irrelevant_kldiv': self.tracker_loss_irrelevant_kldiv.result() / self.tracker_loss_count.result(),
            'loss/reg': self.tracker_loss_reg.result(),
            'weights/neural_loglike': self.tracker_loss_w_neural_loglike.result(),
            'weights/behavioural_loglike': self.tracker_loss_w_behavioural_loglike.result(),
            'weights/relevant_kldiv': self.tracker_loss_w_relevant_kldiv.result(),
            'weights/irrelevant_kldiv': self.tracker_loss_w_irrelevant_kldiv.result(),
            'weights/reg': self.tracker_loss_w_reg.result(),
            'learning_rate': self.tracker_lr.result(),
            **{k: v.result() / self.tracker_batch_count.result() for k, v in self.tracker_gradient_dict.items()},
            **{k: v.result() / self.tracker_batch_count.result() for k, v in self.tracker_norms_dict.items()}
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.tracker_loss,
            self.tracker_loss_neural_loglike,
            self.tracker_loss_behavioural_loglike,
            self.tracker_loss_relevant_kldiv,
            self.tracker_loss_irrelevant_kldiv,
            self.tracker_loss_reg,
            self.tracker_loss_w_neural_loglike,
            self.tracker_loss_w_behavioural_loglike,
            self.tracker_loss_w_relevant_kldiv,
            self.tracker_loss_w_irrelevant_kldiv,
            self.tracker_loss_w_reg,
            self.tracker_lr,
            self.tracker_loss_count,
            self.tracker_batch_count,
        ]

    @tf.function
    def test_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        y_pred = (self(x, training=False), self.losses)
        neural_loglike_loss, behavioural_loglike_loss, relevant_kldiv_loss, irrelevant_kldiv_loss, reg_loss = \
            [func(y, y_pred) for func in self.compiled_loss._losses]
        loss = self.loss_weights[0] * neural_loglike_loss + \
            self.loss_weights[1] * behavioural_loglike_loss + \
            self.loss_weights[2] * relevant_kldiv_loss + \
            self.loss_weights[3] * irrelevant_kldiv_loss + \
            self.loss_weights[4] * reg_loss

        # Update the metrics.
        self.tracker_loss.update_state(loss)
        self.tracker_loss_neural_loglike.update_state(neural_loglike_loss)
        self.tracker_loss_behavioural_loglike.update_state(
            behavioural_loglike_loss)
        self.tracker_loss_relevant_kldiv.update_state(relevant_kldiv_loss)
        self.tracker_loss_irrelevant_kldiv.update_state(irrelevant_kldiv_loss)
        self.tracker_loss_reg.update_state(reg_loss)
        self.tracker_loss_count.update_state(x.shape[0])

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/neural': self.tracker_loss_neural_loglike.result() / self.tracker_loss_count.result(),
            'loss/behavioural': self.tracker_loss_behavioural_loglike.result() / self.tracker_loss_count.result(),
            'loss/relevant_kldiv': self.tracker_loss_relevant_kldiv.result() / self.tracker_loss_count.result(),
            'loss/irrelevant_kldiv': self.tracker_loss_irrelevant_kldiv.result() / self.tracker_loss_count.result(),
        }
