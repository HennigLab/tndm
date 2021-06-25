import tensorflow as tf
from typing import Dict, Any
import tensorflow_probability as tfp
from collections import defaultdict

from latentneural.utils import ArgsParser
from latentneural.layers import GaussianSampling, GeneratorGRU


tf.config.run_functions_eagerly(True)


class LFADS(tf.keras.Model):

    _WEIGHTS_NUM = 3

    def __init__(self, **kwargs: Dict[str, Any]):
        super(LFADS, self).__init__()

        self.encoded_space: int = ArgsParser.get_or_default(
            kwargs, 'encoded_space', 64)
        self.factors: int = ArgsParser.get_or_default(kwargs, 'factors', 3)
        self.neural_space: int = ArgsParser.get_or_default(
            kwargs, 'neural_space', 50)
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
        encoder_dense_args: Dict[str, Any] = layers['encoder_dense']
        self.encoder_dense = tf.keras.layers.Dense(
            self.encoded_space, name="EncoderDense", **encoder_dense_args)

        # DISTRIBUTION
        self.dense_mean = tf.keras.layers.Dense(
            self.encoded_space, name="DenseMean", **layers['dense_mean'])
        self.dense_logvar = tf.keras.layers.Dense(
            self.encoded_space, name="DenseLogVar", **layers['dense_logvar'])

        # SAMPLING
        self.sampling = GaussianSampling(name="GaussianSampling")

        # DECODERS
        self.pre_decoder_activation = tf.keras.layers.Activation('tanh')
        decoder_args: Dict[str, Any] = layers['decoder']
        self.original_generator: float = ArgsParser.get_or_default_and_remove(
            decoder_args, 'original_cell', True)
        if self.original_generator:
            decoder_cell = GeneratorGRU(self.encoded_space, **decoder_args)
            self.decoder = tf.keras.layers.RNN(
                decoder_cell, return_sequences=True, time_major=False, name='DecoderGRU')
        else:
            self.decoder = tf.keras.layers.GRU(
                self.encoded_space, return_sequences=True, time_major=False, name='DecoderGRU', **decoder_args)

        # DIMENSIONALITY REDUCTION
        self.dense = tf.keras.layers.Dense(
            self.factors, name="Dense", activation='tanh', **layers['dense'])

        # NEURAL
        self.neural_dense = tf.keras.layers.Dense(
            self.neural_space, name="NeuralDense", **layers['neural_dense'])

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

        g0_activated = self.pre_decoder_activation(g0)
        g = self.decoder(u, initial_state=g0_activated, training=training)

        z = self.dense(g, training=training)

        # soft-clipping the log-firingrate log(self.timestep) so that the
        # log-likelihood does not return NaN
        # (https://github.com/tensorflow/tensorflow/issues/47019)
        log_f = tf.math.log(self.timestep) + \
            tf.tanh(self.neural_dense(z, training=training)) * 10

        # In order to be able to auto-encode, the dimensions should be the same
        if not self.built:
            assert all([f_i == i_i for f_i, i_i in zip(
                list(log_f.shape), list(inputs.shape))])

        return log_f, z

    @tf.function
    def encode(self, inputs, training: bool = True):
        encoded = self.encoder(inputs, training=training)
        encoded_flattened = self.flatten_post_encoder(
            encoded, training=training)
        encoded_reduced = self.encoder_dense(
            encoded_flattened, training=training)

        mean = self.dense_mean(encoded_reduced, training=training)

        if self.encoded_var_trainable:
            logit_var = tf.exp(self.dense_logvar(
                encoded_reduced, training=training))
            var = tf.nn.sigmoid(logit_var) * (self.encoded_var_max -
                                              self.encoded_var_min) + self.encoded_var_min
            logvar = tf.math.log(var)
        else:
            logvar = tf.zeros_like(mean) + self.encoded_var_min

        g0 = self.sampling(
            tf.stack([mean, logvar], axis=-1), training=training)
        return g0, mean, logvar

    def compile(self, optimizer, loss_weights, *args, **kwargs):
        super(LFADS, self).compile(
            loss=[
                LFADS.loglike_loss(self.timestep),
                LFADS.kldiv_loss(self.prior_variance),
                LFADS.reg_loss()],
            optimizer=optimizer,
        )
        self.loss_weights = loss_weights
        self.tracker_gradient_dict = {'grads/' + LFADS.clean_layer_name(x.name):
                                      tf.keras.metrics.Sum(name=LFADS.clean_layer_name(x.name)) for x in
                                      self.trainable_variables if 'bias' not in x.name.lower()}
        self.tracker_norms_dict = {'norms/' + LFADS.clean_layer_name(x.name):
                                   tf.keras.metrics.Sum(name=LFADS.clean_layer_name(x.name)) for x in
                                   self.trainable_variables if 'bias' not in x.name.lower()}
        self.tracker_batch_count = tf.keras.metrics.Sum(name="batch_count")

    @staticmethod
    def loglike_loss(timestep):
        @tf.function
        def loss_fun(y_true, y_pred):
            # LOG-LIKELIHOOD
            (log_f, (_, _, _), _, inputs), regularization = y_pred
            targets = tf.cast(inputs, dtype=tf.float32)
            log_f = tf.cast(tf.math.log(timestep) +
                            log_f, tf.float32)  # Timestep
            return tf.reduce_sum(tf.nn.log_poisson_loss(
                targets=targets,
                log_input=log_f, compute_full_loss=True
            ))
        return loss_fun

    @staticmethod
    def kldiv_loss(prior_variance):
        @tf.function
        def loss_fun(y_true, y_pred):
            # KL DIVERGENCE
            (_, (_, mean, logvar), _, _), _ = y_pred
            dist_prior = tfp.distributions.Normal(0., tf.sqrt(
                prior_variance), name='PriorNormal')  # PriorVariance
            dist_posterior = tfp.distributions.Normal(
                mean, tf.exp(0.5 * logvar), name='PosteriorNormal')
            return tf.reduce_sum(tfp.distributions.kl_divergence(
                dist_prior, dist_posterior, allow_nan_stats=False, name=None
            ))
        return loss_fun

    @staticmethod
    def reg_loss():
        @tf.function
        def loss_fun(y_true, y_pred):
            # KL DIVERGENCE
            (_, (_, _, _), _, _), regularization = y_pred
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
        self.tracker_batch_count.update_state(1)

        for grad, var in zip(grads, self.trainable_variables):
            if 'bias' not in var.name.lower():
                cleaned_name = LFADS.clean_layer_name(var.name)
                self.tracker_gradient_dict['grads/' +
                                           cleaned_name].update_state(tf.norm(grad, 1))
                self.tracker_norms_dict['norms/' +
                                        cleaned_name].update_state(tf.norm(var, 1))

        return {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/loglike': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
            'loss/kldiv': self.tracker_loss_kldiv.result() / self.tracker_loss_count.result(),
            'loss/reg': self.tracker_loss_reg.result(),
            'weights/loglike': self.tracker_loss_w_loglike.result(),
            'weights/kldiv': self.tracker_loss_w_kldiv.result(),
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
            self.tracker_loss_loglike,
            self.tracker_loss_kldiv,
            self.tracker_loss_reg,
            self.tracker_loss_w_loglike,
            self.tracker_loss_w_kldiv,
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
        }
