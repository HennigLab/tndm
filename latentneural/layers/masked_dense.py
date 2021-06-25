import tensorflow as tf
import numpy as np


class MaskedDense(tf.keras.layers.Dense):

    def __init__(self, units, **kwargs):
        """Masked Dense

        It creates a causal relationship between the input TS and the output TS.

        Args:
            units: units in the output layer (output dimensionality X timesteps)
            timesteps (int): timesteps
        """
        super(MaskedDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        self.timesteps = input_shape[-2]
        self.mask_out_size = self.units
        self.units *= self.timesteps
        super(MaskedDense, self).build(
            tf.TensorShape([None, np.prod(input_shape[1:])]))
        self.mask_in_size = int(input_shape[-1])

        self.input_spec = tf.keras.layers.InputSpec(
            shape=tuple([None] + input_shape[1:]),
            allow_last_axis_squeeze=True)

        self.inbound_reshape = tf.keras.layers.Reshape(
            [np.prod(input_shape[1:])]
        )

        self.mask = tf.constant(
            np.tril(
                np.ones(
                    [self.timesteps, self.timesteps]
                )).flatten(
            ).repeat(
                self.mask_in_size
            ).reshape(
                [self.timesteps * self.mask_in_size, self.timesteps],
                order='F'
            ).repeat(
                self.mask_out_size,
                axis=1
            ),
            dtype='float32'
        )

        self.outbound_reshape = tf.keras.layers.Reshape(
            [self.timesteps, self.mask_out_size]
        )

    @tf.function
    def call(self, inputs):
        x = self.inbound_reshape(inputs)
        y = tf.matmul(x, tf.math.multiply(self.kernel, self.mask))
        if self.use_bias:
            y = y + self.bias
        if self.activation is not None:
            y = self.activation(y)
        return self.outbound_reshape(y)
