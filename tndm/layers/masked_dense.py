import tensorflow as tf
import numpy as np

class MaskedDense(tf.keras.layers.Dense):

    def __init__(self, units, mask_type='causal', feedback_factors=0, **kwargs):
        """Masked Dense

        It creates a causal relationship between the input TS and the output TS.

        Args:
            units: units in the output layer (output dimensionality X timesteps)
            timesteps (int): timesteps
            mask_type: can be 'causal', 'feedback', 'causal+feedback', 'full' or 'gauss'
            feedback_factors (int): number of feedback factors (default=0)
        """
        super(MaskedDense, self).__init__(units, **kwargs)
        self.mask_type = mask_type
        self.feedback_factors = feedback_factors

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

        if self.mask_type=='causal':
            '''
            Upper-triangular wight matrices
            that transform factors to behaviors
            '''
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
        elif self.mask_type=='causal+feedback':
            '''
            Last self.feedback_factors are lower-triangular,
            the rest are upper-triangular.
            '''
            filters = np.empty((self.timesteps, self.mask_in_size, self.timesteps))
            for i in range(self.mask_in_size):
                filters[:,i] = np.tril(
                        np.ones(
                            [self.timesteps, self.timesteps]
                        )).T
                if i>=self.mask_in_size-self.feedback_factors:
                    filters[:,i] = filters[:,i].T
            self.mask = tf.constant(
                filters.reshape(-1,self.timesteps).repeat(
                    self.mask_out_size,
                    axis=1
                ),
                dtype='float32'
            )
        elif self.mask_type=='feedback':
            '''
            Lower-triangular
            '''
            filters = np.empty((self.timesteps, self.mask_in_size, self.timesteps))
            for i in range(self.mask_in_size):
                filters[:,i] = np.tril(
                        np.ones(
                            [self.timesteps, self.timesteps]
                        ))
            self.mask = tf.constant(
                filters.reshape(-1,self.timesteps).repeat(
                    self.mask_out_size,
                    axis=1
                ),
                dtype='float32'
            )
        elif self.mask_type=='gauss':
            '''
            An auto-smoothing version of 'synchronous'.
            Here or kernel weights are still learnable independently, 
            but they are weighted with the 'mask' coefficient which is 
            a 5-pixel-wide gaussian around the diagonal
            '''
            filters = np.empty((self.timesteps, self.mask_in_size, self.timesteps))
            for i in range(self.mask_in_size):
                filters[:,i] = np.zeros(
                            [self.timesteps, self.timesteps]
                        )
                # gauss
                for l in range(21):
                    for j in range(self.timesteps):
                        if j+l<self.timesteps:
                            filters[j+l,i,j] = np.exp(-(l-0)**2/5**2/2)
                        if j-l>=0:
                            filters[j-l,i,j] = np.exp(-(l-0)**2/5**2/2)
            self.mask = tf.constant(
                filters.reshape(-1,self.timesteps).repeat(
                    self.mask_out_size,
                    axis=1
                ),
                dtype='float32'
            )
        elif self.mask_type=='full':
            self.mask = tf.constant(
                np.ones(
                     [self.timesteps * self.mask_in_size, self.timesteps * self.mask_out_size]
                ),
                dtype='float32'
            )
        else:
            raise NotImplementedError(
                'Mask type %s not implemented' % (self.mask_type))

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
