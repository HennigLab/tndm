import tensorflow as tf
import numpy as np

class GaussMatrix(tf.keras.layers.Layer):

    def __init__(self, units, kernel_init='random_normal', 
    bias_init='random_normal', sigma=5, **kwargs):
        super(GaussMatrix, self).__init__(units, **kwargs)
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.units=units
        self.sigma=sigma

    def build(self, input_shape):
        self.timesteps = input_shape[-2]

        self.mask_in_size = int(input_shape[-1])
        self.mask_out_size = self.units

        self.kernel = self.add_weight(shape=(self.mask_in_size, self.mask_out_size),
                                initializer=self.kernel_init,
                                trainable=True)
        self.bias = self.add_weight(shape=(self.timesteps,self.mask_out_size),
                                initializer=self.bias_init,
                                trainable=True)

        filter = np.zeros([self.timesteps, self.timesteps])
        # gauss
        for l in range(4*self.sigma+1):
            for j in range(self.timesteps):
                if j+l<self.timesteps:
                    filter[j+l,j] = np.exp(-(l-0)**2/self.sigma**2/2)
                if j-l>=0:
                    filter[j-l,j] = np.exp(-(l-0)**2/self.sigma**2/2)
        self.mask = tf.constant(
            filter,
            dtype='float32'
        )

    @tf.function
    def call(self, inputs):
        return tf.matmul(tf.matmul(self.mask,inputs), self.kernel) + self.bias
