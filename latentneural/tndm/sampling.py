import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, training: bool=True):
        if training:
            batch = tf.shape(inputs)[-3]
            dim = tf.shape(inputs)[-2]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return inputs[:,:,0] + tf.exp(0.5 * inputs[:,:,1]) * epsilon
        else:
            return inputs[:,:,0]