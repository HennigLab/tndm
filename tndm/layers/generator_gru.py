import tensorflow as tf


class GeneratorGRU(tf.keras.layers.Layer):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    This version is specialized for the generator, but isn't as fast, so
    we have two.  Note this allows for l2 regularization on the recurrent
    weights, but also implicitly rescales the inputs via the 1/sqrt(input)
    scaling in the linear helper routine to be large magnitude, if there are
    fewer inputs than recurrent state.

    """

    def __init__(self, units: int, forget_bias: float = 1.0, clip_value: float = 5, name: str = "GeneratorGRU",
                 kernel_initializer=tf.keras.initializers.GlorotNormal(), kernel_regularizer=tf.keras.regularizers.l2(l=0.01)):
        """Create a GRU object.

        Args:
          num_units: Number of units in the GRU
          forget_bias (optional): Hack to help learning.
          clip_value (optional): if the recurrent values grow above this value,
            clip them.
        """
        super(GeneratorGRU, self).__init__(name=name)
        self._units: int = int(units)
        self._forget_bias: float = float(forget_bias)
        self._clip_value: float = clip_value
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

        self.x_2_ru = tf.keras.layers.Dense(2 * self._units, use_bias=False, name="x_2_ru",
                                            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.h_2_ru = tf.keras.layers.Dense(2 * self._units, use_bias=True, name="h_2_ru",
                                            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.x_2_c = tf.keras.layers.Dense(self._units, name="x_2_c", activation=None, use_bias=False,
                                           kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.rh_2_c = tf.keras.layers.Dense(self._units, name="rh_2_c", activation=None, use_bias=True,
                                            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self._units,
            'forget_bias': self._forget_bias,
            'clip_value': self._clip_value,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
        })
        return config

    @property
    def units(self) -> int:
        return self._units

    @property
    def state_size(self):
        return self._units

    @property
    def output_size(self):
        return self._units

    @property
    def state_multiplier(self):
        return 1

    def output_from_state(self, state):
        """Return the output portion of the state."""
        return state

    @tf.function
    def call(self, inputs, states):
        """Gated recurrent unit (GRU) function.

        Args:
          inputs: A 2D batch x input_dim tensor of inputs.
          state: The previous state from the last time step.

        Returns:
          A tuple (state, state), where state is the newly computed state at time t.
          It is returned twice to respect an interface that works for LSTMs.
        """

        x = inputs
        h = states[0]
        # We start with bias of 1.0 to not reset and not update.
        r_x = u_x = 0.0
        if x is not None:
            r_x, u_x = tf.split(
                axis=1, num_or_size_splits=2, value=self.x_2_ru(x))

        r_h, u_h = tf.split(axis=1, num_or_size_splits=2, value=self.h_2_ru(h))

        r = r_x + r_h
        u = u_x + u_h
        r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)

        c_x = 0.0
        if x is not None:
            c_x = self.x_2_c(x)
        c_rh = self.rh_2_c(r * h)

        c = tf.tanh(c_x + c_rh)

        new_h = u * h + (1 - u) * c
        new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)

        return new_h, new_h
