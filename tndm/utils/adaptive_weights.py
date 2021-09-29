import tensorflow as tf
from typing import List, Optional, Any
import numpy as np


class AdaptiveWeights(tf.keras.callbacks.Callback):

    _DEFAULTS = dict(
        initial=[float, 1.0],
        min_weight=[float, 0.0],
        max_weight=[float, 1.0],
        update_step=[int, 1],
        update_rate=[float, 0.0],
        update_start=[int, 0]
    )

    def __init__(
            self,
            initial: List[float] = [1., 1., 1.],
            min_weight: List[float] = None,
            max_weight: List[float] = None,
            update_step: List[int] = None,
            update_rate: List[float] = None,
            update_start: List[int] = None):
        """
        Starting from update_start, it updates based on the step:

        weight = (initial_weight + update_rate * (max(0, step - update_start) / update_step)).clip(min_weight, max_weight)

        """
        self._dimensionality = len(initial)

        self.w = tf.Variable(initial, shape=tf.TensorShape(
            len(initial)), dtype=tf.float32, trainable=False)
        self.w_initial = np.asarray(initial)
        self.w_min = self._get_or_proxy(
            argname='min_weight', input_array=min_weight)
        self.w_max = self._get_or_proxy(
            argname='max_weight', input_array=max_weight)
        self.w_update_step = self._get_or_proxy(
            argname='update_step', input_array=update_step)
        self.w_update_rate = self._get_or_proxy(
            argname='update_rate', input_array=update_rate)
        self.w_update_start = self._get_or_proxy(
            argname='update_start', input_array=update_start)
        self.batches = 0

    def _get_or_proxy(self, argname,
                      input_array: Optional[List[Any]]) -> np.ndarray:
        if input_array is not None:
            if len(input_array) == self._dimensionality:
                return np.asarray(
                    input_array, dtype=self._DEFAULTS[argname][0])
            else:
                return np.ones((self._dimensionality,),
                               dtype=self._DEFAULTS[argname][0]) * input_array[0]
        else:
            return np.ones((self._dimensionality,),
                           dtype=self._DEFAULTS[argname][0]) * self._DEFAULTS[argname][1]

    def update_adaptive_weights(self):
        self.w.assign(
            np.clip(
                self.w_initial + self.w_update_rate *
                (np.clip(self.batches - self.w_update_start,
                 0, np.Inf) / self.w_update_step),
                self.w_min,
                self.w_max
            )
        )

    def on_train_batch_begin(self, batch, logs=None):
        self.update_adaptive_weights()
        self.batches += 1

    def on_train_begin(self, logs=None):
        self.batches = 0
        self.update_adaptive_weights()
