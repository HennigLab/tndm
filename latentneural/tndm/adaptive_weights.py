import tensorflow as tf
from typing import List
import numpy as np
import sys


class AdaptiveWeights(tf.keras.callbacks.Callback):
  def __init__(
    self,
    initial: List[tf.Variable]=[1., 1., 1., 1., 1.],
    min_weight: List[float]=[0., 0., 0., 0., 0.], 
    max_weight: List[float]=[1., 1., 1., 1., 1.], 
    update_steps: List[int]=[1, 1, 1, 1, 1], 
    update_rates: List[float]=[0., 0., 0., 0., 0.], 
    update_starts: List[int]=[0, 0, 0, 0, 0]):
    """
    Starting from update_starts, it updates based on the step:

    weight = (initial_weight + update_rate * (max(0, step - update_start) / update_step)).clip(min_weight, max_weight)

    """
    self.w = tf.Variable(initial, shape=tf.TensorShape(len(initial)), dtype=tf.float32, trainable=False)
    self.w_initial = np.asarray(initial)
    self.w_min = np.asarray(min_weight)
    self.w_max = np.asarray(max_weight)
    self.w_update_steps = np.asarray(update_steps)
    self.w_update_rates = np.asarray(update_rates)
    self.w_update_starts = np.asarray(update_starts)
    self.batches = 0
  
  def update_adaptive_weights(self):
    self.w.assign(
      np.clip(
        self.w_initial + self.w_update_rates * (np.clip(self.batches - self.w_update_starts, 0, np.Inf) / self.w_update_steps),
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