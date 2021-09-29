import tensorflow as tf
from typing import List, Optional, Any
import numpy as np

from tndm.utils import ArgsParser


class LearningRateStopping(tf.keras.callbacks.Callback):

    def __init__(self, limit_learning_rate: float):
        self.limit_learning_rate = float(limit_learning_rate)

    @tf.function
    def on_train_batch_end(self, batch, logs=None):
        lr = ArgsParser.get_or_default(logs, 'learning_rate', ArgsParser.get_or_default(logs, 'lr', self.limit_learning_rate + 1))
        if lr <= self.limit_learning_rate:
            self.model.stop_training = True