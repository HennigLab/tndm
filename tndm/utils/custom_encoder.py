from json import JSONEncoder
from numpy.lib.arraysetops import isin
import tensorflow as tf
import numpy as np


class CustomEncoder(JSONEncoder):
    def default(self, o):
        try:
            return super(CustomEncoder, self).default(o)
        except TypeError as e:
            if isinstance(o, tf.keras.initializers.Initializer):
                if isinstance(o, tf.keras.initializers.VarianceScaling):
                    return {
                        'type': 'variance_scaling',
                        'arguments': {k: v for k, v in o.__dict__.items() if k[0] != '_'}
                        }
                return {k: v for k, v in o.__dict__.items() if k[0] != '_'}
            elif isinstance(o, tf.keras.regularizers.Regularizer):
                if isinstance(o, tf.keras.regularizers.L2):
                    return {
                        'type': 'l2',
                        'arguments': {k: v for k, v in o.__dict__.items() if k[0] != '_'}
                        }
                return {k: v for k, v in o.__dict__.items() if k[0] != '_'}
            elif isinstance(o, tf.keras.optimizers.Optimizer):
                if isinstance(o, tf.keras.optimizers.Adam):
                    return {
                        'type': 'adam',
                        'arguments': {k: v for k, v in o.__dict__.items() if k[0] != '_'}
                        }
                return {k: v for k, v in o.__dict__.items() if k[0] != '_'}
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, tf.Tensor):
                return o.numpy().tolist()
            elif isinstance(o, tf.Variable):
                return o.numpy().tolist()
            elif isinstance(o, tf.keras.Model):
                return o.name
            else:
                return o.__dict__