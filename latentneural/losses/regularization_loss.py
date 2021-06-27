from typing import List
import tensorflow as tf


def regularization_loss(arg_idx: List[int]):
    losses_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in arg_idx]))
    @tf.function
    def loss_fun(y_true, y_pred):
        # KL DIVERGENCE
        return tf.reduce_sum(losses_getter(y_pred))
    return loss_fun
