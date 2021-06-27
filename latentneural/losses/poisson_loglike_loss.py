from typing import Tuple, List
import tensorflow as tf


def poisson_loglike_loss(timestep, args_idx: Tuple[List[int], List[int]]):
    logrates_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[0]]))
    observations_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[1]]))
    @tf.function
    def loss_fun(y_true, y_pred):
        # POISSON LOG-LIKELIHOOD
        targets = tf.cast(observations_getter(y_pred), dtype=tf.float32)
        logrates = tf.cast(tf.math.log(timestep) +
                        logrates_getter(y_pred), tf.float32)  # Timestep
        return tf.reduce_sum(tf.nn.log_poisson_loss(
            targets=targets,
            log_input=logrates, compute_full_loss=True
        ))
    return loss_fun