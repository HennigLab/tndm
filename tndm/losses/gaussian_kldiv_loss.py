from typing import Tuple, List
import tensorflow_probability as tfp
import tensorflow as tf


def gaussian_kldiv_loss(prior_variance, args_idx: Tuple[List[int], List[int]]):
    mean_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[0]]))
    logvar_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[1]]))
    @tf.function
    def loss_fun(y_true, y_pred):
        # KL DIVERGENCE
        mean_r = mean_getter(y_pred)
        logvar_r = logvar_getter(y_pred)
        dist_prior = tfp.distributions.Normal(0., tf.sqrt(
            prior_variance), name='RelevantPriorNormal')  # PriorVariance
        dist_posterior = tfp.distributions.Normal(
            mean_r, tf.exp(0.5 * logvar_r), name='RelevantPosteriorNormal')
        return tf.reduce_sum(tfp.distributions.kl_divergence(
            dist_prior, dist_posterior, allow_nan_stats=False, name=None
        ))
    return loss_fun