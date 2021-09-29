from typing import Tuple, List
import tensorflow_probability as tfp
import tensorflow as tf


def covariance_loss(disentanglement_batches, args_idx: Tuple[List[int], List[int], List[int], List[int]]):
    gr_mean_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[0]]))
    gr_logvar_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[1]]))
    gi_mean_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[2]]))
    gi_logvar_getter = eval('lambda x: x' + ''.join(['[%d]' % (n) for n in args_idx[3]]))
    @tf.function
    def loss_fun(y_true, y_pred):
        gr_mean = gr_mean_getter(y_pred)
        gr_logvar = gr_logvar_getter(y_pred)
        gi_mean = gi_mean_getter(y_pred)
        gi_logvar = gi_logvar_getter(y_pred)
        sample_r = tf.random.normal(shape=[disentanglement_batches, gr_mean.shape[-2], gr_mean.shape[-1]]) * tf.exp(gr_logvar * .5) + gr_mean
        sample_i = tf.random.normal(shape=[disentanglement_batches, gi_mean.shape[-2], gi_mean.shape[-1]]) * tf.exp(gi_logvar * .5) + gi_mean
        sample_r = tf.reshape(sample_r, [sample_r.shape[0] * sample_r.shape[1], sample_r.shape[2]])
        sample_i = tf.reshape(sample_i, [sample_i.shape[0] * sample_i.shape[1], sample_i.shape[2]])
        cov = tfp.stats.covariance(sample_r, sample_i, sample_axis=0, event_axis=-1)
        squared_cov_matrix_rel_irel = cov ** 2
        return tf.reduce_mean(squared_cov_matrix_rel_irel) * gr_mean.shape[-2] # Multiplied by batch size, to be proportional to it like most other losses
    return loss_fun