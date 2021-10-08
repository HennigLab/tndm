import tensorflow_probability as tfp
import tensorflow as tf

def covariance_loss(disentanglement_batches):
    '''
    Input format: (g_?, mean_?, logvar_?)
    '''
    mean_getter = lambda x: x[1]
    logvar_getter = lambda x: x[2]
    @tf.function
    def loss_fun(g_r,g_i):
        gr_mean = mean_getter(g_r)
        gr_logvar = logvar_getter(g_r)
        gi_mean = mean_getter(g_i)
        gi_logvar = logvar_getter(g_i)
        sample_r = tf.random.normal(shape=[disentanglement_batches, gr_mean.shape[-2], gr_mean.shape[-1]]) * tf.exp(gr_logvar * .5) + gr_mean
        sample_i = tf.random.normal(shape=[disentanglement_batches, gi_mean.shape[-2], gi_mean.shape[-1]]) * tf.exp(gi_logvar * .5) + gi_mean
        sample_r = tf.reshape(sample_r, [sample_r.shape[0] * sample_r.shape[1], sample_r.shape[2]])
        sample_i = tf.reshape(sample_i, [sample_i.shape[0] * sample_i.shape[1], sample_i.shape[2]])
        cov = tfp.stats.covariance(sample_r, sample_i, sample_axis=0, event_axis=-1)
        squared_cov_matrix_rel_irel = cov ** 2
        return tf.reduce_mean(squared_cov_matrix_rel_irel) * gr_mean.shape[-2] # Multiplied by batch size, to be proportional to it like most other losses
    return loss_fun