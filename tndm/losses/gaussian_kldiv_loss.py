import tensorflow_probability as tfp
import tensorflow as tf

def gaussian_kldiv_loss(prior_variance):
    '''
    Input format: (g_?, mean_?, logvar_?)
    '''
    mean_getter = lambda x: x[1]
    logvar_getter = lambda x: x[2]
    @tf.function
    def loss_fun(g_x):
        # KL DIVERGENCE
        mean_r = mean_getter(g_x)
        logvar_r = logvar_getter(g_x)
        dist_prior = tfp.distributions.Normal(0., tf.sqrt(
            prior_variance), name='RelevantPriorNormal')  # PriorVariance
        dist_posterior = tfp.distributions.Normal(
            mean_r, tf.exp(0.5 * logvar_r), name='RelevantPosteriorNormal')
        return tf.reduce_sum(tfp.distributions.kl_divergence(
            dist_prior, dist_posterior, allow_nan_stats=False, name=None
        ))
    return loss_fun