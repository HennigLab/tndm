import tensorflow as tf

def poisson_loglike_loss(timestep):
    @tf.function
    def loss_fun(log_f, y_pred):
        # POISSON LOG-LIKELIHOOD
        targets = tf.cast(y_pred, dtype=tf.float32)
        logrates = tf.cast(tf.math.log(timestep) + log_f, tf.float32)  # Timestep
        return tf.reduce_sum(tf.nn.log_poisson_loss(
            targets=targets,
            log_input=logrates, compute_full_loss=True
        ))
    return loss_fun