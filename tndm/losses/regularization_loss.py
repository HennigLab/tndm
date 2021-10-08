import tensorflow as tf

def regularization_loss():
    @tf.function
    def loss_fun(losses):
        return tf.reduce_sum(losses)
    return loss_fun
