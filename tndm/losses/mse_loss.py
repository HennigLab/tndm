import tensorflow as tf

def mse_loss():
    '''
    MSE loss with backward compatibility
    with older TNDM versions
    (corrects the constant)
    TODO: remove this wrapper after testing is complete
    '''
    @tf.function
    def loss_fun(y_true, y_pred):
        tf_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        return tf.cast(0.5*tf_mse(y_true,y_pred)*y_true.shape[-1],dtype=tf.float32)
    return loss_fun