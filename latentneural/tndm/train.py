import time
import tensorflow as tf
from typing import List, Tuple

from .loss import compute_loss


tf.config.run_functions_eagerly(True)

@tf.function
def train_step(model: tf.keras.Model, neural: tf.Tensor, behaviour: tf.Tensor, optimizer: tf.optimizers.Optimizer, 
    coefficients: List[float]=[1,1,1,1]):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, neural, behaviour, coefficients=coefficients)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train_model(model: tf.keras.Model, optimizer: tf.optimizers.Optimizer, epochs: int, 
    train_dataset: List[Tuple[tf.Tensor, tf.Tensor]], val_dataset: List[Tuple[tf.Tensor, tf.Tensor]], 
    coefficients: List[float]=[1,1,1,1]):
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for (neural, behaviour) in train_dataset:
            train_step(model, neural, behaviour, optimizer, coefficients=coefficients)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for (neural, behaviour) in val_dataset:
            loss(compute_loss(model, neural, behaviour, coefficients=coefficients))
        elbo = -loss.result()
        
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
