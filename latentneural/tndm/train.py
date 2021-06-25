from tensorflow.python.types.core import Value
from latentneural.lfads.adaptive_weights import AdaptiveWeights
import time
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os

from .tndm import TNDM


tf.config.run_functions_eagerly(True)

def train(model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int, 
    train_dataset: Tuple[tf.Tensor, tf.Tensor], adaptive_weights: AdaptiveWeights, 
    val_dataset: Optional[Tuple[tf.Tensor, tf.Tensor]]=None, batch_size: Optional[int]=None, logdir: Optional[str]=None, 
    adaptive_lr: Optional[dict]=None):
    
    assert len(train_dataset) > 1, ValueError('Please provide a non-empty train dataset')
    neural_dims = train_dataset[0].shape[1:] 
    behavioural_dims = train_dataset[1].shape[1:] 

    if val_dataset is not None:
        if len(val_dataset) > 1:
            assert neural_dims == val_dataset[0].shape[1:], ValueError('Validation and training datasets must have coherent sizes')
            assert behavioural_dims == val_dataset[1].shape[1:], ValueError('Validation and training datasets must have coherent sizes')
            kwargs = dict(validation_data=(val_dataset[0], val_dataset[1]))
        else:
            kwargs = dict(validation_data=(None, None))
    else:
        kwargs={}

    model = TNDM(
        neural_space=neural_dims[-1],
        behavioural_space=behavioural_dims[-1],
        **model_settings
    )

    callbacks = [adaptive_weights]
    if logdir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))
    if adaptive_lr is not None:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', **adaptive_lr))

    model.build(input_shape=[None] + list(neural_dims))

    model.compile(
        optimizer=optimizer,
        loss_weights=adaptive_weights.w
    )

    try:
        model.fit(
            x=train_dataset[0],
            y=train_dataset[1],
            callbacks=callbacks,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
    except KeyboardInterrupt:
        return model

    return model