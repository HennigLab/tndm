from tensorflow.python.types.core import Value
from latentneural.lfads.adaptive_weights import AdaptiveWeights
import time
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os

from .lfads import LFADS


def train(model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int, 
    train_dataset: tf.Tensor, adaptive_weights: AdaptiveWeights, 
    val_dataset: Optional[tf.Tensor]=None, batch_size: Optional[int]=None, logdir: Optional[str]=None, 
    adaptive_lr: Optional[dict]=None):
    
    assert len(train_dataset) > 0, ValueError('Please provide a non-empty train dataset')
    dims = train_dataset.shape[1:] 
    kwargs=dict()
    if val_dataset is not None:
        assert dims == val_dataset.shape[1:], ValueError('Validation and training datasets must have coherent sizes')
        kwargs = dict(validation_data=(val_dataset, None))

    model = LFADS(
        neural_space=dims[-1],
        **model_settings
    )

    callbacks = [adaptive_weights]
    if logdir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))
    if adaptive_lr is not None:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', **adaptive_lr))

    model.build(input_shape=[None] + list(dims))

    model.compile(
        optimizer=optimizer,
        loss_weights=adaptive_weights.w
    )

    try:
        model.fit(
            x=train_dataset,
            y=None,
            callbacks=callbacks,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
    except KeyboardInterrupt:
        return model

    return model