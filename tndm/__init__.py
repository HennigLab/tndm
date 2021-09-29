import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import os
from .utils import ArgsParser

if ArgsParser.get_or_default(dict(os.environ), 'CPU_ONLY', 'FALSE') == 'FALSE':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_visible_devices(
            devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

from .models import LFADS, TNDM
