import tensorflow as tf
import numpy as np

from tndm.layers import MaskedDense


def test_masked_dense():
    data = np.asarray(
        [
            [
                [1, 2, 3],      # t=0
                [4, 5, 6],      # t=1
                [7, 8, 9],      # t=2
                [10, 11, 12],   # t=3
            ],
            [
                [21, 22, 23],   # t=4
                [24, 25, 26],   # t=5
                [27, 28, 29],   # t=6
                [30, 31, 32],   # t=7
            ]
        ]
    )

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(4, 3,)))
    model.add(
        MaskedDense(
            2,
            kernel_initializer=tf.keras.initializers.Constant(1)))

    out = model(data)
    tf.debugging.assert_equal(
        out,
        np.asarray(
            [
                [
                    [6., 6],
                    [21, 21],
                    [45, 45],
                    [78, 78],
                ],
                [
                    [66, 66],
                    [141, 141],
                    [225, 225],
                    [318, 318],
                ]
            ],
            dtype='float32'
        )
    )
