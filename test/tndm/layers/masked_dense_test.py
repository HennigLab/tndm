import tensorflow as tf
import numpy as np

from latentneural.tndm.layers import MaskedDense


def test_masked_dense():
    data = np.asarray(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
                [30, 31, 32],
            ]
        ]
    )

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(4,3,)))
    model.add(MaskedDense(2, kernel_initializer=tf.keras.initializers.Constant(1)))

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