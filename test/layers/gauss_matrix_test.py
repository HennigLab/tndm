import tensorflow as tf
import numpy as np

from tndm.layers import GaussMatrix

def test_gauss_matrix():
    data = np.asarray(
        [
            [
                [0, 0, 0],      # var1, t=0
                [0, 0, 0],      # var1, t=1
                [1, 1, 1],      # var1, t=2
                [0, 0, 0],      # var1, t=3
            ],
            [
                [1, 1, 1],      # var2, t=0
                [1, 1, 1],      # var2, t=1
                [1, 1, 1],      # var2, t=2
                [1, 1, 1],      # var2, t=3
            ]
        ]
    )

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(4, 3,)))
    beh_dec = GaussMatrix(2, kernel_init='ones', bias_init='zeros')
    model.add(beh_dec)

    out = model(data)
    print(out)

    tf.debugging.assert_equal(
        out,
        np.asarray(
            [
              [
                [ 2.769349,  2.769349],                                                                
                [ 2.940596,  2.940596],                                                                
                [ 3.      ,  3.      ],                                                                
                [ 2.940596,  2.940596],
              ],                                                               
              [
                [11.215755, 11.215755],                                                                
                [11.65054,  11.65054 ],                                                                
                [11.650541, 11.650541],                                                                
                [11.215755, 11.215755],
              ]
            ],
            dtype='float32'
        )
    )
