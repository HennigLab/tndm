import tensorflow as tf


def Crop(dim, start, end, **kwargs):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]

    def func(x):
        dimension = dim
        if dimension == -1:
            dimension = len(x.shape) - 1
        if dimension == 0:
            return x[start:end]
        if dimension == 1:
            return x[:, start:end]
        if dimension == 2:
            return x[:, :, start:end]
        if dimension == 3:
            return x[:, :, :, start:end]
        if dimension == 4:
            return x[:, :, :, :, start:end]

    return tf.keras.layers.Lambda(func, **kwargs)