import numpy as np


def trunc_exp(x: np.ndarray, bound: float = 10):
    """Truncated exponential

    From: https://github.com/catniplab/vlgp

    Args:
        x (np.ndarray): Input x
        bound (float, optional): upper bound of x. Defaults to 10.

    Returns:
        np.ndarray: exp(min(x, bound))
    """
    return np.exp(np.minimum(x, bound))


class struct(dict, object):
    """Helper class allowing us to access hps dictionary more easily."""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            assert False, ("%s does not exist." % key)

    def __setattr__(self, key, value):
        self[key] = value
