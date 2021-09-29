import numpy as np
from typing import Tuple, Callable


def uniform(low: float = -1,
            high: float = 1) -> Callable[..., Tuple[float, float, float]]:
    """Uniform initial conditions

    Args:
        low (float, optional): Lower bound of the uniform distribution. Defaults to -1.
        high (float, optional): Upper bound of the uniform distribution. Defaults to 1.
    Returns:
        Callable[..., Tuple(float, float, float)]: A generator for the (x,y,z) initial contidions.
    """
    def callable() -> Tuple[float, float, float]:
        return (
            np.random.uniform(low=low, high=high),
            np.random.uniform(low=low, high=high),
            np.random.uniform(low=low, high=high),
        )

    return callable
