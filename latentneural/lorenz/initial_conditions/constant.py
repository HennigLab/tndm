from typing import Callable, Tuple


def constant(x0: float = 0, y0: float = 1,
             z0: float = 1.05) -> Callable[..., Tuple[float, float, float]]:
    """Constant initial conditions

    Args:
        x0 (float, optional): Initial point X coordinate. Defaults to 0.
        y0 (float, optional): Initial point Y coordinate. Defaults to 1.
        z0 (float, optional): Initial point Z coordinate. Defaults to 1.05.
    Returns:
        Callable[..., Tuple(float, float, float)]: A generator for the (x,y,z) initial contidions.
    """
    def callable() -> Tuple[float, float, float]:
        return (x0, y0, z0)
    return callable
