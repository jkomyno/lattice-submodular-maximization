import numpy as np
from nptyping import NDArray


def max(x: NDArray[int], y: NDArray[int]) -> int:
    """
    Return the coordinate-wise maximum between x and y.
    """
    return np.maximum(x, y)
