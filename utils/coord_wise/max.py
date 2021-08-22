import numpy as np
from nptyping import NDArray, Int64


def max(x: NDArray[Int64], y: NDArray[Int64]) -> int:
    """
    Return the coordinate-wise maximum between x and y.
    """
    return np.maximum(x, y)
