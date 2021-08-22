import numpy as np
from nptyping import NDArray, Int64


def min(x: NDArray[Int64], y: NDArray[Int64]) -> int:
    """
    Return the coordinate-wise minimum between x and y.
    """
    return np.minimum(x, y)
