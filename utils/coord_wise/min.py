import numpy as np
from nptyping import NDArray


def min(x: NDArray[int], y: NDArray[int]) -> int:
    """
    Return the coordinate-wise minimum between x and y.
    """
    return np.minimum(x, y)
