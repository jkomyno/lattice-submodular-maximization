import numpy as np
from nptyping import NDArray, Int64


def leq(x: NDArray[Int64], y: NDArray[Int64]) -> bool:
    """
    Return True iff x is coordinate-wise less than or equal to y.
    """
    return np.all(x <= y)
