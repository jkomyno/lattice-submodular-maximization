import numpy as np
from nptyping import NDArray


def leq(x: NDArray[int], y: NDArray[int]) -> bool:
    """
    Return True iff x is coordinate-wise less than or equal to y.
    """
    return np.all(x <= y)
