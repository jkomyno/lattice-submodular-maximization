import numpy as np
from nptyping import NDArray, Int64
from objective import Objective


def char_vector(f: Objective, e: int) -> NDArray[Int64]:
    """
    Return the n-dimensional characteristic vector with 1 on coordinate e.
    :param f: integer-lattice submodular function
    :param e: coordinate of the characteristic vector that should be set to 1
    """
    return (np.in1d(f.V, [e]) * 1).astype(np.int64)
