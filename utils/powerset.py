import numpy as np
import itertools
from objective import Objective
from nptyping import NDArray
from typing import Iterator


def powerset(f: Objective) -> Iterator[NDArray[int]]:
    """
    Inumerate b^n possible vectors in the integer lattice.
    :param f: integer-lattice submodular function objective
    """
    return map(lambda t: np.array([*t]),
               itertools.product(range(f.b + 1), repeat=f.n))
