import numpy as np
from typing import AbstractSet
from nptyping import NDArray
from collections import Counter
from set_objective import SetObjective


def to_integer_lattice(f: SetObjective,
                       S: AbstractSet[int]) -> NDArray[int]:
    """
    Convert a set submodular solution to an integer lattice solution.
    :param f: set submodular function
    :param S: set submodular solution
    """
    # n is the size of the ground set in the integer lattice
    n = f.original_n
    counter = Counter((e % n for e in S))
    x = np.zeros((n, ), dtype=int)

    for e, c in counter.items():
        x[e] = c

    return x
