from typing import Set
from nptyping import NDArray, Int64
from objective import Objective
from nptyping import NDArray, Int64


def to_set(f: Objective, x: NDArray[Int64]) -> Set[int]:
    """
    Convert an integer lattice solution to a set submodular solution.
    :param f: integer lattice submodular function
    :param x: integer lattice solution
    """
    S = set()
    for i, e in enumerate(x):
        for c in range(e):
            S.add(i + c * f.n)

    return S
