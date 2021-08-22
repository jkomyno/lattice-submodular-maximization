import numpy as np
from abc import ABC
from typing import List
from nptyping import NDArray, Int64


class Objective(ABC):
    def __init__(self, ground_set: List[int], b: int):
        """
        Define a new integer-lattice submodular function.
        :param ground_set: ground set of f
        :param b: upper bound of the integer lattice domain of f
        """
        self._ground_set = ground_set
        self._n = len(ground_set)
        self._b = b
        self._B = np.full((self._n,), b)

    @property
    def V(self) -> List[int]:
        """
        Return the ground set
        """
        return self._ground_set

    @property
    def B(self) -> NDArray[Int64]:
        """
        Return the upper bound vector of the integer lattice domain.
        """
        return self._B

    @property
    def b(self) -> int:
        """
        Return the upper bound scalar value of the integer lattice domain.
        """
        return self._b

    @property
    def n(self) -> int:
        """
        Return the size of the ground set
        """
        return self._n

    def value(self, x: NDArray[Int64]) -> int:
        """
        Value oracle for the submodular problem.
        :param x: subset of the ground set
        :return: value oracle for S in the submodular problem
        """
        pass

    def marginal_gain(self, x: NDArray[Int64], y: NDArray[Int64]) -> int:
        """
        Value oracle for f(x | y) := f(x + y) - f(y)
        """
        return self.value(x + y) - self.value(y)
