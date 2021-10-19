import numpy as np
from abc import ABC
from typing import List, AbstractSet
from nptyping import NDArray


class SetObjective(ABC):
    def __init__(self, ground_set: List[int], B: NDArray[int]):
        """
        Define a new set submodular function over an expanded (n x b) ground set.
        Basically, define a set submodular function that emulates an integer lattice
        submodular function.
        :param ground_set: ground set of f
        :param b: upper bound of the integer lattice domain of f
        """
        self._n = len(ground_set)
        self._B = B
        self._original_ground_set = ground_set
        self._ground_set = set(range(np.sum(B)))

    @property
    def V(self) -> AbstractSet[int]:
        """
        Return the ground set
        """
        return self._ground_set

    @property
    def n(self) -> int:
        """
        Return the size of the ground set
        """
        return len(self._ground_set)

    @property
    def original_n(self) -> int:
        """
        Return the size of the ground set in the integer lattice
        """
        return self._n

    def value(self, S: AbstractSet[int]) -> int:
        """
        Value oracle for the submodular problem.
        :param S: subset of the ground set
        :return: value oracle for S in the submodular problem
        """
        pass

    def marginal_gain(self, S: AbstractSet[int], T: AbstractSet[int]) -> int:
        """
        Value oracle for f(S | T) := f(S \cup T) - f(S)
        """
        return self.value(S | T) - self.value(S)
