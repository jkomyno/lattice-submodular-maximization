from abc import ABC
from typing import List, AbstractSet


class SetObjective(ABC):
    def __init__(self, ground_set: List[int], b: int):
        """
        Define a new set submodular function over an expanded (n x b) ground set.
        Basically, define a set submodular function that emulates an integer lattice
        submodular function.
        :param ground_set: ground set of f
        :param b: upper bound of the integer lattice domain of f
        """
        self._n = len(ground_set)
        self._b = b
        self._original_ground_set = ground_set
        self._ground_set = set(range(self._n * self._b))

    @property
    def V(self) -> AbstractSet[int]:
        """
        Return the ground set
        """
        return self._ground_set

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
        return self._n * self._b

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
