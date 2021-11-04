from abc import ABC
from typing import List, Tuple
from nptyping import NDArray


class Objective(ABC):
    def __init__(self, ground_set: List[int], B: NDArray[int], B_range: Tuple[int, int]):
        """
        Define a new integer-lattice submodular function.
        :param ground_set: ground set of f
        :param b: upper bound of the integer lattice domain of f
        """
        self._ground_set = ground_set
        self._n = len(ground_set)
        self._B = B
        self._B_range = B_range

        # keep track of the number of oracle calls
        self._n_calls = 0

    @property
    def V(self) -> List[int]:
        """
        Return the ground set
        """
        return self._ground_set

    @property
    def B(self) -> NDArray[int]:
        """
        Return the upper bound vector of the integer lattice domain.
        """
        return self._B

    @property
    def B_range(self) -> Tuple[int, int]:
        """
        Return the range of the upper bound vector of the integer lattice domain.
        """
        return self._B_range

    @property
    def n(self) -> int:
        """
        Return the size of the ground set
        """
        return self._n

    @property
    def n_calls(self) -> int:
        """
        Return the number of oracle calls
        """
        return self._n_calls

    def value(self, x: NDArray[int]) -> int:
        """
        Value oracle for the submodular problem.
        :param x: subset of the ground set
        :return: value oracle for S in the submodular problem
        """
        self._n_calls += 1
        
        if self._n_calls % 10000 == 0:
            print(f'Oracle calls: {self._n_calls}')

        return None

    def marginal_gain(self, x: NDArray[int], y: NDArray[int]) -> int:
        """
        Value oracle for f(x | y) := f(x + y) - f(y)
        """
        return self.value(x + y) - self.value(y)

    def reset(self):
        """
        Reset the number of oracle calls to zero.
        """
        self._n_calls = 0
