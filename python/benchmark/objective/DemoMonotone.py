import numpy as np
from typing import Tuple
from nptyping import NDArray
from .Objective import Objective


class DemoMonotone(Objective):
    def __init__(self, rng: np.random.Generator, n: int, B: NDArray[int], B_range: Tuple[int, int]):
        """
        Generate a random integer-lattice modular, monotone function
        :param rng: numpy random generator instance
        :param n: size of the ground set
        :param b: upper bound of the integer lattice domain of f
        """
        super().__init__(list(range(n)), B, B_range)

        # reference to the numpy random generator instance
        self.rng = rng

        # generate n random weights sorted in ascending order
        self.w = rng.integers(low=0, high=100, size=n)
        np.sort(self.w)

    def value(self, x: NDArray[int]) -> int:
        """
        Value oracle for demo monotone maximization
        """
        super().value(x)
        return x @ self.w
