import numpy as np
from nptyping import NDArray, Int64
from .Objective import Objective


class DemoNonMonotone(Objective):
    def __init__(self, rng: np.random.Generator, n: int, b: int = 1):
        """
        Generate a random integer-lattice modular, non-monotone function
        :param rng: numpy random generator instance
        :param n: size of the ground set
        :param b: upper bound of the integer lattice domain of f
        """
        super().__init__(list(range(n)), b)

        # reference to the numpy random generator instance
        self.rng = rng

        # generate n random weights
        self.w = rng.integers(low=-100, high=100, size=n)

    def value(self, x: NDArray[Int64]) -> int:
        """
        Value oracle for demo non-monotone maximization
        """
        return x @ self.w
