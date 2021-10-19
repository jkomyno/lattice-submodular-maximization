import numpy as np
from nptyping import NDArray
from .Objective import Objective


class DemoNonMonotone(Objective):
    def __init__(self, rng: np.random.Generator, n: int, B: NDArray[int]):
        """
        Generate a random integer-lattice modular, non-monotone function
        :param rng: numpy random generator instance
        :param n: size of the ground set
        :param b: upper bound of the integer lattice domain of f
        """
        super().__init__(list(range(n)), B)

        # reference to the numpy random generator instance
        self.rng = rng

        # generate n random weights
        self.w = rng.integers(low=-100, high=100, size=n)

    def value(self, x: NDArray[int]) -> int:
        """
        Value oracle for demo non-monotone maximization
        """
        super().value(x)
        return x @ self.w
