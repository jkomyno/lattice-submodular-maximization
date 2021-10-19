import numpy as np
from typing import Tuple
from nptyping import NDArray
from objective import Objective
from set_objective import SetObjective
import set_algo
from utils import bridge


def SSG(rng: np.random.Generator,
        f: Objective, r: int, eps: float = None) -> Tuple[NDArray[int], int]:
    """
    Simulated StochasticGreedy algorithm in the integer lattice domain.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: error threshold
    """
    f_prime: SetObjective = bridge.to_set_objective(f)
    S, value = set_algo.stochastic_greedy(rng, f_prime, r, eps)
    x = bridge.to_integer_lattice(f_prime, S)

    return x, value
