import numpy as np
from typing import Iterator, Tuple, Set
from objective import Objective
from set_objective import SetObjective
import set_algo
from utils import bridge


def SSG(rng: np.random.Generator,
        f: Objective, r: int, eps: float = None) -> Tuple[Set[int], int]:
    """
    Simulated StochasticGreedy algorithm in the integer lattice domain.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: error threshold
    """
    f_prime: SetObjective = bridge.to_set_objective(f)
    S = set_algo.stochastic_greedy(rng, f_prime, r, eps)
    return S, f_prime.value(S)


def SSG_it(rng: np.random.Generator,
           f: Objective, r: int, eps: float = None) -> Iterator[Set[int]]:
    """
    Simulated StochasticGreedy algorithm in the integer lattice domain.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: error threshold
    """
    f_prime: SetObjective = bridge.to_set_objective(f)
    return set_algo.stochastic_greedy_it(rng, f_prime, r, eps)
