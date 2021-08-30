import math
import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Iterator, Tuple
import utils


def SGL_II(rng: np.random.Generator,
           f: Objective, r: int, eps: float = None) -> NDArray[Int64]:
    """
    Randomized algorithm for DR-ubmodular maximization of monotone functions
    defined on the integer lattice with cardinality constraints.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: non-negative error threshold
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = math.ceil((f.n / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # norm keeps track of the L-1 norm of x
    norm = 0

    # iteration counter
    t = 0

    while norm < r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = set(rng.choice(sample_space, size=s, replace=True))

        # lazy list of (e, best_k), where best_k is the highest k such that
        # f(x + k * 1_e) >= f(x) while making sure that the cardinality constraint
        # is respected

        e_k_best: Tuple[int, int] = [
            (e, min(f.b, r - norm) - x[e])
            for e in Q
        ]

        print(f'e_k_best: {e_k_best}')

        # lazy list of (one_e, best_k)
        # one_e_k_best = utils.map_fst(lambda e: utils.char_vector(f, e), e_k_best)
        one_e_k_best = map(lambda ek: (utils.char_vector(f, ek[0]) , ek[1]), e_k_best)

        _, one_e, k = max(((f.marginal_gain(k * one_e, x), one_e, k)
                           for one_e, k in one_e_k_best), key=utils.fst)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + k * one_e
        norm += k

        # increment iteration counter
        t += 1

    assert np.sum(x) <= r
    return x
