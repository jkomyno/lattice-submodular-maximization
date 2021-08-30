import math
import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Iterator, List, Tuple
import utils


def SGN_III(rng: np.random.Generator,
            f: Objective, r: int, eps: float = None) -> NDArray[Int64]:
    """
    Randomized algorithm for integer-lattice submodular maximization of monotone functions with cardinality
    constraints in linear time.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: approximation threshold in (0, 0.5)
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = math.ceil((f.n / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # iteration counter
    t = 0

    # d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    # theta = d
    # stop_theta = (eps / r) * d

    # norm keeps track of the L-1 norm of x
    norm = 0

    while norm < r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = set(rng.choice(sample_space, size=s, replace=True))

        # find k e maximal for f such that x[e] + k * 1[e] <= b
        # AND x[e] + k * 1[e] <= r - \|x\|_1
        # --->
        # x[e] + k * 1[e] <= min(b, r - \|x\|_1)
        # k * 1[e] <= min(b, r - \|x\|_1) - x[e]

        e_ks: List(Tuple[int, Iterator[int]]) = (
            (e, range(1, min(f.b, r - norm) - x[e] + 1))
            for e in Q
        )
        one_e_ks = map(lambda ek: (utils.char_vector(f, ek[0]) , ek[1]), e_ks)

        # instead of picking the greedy +1 among the random subset pick the one
        # that maximises the binary search of moving along that component,
        # like a hybrid of the soma and sgm methods.
        # x[e]=x[e]+k where k is argmax of f(x+k*1_e)
        _, one_e, k = max((
            (f.value(x + k * one_e), one_e, k)
            for one_e, ks in one_e_ks
            for k in ks
        ), key=utils.fst)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + k * one_e
        norm += k

        # increment iteration counter
        t += 1

        # theta = max(theta(1-eps), stop_theta)

    assert norm <= r
    return x
