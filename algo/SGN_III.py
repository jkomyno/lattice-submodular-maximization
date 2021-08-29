import math
import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Iterator
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

    while np.sum(x) <= r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        q = set(rng.choice(sample_space, size=s, replace=True))

        # We look for targets >= k * theta.
        # Look for max k s.t. f(k * 1_e | x) >= k * theta

        # instead of picking the greedy +1 among the random subset pick the one
        # that maximises the binary search of moving along that component,
        # like a hybrid of the soma and sgm methods.
        # x[e]=x[e]+k where k is argmax of f(x+k*1_e)
        _, e, k = max((
            (f.value(x + k * utils.char_vector(f, e)), e, k)
            for e in q for k in range(1, f.b - x[e] + 1)
        ), key=utils.fst)

        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + k * one_e

        # increment iteration counter
        t += 1

        # theta = max(theta(1-eps), stop_theta)

    assert np.sum(x) <= r
    return x


def SGN_III_it(rng: np.random.Generator,
               f: Objective, r: int, eps: float = None) -> Iterator[NDArray[Int64]]:
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

    while np.sum(x) <= r:
        yield x

        # sub-sampling step
        sample_space = np.where(x < f.b)[0]
        q = rng.choice(sample_space, size=s, replace=True)

        # instead of picking the greedy +1 among the random subset pick the one
        # that maximises the binary search of moving along that component,
        # like a hybrid of the soma and sgm methods.
        # x[e]=x[e]+k where k is argmax of f(x+k*1_e)
        _, e, k = max((
            (f.value(x + k + utils.char_vector(f, e)), e, k)
            for e in q for k in range(1, f.b - x[e] + 1)
        ), key=utils.fst)

        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + k * one_e

    yield x
