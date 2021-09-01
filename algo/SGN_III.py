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
    s = utils.compute_sample_size(n=f.n, r=r, eps=eps)

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # prev_value keeps track of the value of f(x)
    prev_value = 0

    # iteration counter
    t = 0

    d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    # norm keeps track of the L-1 norm of x
    norm = 0

    while norm < r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = rng.choice(sample_space, size=min(s, len(sample_space)), replace=False)

        for e in Q:
            one_e = utils.char_vector(f, e)
            
            # find k in k_interval maximal such that f(k * 1_e | x) >= k * theta
            k, candidate_x, candidate_value = max((
                (k, candidate_x := x + k * one_e, candidate_value := f.value(candidate_x))
                for k in range(1, min(f.b - x[e], r - norm) + 1)
                if candidate_value - prev_value >= k * theta
            ), key=utils.snd, default=(None, None))

            k = max((
                (mg := f.marginal_gain(k * one_e, x), k)
                for k in range(1, min(f.b - x[e], r - norm) + 1)
                if mg >= k * theta
            ), key=utils.snd, default=(None, None, None))

            if k == None:
                continue

            if candidate_value >= prev_value:
                # We add to x the element in the sample q that increases the value of f
                # the most.
                x = candidate_x
                norm += k

        # update theta
        theta = max(theta * (1 - eps), stop_theta)

        # increment iteration counter
        t += 1

    assert norm <= r
    return x
