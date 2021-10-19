import numpy as np
from nptyping import NDArray
from objective import Objective
from typing import Tuple
import utils


def SGL_II(rng: np.random.Generator,
           f: Objective, r: int, eps: float) -> Tuple[NDArray[int], float]:
    """
    Randomized algorithm for DR-ubmodular maximization of monotone functions
    defined on the integer lattice with cardinality constraints.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: non-negative error threshold
    """
    # compute s, the sample size
    s = utils.compute_sample_size(n=f.n, r=r, eps=eps)

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # prev_value keeps track of the value of f(x)
    prev_value = 0

    # norm keeps track of the L-1 norm of x
    norm = 0

    # iteration counter
    t = 0

    while norm < r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.B)[0]
        Q = rng.choice(sample_space, size=min(s, len(sample_space)), replace=False)

        # lazy list of (e, best_k), where best_k is the highest k such that
        # f(x + k * 1_e) >= f(x) while making sure that the cardinality constraint
        # is respected
        e_k_best: Tuple[int, int] = [
            (e, min(f.B[e] - x[e], r - norm))
            for e in Q
        ]

        # lazy list of (one_e, best_k)
        # one_e_k_best = utils.map_fst(lambda e: utils.char_vector(f, e), e_k_best)
        one_e_k_best = map(lambda ek: (utils.char_vector(f, ek[0]) , ek[1]), e_k_best)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x, prev_value, marginal_gain, k = max((
            (
                candidate_x := x + k * one_e,
                candidate_value := f.value(candidate_x),
                candidate_value - prev_value,
                k
            ) for one_e, k in one_e_k_best
        ), key=utils.trd)

        # update norm
        norm += k

        # increment iteration counter
        t += 1

    assert np.sum(x) <= r
    return x, prev_value
