import numpy as np
from nptyping import NDArray
from typing import Tuple
from ..objective import Objective
from .. import utils


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

    for _ in range(r):
        V = np.copy(np.where(x < f.B)[0])
        rng.shuffle(V)

        # split list V in batches of size at most s
        batches = utils.split_list(V, s)

        for Q in batches:
            # lazy list of (e, k_max), where k_max is the highest k such that
            # f(x + k * 1_e) >= f(x) while making sure that the cardinality constraint
            # is respected
            e_k_max: Tuple[int, int] = [
                (e, min(f.B[e] - x[e], r - np.sum(x)))
                for e in Q
            ]

            # lazy list of (one_e, max_k)
            # one_e_k_max = utils.map_fst(lambda e: utils.char_vector(f, e), e_k_max)
            one_e_k_max = map(lambda ek: (utils.char_vector(f, ek[0]) , ek[1]), e_k_max)

            # We add k copies of the element in the sample q that increases the value of f
            # the most to the solution x.
            x, prev_value, marginal_gain, k = max((
                (
                    candidate_x := x + k * one_e,
                    candidate_value := f.value(candidate_x),
                    candidate_value - prev_value,
                    k
                ) for one_e, k in one_e_k_max
            ), key=utils.trd)

            if np.sum(x) == r:
                break

        if np.sum(x) == r:
            break


    assert np.sum(x) <= r
    print(f'SGL-II   n={f.n}; B={f.B_range}; r={r}; norm={np.sum(x)}')
    return x, prev_value
