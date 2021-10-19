import numpy as np
from nptyping import NDArray
from objective import Objective
from typing import Tuple
import utils


def SGL_II_b(rng: np.random.Generator,
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

        # We add to x the element in the sample q that increases the value of f
        # the most. k might also be 0.
        x, prev_value, marginal_gain, k = max((
            (
                candidate_x := x + k * utils.char_vector(f, e),
                candidate_value := f.value(candidate_x),
                candidate_value - prev_value,
                k
            )
            for e in Q
            for k in range(min(f.B[e] - x[e], r - norm) + 1)
        ), key=utils.trd)

        # update norm
        norm = np.sum(x)

        # increment iteration counter
        t += 1

    assert np.sum(x) <= r
    return x, prev_value
