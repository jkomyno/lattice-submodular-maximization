import numpy as np
from nptyping import NDArray, Int64
from typing import Tuple
from objective import Objective
import utils


def SGL_I(rng: np.random.Generator,
          f: Objective, r: int, eps: float = None) -> Tuple[NDArray[Int64], float]:
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
    s = utils.compute_sample_size(n=f.n, r=r, eps=eps)

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # prev_value keeps track of the value of f(x)
    prev_value = 0

    for _ in range(r):
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = rng.choice(sample_space, size=min(s, len(sample_space)), replace=False)
        Q_one = map(lambda e: utils.char_vector(f, e), Q)

        # e \gets \argmax_{e \in Q} f(\mathbf{1}_e\ |\ \mathbf{x}).
        # We add to x the element e in the sample Q that increases the value of f
        # the most.
        x, prev_value, marginal_gain = max((
            (
                candidate_x := x + one_e,
                candidate_value := f.value(candidate_x),
                candidate_value - prev_value
            ) for one_e in Q_one
        ), key=utils.trd)

    assert np.sum(x) == r
    return x, prev_value
