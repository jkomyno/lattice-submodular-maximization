import numpy as np
from nptyping import NDArray
from typing import Tuple
from ..objective import Objective
from .. import utils


def SGL_I(rng: np.random.Generator,
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
        V = np.copy(f.V[x < f.B])
        rng.shuffle(V)

        # split list V in batches of size at most s
        batches = utils.split_list(V, s)

        for Q in batches:
            Q_one = map(lambda e: utils.char_vector(f, e), Q)

            # e \gets \argmax_{e \in Q} f(\symbf{1}_e\ |\ \symbf{x}).
            # We add to x the element e in the sample Q that increases the value of f
            # the most.
            x, prev_value, marginal_gain = max((
                (
                    candidate_x := x + one_e,
                    candidate_value := f.value(candidate_x),
                    candidate_value - prev_value
                ) for one_e in Q_one
            ), key=utils.trd)

            if np.sum(x) == r:
                break

    assert np.sum(x) <= r
    print(f'SGL-I    t={t}; n={f.n}; B={f.B_range}; r={r}; norm={norm}')
    return x, prev_value
