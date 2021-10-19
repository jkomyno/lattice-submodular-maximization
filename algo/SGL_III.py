import numpy as np
from nptyping import NDArray
from objective import Objective
from typing import Tuple
import utils


def SGL_III(rng: np.random.Generator,
            f: Objective, r: int, eps: float) -> Tuple[NDArray[int], float]:
    """
    Randomized algorithm for integer-lattice submodular maximization of monotone functions with cardinality
    constraints in linear time.
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

    # iteration counter
    t = 0

    # norm keeps track of the L-1 norm of x
    norm = 0

    d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    while norm < r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = rng.choice(sample_space, size=min(s * 8, len(sample_space)), replace=False)
        Q_one = list(map(lambda e: utils.char_vector(f, e), Q))

        # potentially add multiple copies of every item in Q
        for i, e in enumerate(Q):
            one_e = Q_one[i]
            k_max = np.min([f.b - x[e], r - norm])
            k_range = list(range(1, k_max + 1))

            # find k in k_interval maximal such that f(k * 1_e | x) >= k * theta
            best_t = utils.binary_search(f, k_range, one_e=one_e, x=x,
                                         prev_value=prev_value, theta=theta)

            if best_t is None:
                # no feasible k was found, nothing gets added to x this iteration.
                continue

            k, candidate_x, candidate_value = best_t
            
            # We add to x the element in the sample q that increases the value of f
            # the most, extracted k times.
            x = candidate_x
            norm += k
            prev_value = candidate_value

        # update theta
        theta = max(theta * (1 - eps), stop_theta)

        # increment iteration counter
        t += 1

    assert norm <= r
    return x, prev_value
