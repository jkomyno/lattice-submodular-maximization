import numpy as np
from nptyping import NDArray, Int64
from typing import Tuple
from objective import Objective
import utils


def soma_DR_I(f: Objective, c: NDArray[Int64], r: int, eps: float) -> Tuple[NDArray[Int64], float]:
    """
    Implement Soma'18 algorithm for maximizing a DR-submodular monotone function
    over the integer lattice under cardinality constraint.
    :param f: a DR-submodular monotone function
    :param c: the vector upper bound of the lattice domain
    :param r: the cardinality constraint
    :param eps: the error threshold
    """
    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # prev_value keeps track of the value of f(x)
    prev_value = 0

    # norm keeps track of the L-1 norm of x
    norm = 0

    d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    while theta >= stop_theta:
        for e in f.V:
            one_e = utils.char_vector(f, e)
            k_max = np.min([c[e] - x[e], r - norm])

            # find k in k_interval maximal such that f(k * 1_e | x) >= k * theta
            k_range = list(range(1, k_max + 1))
            best_t = utils.binary_search(f, k_range, one_e=one_e, x=x,
                                         prev_value=prev_value, theta=theta)
            
            if best_t is None:
                # no feasible k was found, nothing gets added to x this iteration.
                continue

            k, candidate_x, candidate_value = best_t

            # We add to x the element in the that increases the value of f
            # the most, extracted k times.
            x = candidate_x
            norm += k
            prev_value = candidate_value

        theta = theta * (1 - eps)

    return x, prev_value
