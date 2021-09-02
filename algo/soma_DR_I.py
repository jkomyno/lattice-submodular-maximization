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

    def constraint_k_theta(prev_value: float, theta: float):
        """
        Higher-order function to be applied to filter.
        :param prev_value: value of f(x) at the previous iteration
        :param theta: decreasing threshold
        """
        def helper(t: Tuple[int, NDArray[Int64], float]) -> bool:
            """
            Returns true iff f(k * 1_e | x) > k * theta
            :param t: (k, x + k * 1_e, f(x + k * 1_e)) tuple
            """
            k, _, candidate_value = t
            return candidate_value - prev_value > k * theta

        return helper

    while theta >= stop_theta:
        for e in f.V:
            one_e = utils.char_vector(f, e)
            k_max = np.min([c[e] - x[e], r - norm])

            # find the maximum integer 1 <= k <= min(c[e] - x[e], r - np.sum(x))
            # with f.margin_gain(k * utils.char_vector(f, e)) >= k * theta
            lazy_list = (
                (k, candidate_x := x + k * one_e, f.value(candidate_x))
                for k in range (1, k_max + 1)
            )
            lazy_list = filter(constraint_k_theta(prev_value, theta), lazy_list)
            k, candidate_x, candidate_value = max(lazy_list, key=utils.fst, default=(None, None, None))

            if k is not None:
                # We add to x the element in the that increases the value of f
                # the most, extracted k times.
                x = candidate_x
                norm += k
                prev_value = candidate_value

        theta = theta * (1 - eps)

    return x, prev_value
