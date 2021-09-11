import numpy as np
from nptyping import NDArray, Int64
from typing import Tuple
from objective import Objective
import utils


def soma_DR_I_b(f: Objective, c: NDArray[Int64], r: int, eps: float) -> Tuple[NDArray[Int64], float]:
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

    def constraint_k_theta(prev_value: float, theta: float):
        """
        Higher-order function to be applied to filter.
        :param prev_value: value of f(x) at the previous iteration
        :param theta: decreasing threshold
        """
        def helper(t: Tuple[int, NDArray[Int64], float]) -> bool:
            """
            Returns true iff f(k * 1_e | x) >= k * theta
            :param t: (k, x + k * 1_e, f(x + k * 1_e)) tuple
            """
            k, _, candidate_value = t
            return candidate_value - prev_value >= k * theta

        return helper

    while norm < r:
        for e in f.V:
            one_e = utils.char_vector(f, e)
            k_max = np.min([c[e] - x[e], r - norm])

            # find k that maximizes f(k * 1_e | x)
            # TODO

            if k == None:
                # no feasible k was found, nothing gets added to x this iteration.
                continue

            if candidate_value >= prev_value:
                # We add to x the element in the that increases the value of f
                # the most, extracted k times.
                x = candidate_x
                norm += k
                prev_value = candidate_value

    return x, prev_value
