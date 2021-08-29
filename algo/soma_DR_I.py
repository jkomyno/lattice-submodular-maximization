import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
import utils


def soma_DR_I(f: Objective, c: NDArray[Int64], r: int, eps: float) -> NDArray[Int64]:
    """
    Implement Soma'18 algorithm for maximizing a DR-submodular monotone function
    over the integer lattice under cardinality constraint.
    :param f: a DR-submodular monotone function
    :param c: the vector upper bound of the lattice domain
    :param r: the cardinality constraint
    :param eps: the error threshold
    """
    y = np.zeros((f.n, ), dtype=int)

    d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    

    while theta >= stop_theta:
        for e in f.V:
            one_e = utils.char_vector(f, e)
            k_max = np.min([c[e] - y[e], r - np.sum(y)])

            # find the maximum integer 1 <= k <= min(c[e] - y[e], r - np.sum(y))
            # with f.margin_gain(k * utils.char_vector(f, e)) >= k * theta
            it = ((k, f.marginal_gain(k * one_e, y)) for k in range(1, k_max + 1))
            it = filter(lambda x: x[1] > x[0] * theta, it)
            it = map(lambda x: x[0], it)
            k = max(it, default=None)

            if k is not None:
                y = y + k * one_e

        theta = theta * (1 - eps)

    return y
