import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Union
import utils


def soma_II(f: Objective, c: NDArray[Int64], r: int, eps: float) -> NDArray[Int64]:
    """
    Implement Soma'18 algorithm for maximizing a submodular monotone function
    over the integer lattice under cardinality constraint.
    :param f: a DR-submodular monotone function
    :param c: the vector upper bound of the lattice domain
    :param r: the cardinality constraint
    :param eps: the error threshold
    """
    y = np.zeros((f.n, ), dtype=int)

    d = max((f.value(c[e] * utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    while theta >= stop_theta:
        for e in f.V:
            one_e = utils.char_vector(f, e)
            k_max = np.min([c[e] - y[e], r - np.sum(y)])
            
            k = binary_search_lattice(f=f, one_e=one_e, theta=theta, k_max=k_max, eps=eps)

            if k is not None:
                y = y + k * one_e

        theta = theta * (1 - eps)

    return y


def binary_search_lattice(f: Objective, one_e: NDArray[Int64], theta: float,
                          k_max: int, eps: float) -> Union[float, None]:
    # find the minimum k_min with 0 <= k_min <= k_max such that f(k_min * one_e) > 0.
    it = ((k_min, f.value(k_min * one_e)) for k_min in range(0, k_max + 1))
    it = filter(lambda x: x[1] > 0, it)
    it = map(lambda x: x[0], it)
    k_min = min(it, default=None)

    if k_min is None:
      return None

    h = f.value(k_max * one_e)
    stop_h = (1 - eps) * f.value(k_min * one_e)

    while h >= stop_h:
        it = ((k, f.value(k * one_e)) for k in range(k_min, k_max + 1))
        it = filter(lambda x: x[1] >= h, it)
        it = map(lambda x: x[0], it)
        k = max(it, default=-1)

        if f.value(k * one_e) >= (1 - eps) * k * theta:
            return k

        h = (1 - eps) * h

    return None
