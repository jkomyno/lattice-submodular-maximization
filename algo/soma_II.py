import numpy as np
from nptyping import NDArray
from objective import Objective
from typing import Union
from typing import Tuple
import utils


def soma_II(f: Objective, r: int, eps: float) -> Tuple[NDArray[int], float]:
    """
    Implement Soma'18 algorithm for maximizing a submodular monotone function
    over the integer lattice under cardinality constraint.
    :param f: a DR-submodular monotone function
    :param r: the cardinality constraint
    :param eps: the error threshold
    """
    # c is the vector upper bound of the lattice domain
    c = f.B

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # norm keeps track of the L-1 norm of x
    norm = 0

    d = max((f.value(c[e] * utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    while theta >= stop_theta:
        for e in f.V:
            one_e = utils.char_vector(f, e)
            k_max = np.min([c[e] - x[e], r - norm])
            
            k = binary_search_lattice(f=f, one_e=one_e, theta=theta, k_max=k_max, eps=eps)

            if k is not None:
                x = x + k * one_e
                norm += k

        theta = theta * (1 - eps)

    return x, f.value(x)


def binary_search_lattice(f: Objective, one_e: NDArray[int], theta: float,
                          k_max: int, eps: float) -> Union[float, None]:
    # find the minimum k_min with 0 <= k_min <= k_max such that f(k_min * one_e) > 0.
    lazy_list = ((k_min, f.value(k_min * one_e)) for k_min in range(0, k_max + 1))
    lazy_list = filter(lambda x: x[1] > 0, lazy_list)
    k_min, k_min_e_value = min(lazy_list, key=utils.fst, default=(None, None))

    if k_min is None:
      return None

    h = f.value(k_max * one_e)
    stop_h = (1 - eps) * k_min_e_value

    while h >= stop_h:
        lazy_list = ((k, f.value(k * one_e)) for k in range(k_min, k_max + 1))
        lazy_list = filter(lambda x: x[1] >= h, lazy_list)
        k, k_e_value = max(lazy_list, default=-1)

        if k_e_value >= (1 - eps) * k * theta:
            return k

        h = (1 - eps) * h

    return None
