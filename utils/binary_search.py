from typing import List, Union, Tuple
from nptyping import NDArray, Int64
from objective import Objective


def binary_search(f: Objective, k_range: List[int], one_e: NDArray[Int64],
                  x: NDArray[Int64], prev_value: float,
                  theta: float) -> Union[None, Tuple[int, NDArray[Int64], float]]:
    """
    Iterative binary search for the maximum k in k_range such that
    f.marginal_gain(k * one_e, x) >= k * theta.
    :param f: monotone integer lattice submodular function
    :param k_range: sorted range of k to search
    :param one_e: n-dimensional characteristic vector of e
    :param x: previous iterate
    :param prev_value: value of f(x)
    :param theta: threshold
    :return: (k, x + k * one_e, f(x + k * one_e)) or None of no k such that
             f.marginal_gain(k * one_e, x) >= k * theta could be found.
    """
    if len(k_range) == 0:
        return None

    k_max = k_range[-1]
    k_min = k_range[0]
    candidate_k = k_max
    best_t = None

    while k_max - k_min > 1:
        candidate_x = x + candidate_k * one_e
        candidate_value = f.value(candidate_x)
        if (candidate_value - prev_value) >= candidate_k * theta:
            k_min = candidate_k

            if best_t is None or best_t[0] < candidate_k:
                best_t = (candidate_k, candidate_x, candidate_value)
        else:
            k_max = candidate_k

        candidate_k = (k_max + k_min) // 2
    
    return best_t
