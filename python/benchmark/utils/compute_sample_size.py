import numpy as np


def compute_sample_size(n: int, r: int, eps: float) -> int:
    """
    Compute the sample size for the stochastic greedy algorithms.
    :param n: size of the ground set
    :param r: cardinality constraint
    :param eps: error threshold
    """
    s = -np.log(eps) * n / r
    s = max(int(s), 1)
    return s
