import math
import numpy as np
from nptyping import NDArray, Int64
from typing import Set, Tuple
from set_objective import SetObjective
import utils


def stochastic_greedy(rng: np.random.Generator, f: SetObjective,
                      r: int, eps: float = None) -> Tuple[Set[int], float]:
    """
    Computes a set A \subseteq V such that |A| \leq r.
    :param rng: numpy random generator instance
    :param f: set-submodular function to maximize
    :param r: cardinality constraint
    :param eps: error threshold
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = utils.compute_sample_size(n=f.n, r=r, eps=eps)

    # the solution starts from the empty set
    A: Set[int] = set()

    # prev_value keeps track of the value of f(A)
    prev_value = 0

    while len(A) < r:
        # R is a random subset obtained by sampling s random elements
        # from V - A
        sample_space = list(f.V - A)
        R: NDArray[Int64] = rng.choice(f.V - A, size=min(s, len(sample_space)), replace=False)
        prev_value, marginal_gain, a = max((
            (candidate_value := f.value(A | {a}), candidate_value - prev_value, a)
            for a in R
        ), key=utils.snd)
        A.add(a)

    return A, prev_value
