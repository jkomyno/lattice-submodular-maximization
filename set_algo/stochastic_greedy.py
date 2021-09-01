import math
import numpy as np
from nptyping import NDArray, Int64
from typing import Set, Iterator
from set_objective import SetObjective
import utils


def stochastic_greedy(rng: np.random.Generator, f: SetObjective,
                      r: int, eps: float = None) -> Set[int]:
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

    A: Set[int] = set()
    
    while len(A) < r:
        # R is a random subset obtained by sampling s random elements
        # from V - A
        R: NDArray[Int64] = rng.choice(list(f.V - A), size=s, replace=False)
        a: int = max(R.tolist(), key=lambda x: f.marginal_gain(A, {x}))
        A.add(a)

    return A


def stochastic_greedy_it(rng: np.random.Generator, f: SetObjective,
                         r: int, eps: float = None) -> Iterator[Set[int]]:
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
    s = int((f.n / r) * math.log(1 / eps))

    A: Set[int] = set()
    
    while len(A) < r:
        yield A

        # R is a random subset obtained by sampling s random elements
        # from V - A
        R: NDArray[Int64] = rng.choice(list(f.V - A), size=s, replace=True)
        a: int = max(R.tolist(), key=lambda x: f.marginal_gain(A, {x}))
        A.add(a)

    yield A
