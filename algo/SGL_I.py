import math
import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Iterator
import utils


def SGL_I(rng: np.random.Generator,
          f: Objective, r: int, eps: float = None) -> NDArray[Int64]:
    """
    Randomized algorithm for DR-ubmodular maximization of monotone functions
    defined on the integer lattice with cardinality constraints.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: non-negative error threshold
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = utils.compute_sample_size(n=f.n, r=r, eps=eps)

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    for _ in range(r):
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = rng.choice(sample_space, size=min(s, len(sample_space)), replace=False)
        
        # e \gets \argmax_{e \in Q} f(\mathbf{1}_e\ |\ \mathbf{x})
        one_e = max(map(lambda e: utils.char_vector(f, e), Q),
                    key=lambda one_e: f.marginal_gain(one_e, x))

        # We add to x the element e in the sample Q that increases the value of f
        # the most.
        x = x + one_e

    assert np.sum(x) == r
    return x


def SGL_I_it(rng: np.random.Generator,
             f: Objective, r: int, eps: float = None) -> Iterator[NDArray[Int64]]:
    """
    Randomized algorithm for DR-ubmodular maximization of monotone functions
    defined on the integer lattice with cardinality constraints.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: non-negative error threshold
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = math.ceil((f.n / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    for _ in range(r):
        yield x

        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = set(rng.choice(sample_space, size=s, replace=True))
        
        # e \gets \argmax_{e \in Q} f(\mathbf{1}_e\ |\ \mathbf{x})
        one_e = max(map(lambda e: utils.char_vector(f, e), Q),
                    key=lambda one_e: f.marginal_gain(one_e, x))

        # We add to x the element e in the sample Q that increases the value of f
        # the most.
        x = x + one_e

    assert np.sum(x) == r
    yield x
