import math
import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Iterator
import utils


def SGN_II(rng: np.random.Generator,
           f: Objective, r: int, eps: float = None) -> NDArray[Int64]:
    """
    Randomized algorithm for integer-lattice submodular maximization of monotone functions with cardinality
    constraints in linear time.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: approximation threshold in (0, 0.5)
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = math.ceil((f.n / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    for _ in range(r):
        # sub-sampling step
        sample_space = np.where(x < f.b)[0]
        q = rng.choice(sample_space, size=s, replace=True)
        
        # e \gets \argmax_e \{ f(\mathbf{1}_e\ |\ \mathbf{x}) | e \in supp(q) \}
        e, _ = max((
          (e, f.marginal_gain(utils.char_vector(f, e), x)) for e in q
        ), key=utils.snd)
        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + one_e

    return x


def SGN_II_it(rng: np.random.Generator,
              f: Objective, r: int, eps: float = None) -> Iterator[NDArray[Int64]]:
    """
    Randomized algorithm for integer-lattice submodular maximization of monotone functions with cardinality
    constraints in linear time.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: approximation threshold in (0, 0.5)
    """
    if eps is None:
        eps = 1 / (4 * f.n)

    # compute s, the sample size
    s = math.ceil((f.n / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    for _ in range(r):
        yield x
        # sub-sampling step
        sample_space = np.where(x < f.b)[0]
        q = rng.choice(sample_space, size=s, replace=True)

        # e \gets \argmax_e \{ f(\mathbf{1}_e\ |\ \mathbf{x}) | e \in supp(q) \}
        e, _ = max((
          (e, f.marginal_gain(utils.char_vector(f, e), x)) for e in q
        ), key=utils.snd)
        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + one_e

    yield x
