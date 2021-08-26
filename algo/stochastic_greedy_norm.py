import math
import numpy as np
from nptyping import NDArray, Int64
from typing import Iterator
from objective import Objective
import utils


def stochastic_greedy_norm(rng: np.random.Generator,
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
    s = int(((f.n * f.b) / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    for _ in range(r):
        # sub-sampling step, sample a random vector with Manhattan norm equal to s.
        # s >= r as long as 0 < eps <= e^\frac{-r^2}{n}, so in these cases q != x.
        # How should we deal with the other cases?
        q = utils.random_vector_with_norm(rng, f=f, norm=s)

        # e \gets \argmax_e \{ f(\mathbf{1}_e\ |\ \mathbf{x}) | e \in supp(q) \wedge x_e < b \}
        e, _ = max((
          (e, f.value(x + utils.char_vector(f, e)) - f.value(x)) for e, qe in enumerate(q)
          if qe > 0 and x[e] < f.b
        ), key=utils.snd)
        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + one_e

    return x


def stochastic_greedy_norm_it(rng: np.random.Generator,
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
    s = int(((f.n * f.b) / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)
    yield x

    for i in range(r):
        # sub-sampling step, sample a random vector with Manhattan norm equal to s.
        # s >= r as long as 0 < eps <= e^\frac{-r^2}{n}, so in these cases q != x.
        # How should we deal with the other cases?
        q = utils.random_vector_with_norm(rng, f=f, norm=s)

        # e \gets \argmax_e \{ f(\mathbf{1}_e\ |\ \mathbf{x}) | e \in supp(q) \wedge x_e < b \}
        e, _ = max((
          (e, f.value(x + utils.char_vector(f, e)) - f.value(x)) for e, qe in enumerate(q)
          if qe > 0 and x[e] < f.b
        ), key=utils.snd)
        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        y = x + one_e

        # update the approximated solution x
        x = y
        yield y

    yield x
