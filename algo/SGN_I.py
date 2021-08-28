import math
import numpy as np
from itertools import chain
from nptyping import NDArray, Int64
from typing import Iterator
from objective import Objective
import utils


def SGN_I(rng: np.random.Generator,
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
    s = math.ceil(((f.n * f.b) / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    for _ in range(r):
        # sub-sampling step, sample a random vector with Manhattan norm equal to s.
        q = feasible_vector_with_norm(rng=rng, x=x, n=f.n, b=f.b, norm=s)

        # e \gets \argmax_e \{ f(\mathbf{1}_e\ |\ \mathbf{x}) | e \in supp(q) \}
        e, _ = max((
          (e, f.marginal_gain(utils.char_vector(f, e), x)) for e, qe in enumerate(q)
          if qe > 0
        ), key=utils.snd)
        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        x = x + one_e

    return x


def SGN_I_it(rng: np.random.Generator,
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
    s = math.ceil(((f.n * f.b) / r) * math.log(1 / eps))

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)
    yield x

    for i in range(r):
        # sub-sampling step, sample a random vector with Manhattan norm equal to s.
        q = feasible_vector_with_norm(rng=rng, x=x, n=f.n, b=f.b, norm=s)

        # e \gets \argmax_e \{ f(\mathbf{1}_e\ |\ \mathbf{x}) | e \in supp(q) \}
        e, _ = max((
          (e, f.marginal_gain(utils.char_vector(f, e), x)) for e, qe in enumerate(q)
          if qe > 0
        ), key=utils.snd)
        one_e = utils.char_vector(f, e)

        # We add to x the element in the sample q that increases the value of f
        # the most.
        y = x + one_e
        yield y

        # update the solution
        x = y

    yield x


def feasible_vector_with_norm(rng: np.random.Generator, x: NDArray[Int64],
                              n: int, b: int, norm: int) -> NDArray[Int64]:
    """
    :param rng: numpy random generator instance
    :param x: integer lattice solution at the previous iteration
    :param n: size of the ground set
    :param b: upper bound of the integer lattice
    :param norm: desired L1 norm of the q returned vector
    """
    
    B = np.full((n, ), fill_value=b)
    pool = np.fromiter(chain.from_iterable(
        ((i for _ in range(e)) for i, e in enumerate(B - x))), dtype=int)

    # sampling with replacement
    choice = rng.choice(pool, size=norm, replace=True)
    sample_idx = pool[choice]

    q = np.zeros((n, ), dtype=int)

    for e in pool[sample_idx]:
        q[e] += 1

    return q
