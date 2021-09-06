import numpy as np
from nptyping import NDArray, Int64
from objective import Objective
from typing import Tuple
import utils


def SGL_III_b(rng: np.random.Generator,
              f: Objective, r: int, eps: float = None) -> Tuple[NDArray[Int64], float]:
    """
    Randomized algorithm for integer-lattice submodular maximization of monotone functions with cardinality
    constraints in linear time.
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

    # prev_value keeps track of the value of f(x)
    prev_value = 0

    # iteration counter
    t = 0

    d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    # norm keeps track of the L-1 norm of x
    norm = 0

    def constraint_k_theta(prev_value: float, theta: float):
        """
        Higher-order function to be applied to filter.
        :param prev_value: value of f(x) at the previous iteration
        :param theta: decreasing threshold
        """
        def helper(t: Tuple[int, NDArray[Int64], float]) -> bool:
            """
            Returns true iff f(k * 1_e | x) >= k * theta
            :param t: (k, x + k * 1_e, f(x + k * 1_e)) tuple
            """
            k, _, candidate_value = t
            return candidate_value - prev_value >= k * theta

        return helper

    while norm < r and t < r:
        # random sub-sampling step
        sample_space = np.where(x < f.b)[0]
        Q = rng.choice(sample_space, size=min(s, len(sample_space)), replace=False)

        # potentially add multiple copies of every item in Q
        for e in Q:
            one_e = utils.char_vector(f, e)
            
            # find k in k_interval maximal such that f(k * 1_e | x) >= k * theta
            lazy_list = (
                (k, x + k * one_e, f.value(x + k * one_e))
                for k in range(min(f.b - x[e], r - norm) + 1)
            )
            lazy_list = filter(constraint_k_theta(prev_value, theta), lazy_list)
            k, candidate_x, candidate_value = max(lazy_list, key=utils.fst, default=(None, None, None))

            if k == None:
                # no feasible k was found, nothing gets added to x this iteration.
                continue

            if candidate_value >= prev_value:
                # We add to x the element in the sample q that increases the value of f
                # the most, extracted k times.
                x = candidate_x
                norm += k
                prev_value = candidate_value

        # update theta
        theta = max(theta * (1 - eps), stop_theta)

        # increment iteration counter
        t += 1

    assert norm <= r
    return x, prev_value
