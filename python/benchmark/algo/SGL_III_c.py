import numpy as np
from nptyping import NDArray
from typing import Tuple
from ..objective import Objective
from .. import utils


def SGL_III_c(rng: np.random.Generator, f: Objective,
              r: int, eps: float) -> Tuple[NDArray[int], float]:
    """
    Randomized algorithm for integer-lattice submodular maximization of monotone functions with cardinality
    constraints in linear time.
    This is a generalization of the StochasticGreedy algorithm for set-submodular monotone functions.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param r: cardinality constraint
    :param eps: non-negative error threshold
    """
    # compute s, the sample size
    s = utils.compute_sample_size(n=f.n, r=r, eps=eps)

    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # prev_value keeps track of the value of f(x)
    prev_value = 0

    # iteration counter
    t = 0

    # norm keeps track of the L-1 norm of x
    norm = 0

    d = max((f.value(utils.char_vector(f, e)) for e in f.V))
    theta = d
    stop_theta = (eps / r) * d

    while norm < r:
        V = np.copy(f.V)
        rng.shuffle(V)

        # split list V in batches of size at most s
        batches = utils.split_list(V, s)

        for Q in batches:
            Q_one = list(map(lambda e: utils.char_vector(f, e), Q))

            # keep track of the (k, candidate_x, candidate_value) tuples in Q
            best_t_list = []

            # potentially add multiple copies of every item in Q
            for i, e in enumerate(Q):
                one_e = Q_one[i]
                k_max = np.min([f.B[e] - x[e], r - norm])
                k_range = list(range(1, k_max + 1))

                # find k in k_interval maximal such that f(k * 1_e | x) >= k * theta
                best_t = utils.binary_search(f, k_range, one_e=one_e, x=x,
                                            prev_value=prev_value, theta=theta)

                if best_t is not None:
                    best_t_list.append(best_t)

            if len(best_t_list) > 0:
                # select the best_t with the largest marginal gain
                k, candidate_x, candidate_value = max(best_t_list, key=lambda best_t: best_t[2] - prev_value)

                # We add to x the element in the sample q that increases the value of f
                # the most, extracted k times.
                x = candidate_x
                norm += k
                prev_value = candidate_value

            # update theta
            theta = max(theta * (1 - eps * (s / f.n)), stop_theta)

            # increment iteration counter
            t += 1

            if norm == r:
                break

    print(f'SGL-III-c  t={t}; n={f.n}; B={f.B_range}; r={r}; norm={norm}')
    assert norm <= r
    return x, prev_value
