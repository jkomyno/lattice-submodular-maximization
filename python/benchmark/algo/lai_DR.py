import numpy as np
import cvxpy as cvx
from nptyping import NDArray
from typing import Tuple
from ..objective import Objective
from .. import utils


def lai_DR(rng: np.random.Generator, f: Objective, r: int) -> Tuple[NDArray[int], float]:
    """
    Implement Lai'19 algorithm for maximizing a DR-submodular monotone function
    over the integer lattice under cardinality constraint.
    :param rng: numpy random generator instance
    :param f: a DR-submodular monotone function
    :param r: the cardinality constraint
    """
    # the solution starts from the zero vector
    x = np.zeros((f.n, ), dtype=int)

    # we assume f is normalized
    prev_value = 0

    # characteristic_vectors keeps track of the characteristic vector representation
    # of all e in f.V, assuming f.V is 0-indexed
    characteristic_vectors = [
      utils.char_vector(f, e)
      for e in f.V
    ]

    # initialize the procedure to find m
    find_m = argmax_m(f, r, characteristic_vectors)

    for t in range(r):
        # find the optimal m w.r.t. the current x
        m = find_m(x, prev_value)
        
        # m_norm is the L-1 norm of m
        norm_m = np.sum(m)

        # choose e in f.V randomly with probability m[e] / norm_m for all e in f.V
        e = rng.choice(f.V, p=[m[e] / norm_m for e in f.V])
        one_e = characteristic_vectors[e]

        # update the solution adding a single element
        x = x + one_e
        prev_value = f.value(x)

    print(f'Lai-DR     t={t}; n={f.n}; B={f.B_range}; r={r}; norm={np.sum(x)}')
    assert np.sum(x) == r
    return x, prev_value



def argmax_m(f: Objective, r: int,
             characteristic_vectors: NDArray[int]) -> NDArray[int]:
    # variable to be found with optimization
    m = cvx.Variable(shape=(f.n, ), integer=True)

    # define static constraints
    cardinality_constraint = np.sum(m) <= r
    lower_bounds = [
      m[e] >= 0
      for e in f.V
    ]
    upper_bounds = [
      m[e] <= f.B[e]
      for e in f.V
    ]
    
    def helper(x: NDArray[int], prev_value: float):     
      # compute f(e | x) for all e in f.V
      f_marginal_gains = [
        f.value(one_e + x) - prev_value
        for one_e in characteristic_vectors
      ]

      # objective function
      objective = cvx.Maximize(m.T @ f_marginal_gains)

      # dynamic constraints
      constraints = [
        cardinality_constraint,
        *lower_bounds,
        *upper_bounds,
        *(
          m[e] + x[e] <= f.B[e]
          for e in f.V
        )
      ]

      # use cvxpy to solve the objective
      problem = cvx.Problem(objective, constraints)

      # solve the optimization problem
      problem.solve(verbose=False)

      # retrieve the value of y
      m_values = m.value
      return m_values

    return helper
