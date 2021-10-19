import numpy as np
from nptyping import NDArray
from objective import Objective


def random_vector_with_norm(rng: np.random.Generator,
                            f: Objective, norm: int) -> NDArray[int]:
    """
    Generate an n-dimensional vector with the given Manhattan norm.
    :param rng: numpy random generator instance
    :param f: integer-lattice submodular function objective
    :param norm: L1-norm that the return vector should have
    """

    q = np.zeros((f.n, ), dtype=int)
    max_num = f.b
    min_num = 0
    cum_sum = 0

    for i in range(f.n):
        max_num = min(max_num, norm - cum_sum)
        max_after = max_num * (f.n - i - 1)
        min_num = min(max_num, max(min_num, norm - cum_sum - max_after))
        entry = rng.integers(low=min_num, high=max_num + 1, size=1)
        q[i] = entry
        cum_sum += entry

    rng.shuffle(q)
    return q
