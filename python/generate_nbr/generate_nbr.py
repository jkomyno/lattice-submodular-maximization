import math
import numpy as np
from typing import Iterator, Tuple


def generate_nbr(n_start: int = 100, b_times: int = 6) -> Iterator[Tuple[int, int, Tuple[int, int]]]:
    n_factors = [
        1,
        2,
        5,
        7.5,
    ]

    r_factors = [
        0.25,
        0.5,
        1,
        2
    ]

    for n_factor in n_factors:
        n = math.floor(n_start * n_factor)
            
        for r_factor in r_factors:
            r = n * r_factor
            r = math.floor(r)
            
            if r < n:
                yield n, r, (1, 1)

            b_factors = np.linspace(r // 20, r // 2, b_times)

            for b in b_factors:
                B = (math.floor(b), math.floor(b * 4))
                if min(B) >= 1 and max(B) <= r and r < n * max(B):
                    yield n, r, B
