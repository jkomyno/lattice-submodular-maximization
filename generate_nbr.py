import math
import numpy as np
import utils


# numpy random generator instance
rng: np.random.Generator = utils.get_rng()


def generate_nbr(n_start: int, b_times: int):
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
        1.5
    ]

    for n_factor in n_factors:
        n = math.floor(n_start * n_factor)
            
        for r_factor in r_factors:
            r = n * r_factor
            r = math.floor(r)
            
            if r < n:
                yield n, r, (1, 1)

            b_factors = np.linspace(r // 20, r, b_times)

            for b in b_factors:
                B = (math.floor(b), math.floor(b * 4))
                if min(B) >= 1 and max(B) <= r and r < n * max(B):
                    yield n, r, B


if __name__ == '__main__':
    n_start = 50
    b_times = 8
    nbr_set = sorted(set(generate_nbr(n_start, b_times)))

    # padding = ' '
    # pad_length = 6
    # 
    # for n, r, b in nbr_set:
    #     print(f'''    -
    #   - {n:{padding}<{pad_length}}# n
    #   - {b:{padding}<{pad_length}}# b
    #   - {r:{padding}<{pad_length}}# r''')

    for n, r, B in nbr_set:
        print(f'''    -
      - {n}
      -
        - {B[0]}
        - {B[1]}
      - {r}\n''')
