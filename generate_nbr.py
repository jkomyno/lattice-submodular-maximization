def generate_nbr(n_base: int, b_base: int):
    n_factors = [
        1,
        2,
        5,
        10,
        50
    ]

    b_factors = [
        1,
        2,
        4,
        8,
        12,
        25,
        50
    ]

    r_factors = [
        1,
        1.5,
        2,
        5,
        10
    ]

    for n_factor in n_factors:
        for b_factor in b_factors:
            for r_factor in r_factors:
                n = int(n_base * n_factor)
                b = int(b_base * b_factor)
                r = int(b * r_factor)

                if b <= r and r < (n * b):
                    yield n, b, r


if __name__ == '__main__':
    n_base = 50
    b_base = 5
    
    nbr_list = [*generate_nbr(n_base, b_base)]

    padding = ' '
    pad_length = 6

    for n, b, r in nbr_list:
        print(f'''    -
      - {n:{padding}<{pad_length}}# n
      - {b:{padding}<{pad_length}}# b
      - {r:{padding}<{pad_length}}# r''')

