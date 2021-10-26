from . import generate_nbr


if __name__ == '__main__':
    # padding = ' '
    # pad_length = 6
    # 
    # for n, r, b in nbr_set:
    #     print(f'''    -
    #   - {n:{padding}<{pad_length}}# n
    #   - {b:{padding}<{pad_length}}# b
    #   - {r:{padding}<{pad_length}}# r''')

    for n, r, B in sorted(set(generate_nbr())):
        print(f'''    -
      - {n}
      -
        - {B[0]}
        - {B[1]}
      - {r}\n''')

