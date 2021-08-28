import numpy as np
import pandas as pd
from io import TextIOWrapper


class BenchmarkDF(object):
    def __init__(self, n: int, b: int, r: int, out_csv: TextIOWrapper,
                 verbose: bool = False):
        """
        :param n: size of the ground set
        :param b: upper bound of the integer lattice domain of f
        :param r: cardinality constraint size
        :param opt: optimum of the maximization with cardinality constraint problem
        :param out_csv: output csv file
        """

        self.n = n
        self.b = b
        self.r = r
        self.out_csv = out_csv
        self.verbose = verbose

        dtypes = np.dtype(
            [
                ('i', int),
                ('n', int),
                ('b', int),
                ('r', int),
                ('approx', int),
                ('n_calls', int),
                ('time_ms', int),
            ]
        )
        self.df = pd.DataFrame(np.empty(0, dtype=dtypes))

    def __enter__(self):
        if self.verbose:
            print(f'\nBenchmarking n={self.n}; b={self.b}; r={self.r}')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write()

    def write(self):
        """
        Write to the output csv file in append-mode
        """
        if self.verbose:
            print(f'...writing to CSV...')

        should_add_header = self.out_csv.tell() == 0
        file_open_mode = 'a' if should_add_header else 'w'
        self.df.to_csv(self.out_csv, mode=file_open_mode, index=False, header=should_add_header,
                       sep=',', encoding='utf-8', decimal='.')

    def add(self, i: int, approx: int, n_calls: int, time_ns: float):
        """
        Add a row to the self.df dataframe
        """
        time_ms = time_ns // 1_000_000

        if self.verbose:
            print(f'\t ({i}): {approx} found in {time_ms}ms ({n_calls} oracle calls)')

        self.df = self.df.append(
            {
                'i': i,
                'n': self.n,
                'b': self.b,
                'r': self.r,
                'approx': approx,
                'n_calls': n_calls,
                'time_ms': time_ms,
            },
            ignore_index=True,
        )
