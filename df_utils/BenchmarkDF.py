import numpy as np
import pandas as pd
from typing import Tuple
from io import TextIOWrapper


class BenchmarkDF(object):
    def __init__(self, n: int, B_range: Tuple[int, int], r: int, out_csv: TextIOWrapper,
                 verbose: bool = False):
        """
        :param n: size of the ground set
        :param b: upper bound of the integer lattice domain of f
        :param r: cardinality constraint size
        :param opt: optimum of the maximization with cardinality constraint problem
        :param out_csv: output csv file
        """

        self.n = n
        self.b_low, self.b_high = B_range
        self.r = r
        self.out_csv = out_csv
        self.verbose = verbose

        self.dtypes = [
            ('i', np.int8),
            ('n', np.int32),
            ('b_low', np.int32),
            ('b_high', np.int32),
            ('r', np.int32),
            ('approx', np.float64),
            ('n_calls', np.int64),
            ('time_ms', np.int64),
        ]
        
        self.buf = []

        self.df: pd.DataFrame = None
        self.__reset_df()

    def __reset_df(self):
        self.df = pd.DataFrame(np.empty(0, dtype=self.dtypes))

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

        # copy the buffer into an empty dataframe
        self.df = self.df.append(self.buf, ignore_index=True)

        # write to disk, appending to the file if it exists
        self.df.to_csv(self.out_csv, mode=file_open_mode, index=False, header=should_add_header,
                       sep=',', encoding='utf-8', decimal='.')
        
        # reset the dataframe and empty the buffer
        self.__reset_df()
        self.buf = []

    def add(self, i: int, approx: float, n_calls: int,
            time_ns: float):
        """
        Add a row to the self.df dataframe
        """
        time_ms = time_ns // 1_000_000

        if self.verbose:
            print(f'\t ({i}): {approx} found in {time_ms}ms ({n_calls} oracle calls)')
        
        # update buffer
        self.buf.append(
            {
                'i': i,
                'n': self.n,
                'b_low': self.b_low,
                'b_high': self.b_high,
                'r': self.r,
                'approx': approx,
                'n_calls': n_calls,
                'time_ms': time_ms,
            }
        )
