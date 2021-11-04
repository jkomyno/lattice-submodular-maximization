import numpy as np
import pandas as pd
from io import TextIOWrapper
from ..objective import Objective


class BenchmarkDF(object):
    def __init__(self, f: Objective, r: int, out_csv: TextIOWrapper,
                 verbose: bool = False):
        """
        :param f: the integer-lattice objective function to benchmark
        :param r: cardinality constraint size
        :param opt: optimum of the maximization with cardinality constraint problem
        :param out_csv: output csv file
        """
        # size of the ground set
        self.n = f.n

        # bounds of the molteplicity of each element in the ground set
        self.b_low, self.b_high = f.B_range

        # cardinality of the multiset
        self.b_sum = np.sum(f.B)

        self.r = r
        self.out_csv = out_csv
        self.verbose = verbose

        self.dtypes = [
            ('i', np.int8),
            ('n', np.int32),
            ('b_low', np.int32),
            ('b_high', np.int32),
            ('b_sum', np.int32),
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
            print(f'\nBenchmarking n={self.n}; B=[{self.b_low}, {self.b_high}]; r={self.r}')

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
                'b_sum': self.b_sum,
                'r': self.r,
                'approx': approx,
                'n_calls': n_calls,
                'time_ms': time_ms,
            }
        )
