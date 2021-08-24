import numpy as np
import pandas as pd
from io import TextIOWrapper


class BenchmarkDF(object):
    def __init__(self, n: int, b: int, r: int, opt: int, out_csv: TextIOWrapper):
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
        self.opt = opt
        self.out_csv = out_csv

        dtypes = np.dtype(
            [
                ('i', int),
                ('n', int),
                ('b', int),
                ('r', int),
                ('opt', int),
                ('approx', int),
                ('timeout_s', int),
            ]
        )
        self.df = pd.DataFrame(np.empty(0, dtype=dtypes))

    def write(self):
        """
        Write to the output csv file in append-mode
        """
        should_add_header = self.out_csv.tell() == 0
        file_open_mode = 'a' if should_add_header else 'w'
        self.df.to_csv(self.out_csv, mode=file_open_mode, index=False, header=should_add_header,
                       sep=',', encoding='utf-8', decimal='.')

    def add(self, i: int, timeout_s: int, approx: int):
        """
        Add a row to the self.df dataframe
        """
        self.df = self.df.append(
            {
                'i': i,
                'n': self.n,
                'b': self.b,
                'r': self.r,
                'opt': self.opt,
                'approx': approx,
                'timeout_s': timeout_s,
            },
            ignore_index=True,
        )
