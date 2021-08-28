from typing import Iterator, Tuple, List
import numpy as np
from omegaconf import DictConfig
from objective import Objective
from algo import SGN_I, SGN_II, SGN_III, SSG, soma_DR_I
from utils import bridge


ALGO_MAP = {
    'SGN-I': lambda *args: load_SGN_I(*args),
    'SGN-II': lambda *args: load_SGN_II(*args),
    'SGN-III': lambda *args: load_SGN_III(*args),
    'SSG': lambda *args: load_SSG(*args),
    'Soma-DR-I': lambda *args: load_soma_DR_I(*args),
}


def load_SGN_I(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x = SGN_I(rng, f, r, eps=get_eps(f))
        return f.value(x)

    return load


def load_SGN_II(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x = SGN_II(rng, f, r, eps=get_eps(f))
        return f.value(x)

    return load


def load_SGN_III(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x = SGN_III(rng, f, r, eps=get_eps(f))
        return f.value(x)

    return load


def load_SSG(rng: np.random.Generator, f: Objective, r: int):
    def load():
        _, value = SSG(rng, f, r, eps=get_eps(f))
        return value

    return load


def load_soma_DR_I(_: np.random.Generator, f: Objective, r: int):
    c = np.full((f.n, ), fill_value=f.b)
    def load():
        x = soma_DR_I(f, c, r, eps=get_eps(f))
        return f.value(x)

    return load


def get_eps(f: Objective):
    return 1 / (f.n * 4)


def get_algo(rng: np.random.Generator, f: Objective,
             r: int, cfg: DictConfig):
    """
    Return an instance of the selected set-submodular objective
    :param rng: numpy random generator instance
    :param f: integer lattice submodular function
    :param r: cardinality constraint
    :param cfg: Hydra configuration dictionary
    """
    algo_name = cfg.algo.algorithm
    print(f'Importing algorithm: {algo_name}\n')
    return ALGO_MAP[algo_name](rng, f, r)
