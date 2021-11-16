import numpy as np
from omegaconf import DictConfig
from ..objective import Objective
from ..algo import SGL_a, SGL_b, SGL_c, SGL_d, \
                  SSG, \
                  soma_DR_I, soma_II, \
                  lai_DR


ALGO_MAP = {
    'SGL-a': lambda *args: load_SGL_a(*args),
    'SGL-b': lambda *args: load_SGL_b(*args),
    'SGL-c': lambda *args: load_SGL_c(*args),
    'SGL-d': lambda *args: load_SGL_d(*args),
    # 'SGL-I': lambda *args: load_SGL_I(*args),
    # 'SGL-II': lambda *args: load_SGL_II(*args),
    # 'SGL-II-b': lambda *args: load_SGL_II_b(*args),
    'SSG': lambda *args: load_SSG(*args),
    'Soma-DR-I': lambda *args: load_soma_DR_I(*args),
    'Soma-II': lambda *args: load_soma_II(*args),
    'Lai-DR': lambda *args: load_laid_DR(*args),
}


def load_SGL_a(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_a(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_b(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_b(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_c(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_c(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_d(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_d(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SSG(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SSG(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_soma_DR_I(_: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = soma_DR_I(f, r, eps=get_eps(f))
        return x, value

    return load


def load_soma_II(_: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = soma_II(f, r, eps=get_eps(f))
        return x, value

    return load


def load_laid_DR(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = lai_DR(rng, f, r)
        return x, value

    return load


"""
def load_SGL_I(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_I(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_II(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_II(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_II_b(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_II_b(rng, f, r, eps=get_eps(f))
        return x, value

    return load
"""


def get_eps(f: Objective):
    return 1 / (f.n * 4)


def get_algo(rng: np.random.Generator, f: Objective,
             r: int, cfg: DictConfig):
    """
    Return an instance of the selected integer-lattice submodular objective
    :param rng: numpy random generator instance
    :param f: integer lattice submodular function
    :param r: cardinality constraint
    :param cfg: Hydra configuration dictionary
    """
    algo_name = cfg.algo.algorithm
    print(f'Importing algorithm: {algo_name}\n')
    return ALGO_MAP[algo_name](rng, f, r)
