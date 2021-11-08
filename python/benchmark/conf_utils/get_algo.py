import numpy as np
from omegaconf import DictConfig
from ..objective import Objective
from ..algo import SGL_I, SGL_II, SGL_III, \
                  SGL_II_b, SGL_III_b, SGL_III_c, SSG, \
                  soma_DR_I, soma_II, \
                  lai_DR


ALGO_MAP = {
    'SGL-I': lambda *args: load_SGL_I(*args),
    'SGL-II': lambda *args: load_SGL_II(*args),
    'SGL-III': lambda *args: load_SGL_III(*args),
    'SGL-II-b': lambda *args: load_SGL_II_b(*args),
    'SGL-III-b': lambda *args: load_SGL_III_b(*args),
    'SGL-III-c': lambda *args: load_SGL_III_c(*args),
    'SSG': lambda *args: load_SSG(*args),
    'Soma-DR-I': lambda *args: load_soma_DR_I(*args),
    'Soma-II': lambda *args: load_soma_II(*args),
    'Lai-DR': lambda *args: load_laid_DR(*args),
}


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


def load_SGL_III(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_III(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_II_b(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_II_b(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_III_b(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_III_b(rng, f, r, eps=get_eps(f))
        return x, value

    return load


def load_SGL_III_c(rng: np.random.Generator, f: Objective, r: int):
    def load():
        x, value = SGL_III_c(rng, f, r, eps=get_eps(f))
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
