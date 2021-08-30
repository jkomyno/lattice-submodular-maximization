import numpy as np

SEED = 2022


def get_seed() -> int:
    return SEED


def get_rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
