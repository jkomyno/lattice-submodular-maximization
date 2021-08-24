from typing import Iterator, Tuple, List
import numpy as np
from omegaconf import DictConfig
from objective import Objective, DemoMonotone, DemoNonMonotone


OBJ_MAP = {
    'demo_monotone': lambda *args: load_demo_monotone(*args),
    'demo_non_monotone': lambda *args: load_demo_non_monotone(*args),
}


def load_demo_monotone(rng: np.random.Generator, params) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random set-modular, monotone function
    :param rng: numpy random generator instance
    :param params: 'params.demo_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = params.benchmark.nbr

    for (n, b, r) in nbr:
        yield (DemoMonotone(rng, n=n, b=b), r)


def load_demo_non_monotone(rng: np.random.Generator, params) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random set-modular, non_monotone function
    :param rng: numpy random generator instance
    :param params: 'params.demo_non_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = params.benchmark.nbr
    
    for (n, b, r) in nbr:
        yield (DemoNonMonotone(rng, n=n, b=b), r)


def get_objective(rng: np.random.Generator, cfg: DictConfig) -> Iterator[Tuple[Objective, int]]:
    """
    Return an instance of the selected set-submodular objective
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """
    objective_name = cfg.obj.objective
    print(f'Loading f: {objective_name}\n')
    return OBJ_MAP[objective_name](rng, cfg.obj)
