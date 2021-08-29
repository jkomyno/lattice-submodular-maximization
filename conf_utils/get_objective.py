from typing import Iterator, Tuple, List, Callable
import numpy as np
from omegaconf import DictConfig
from objective import Objective, DemoMonotone, DemoNonMonotone


OBJ_MAP = {
    'demo_monotone': lambda *args: load_demo_monotone(*args),
    'demo_non_monotone': lambda *args: load_demo_non_monotone(*args),
}


def load_demo_monotone(rng: np.random.Generator,
                       multiply: Callable[[int], int],
                       params) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random set-modular, monotone function
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = list(map(lambda t: map(multiply, t), params.benchmark.nbr))

    for (n, b, r) in nbr:
        yield (DemoMonotone(rng, n=n, b=b), r)


def load_demo_non_monotone(rng: np.random.Generator,
                           multiply: Callable[[int], int],
                           params) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random set-modular, non_monotone function
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_non_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = params.benchmark.nbr
    
    for (n, b, r) in nbr:
        n, b, r = map(multiply, (n, b, r))
        yield (DemoNonMonotone(rng, n=n, b=b), r)


def get_objective(rng: np.random.Generator, cfg: DictConfig) -> Iterator[Tuple[Objective, int]]:
    """
    Return an instance of the selected set-submodular objective
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """
    objective_name = cfg.obj.objective
    multiplier = cfg.runtime.multiplier

    def multiply(a: int):
        return int(multiplier * a)

    print(f'Loading f: {objective_name}\n')
    return OBJ_MAP[objective_name](rng, multiply, cfg.obj)
