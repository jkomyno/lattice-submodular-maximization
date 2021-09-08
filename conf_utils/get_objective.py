from typing import Iterator, Tuple, List, Callable
import math
import numpy as np
from omegaconf import DictConfig
from objective import Objective, DemoMonotone, DemoMonotoneSkewed, \
                      DemoNonMonotone, FacilityLocation, BudgetAllocation
import dataset_utils
import utils


OBJ_MAP = {
    'demo_monotone': lambda **kwargs: load_demo_monotone(**kwargs),
    'demo_monotone_skewed': lambda **kwargs: load_demo_monotone_skewed(**kwargs),
    'demo_non_monotone': lambda **kwargs: load_demo_non_monotone(**kwargs),
    'facility_location': lambda **kwargs: load_facility_location(**kwargs),
    'budget_allocation': lambda **kwargs: load_budget_allocation(**kwargs),
}


def load_demo_monotone(rng: np.random.Generator,
                       multiply: Callable[[int], int],
                       params,
                       **kwargs) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random modular, monotone function
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = list(map(lambda t: map(multiply, t), params.benchmark.nbr))

    for (n, b, r) in nbr:
        yield (DemoMonotone(rng, n=n, b=b), r)


def load_demo_monotone_skewed(rng: np.random.Generator,
                              multiply: Callable[[int], int],
                              params,
                              **kwargs) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random skewed modular, monotone function
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = list(map(lambda t: map(multiply, t), params.benchmark.nbr))

    for (n, b, r) in nbr:
        yield (DemoMonotoneSkewed(rng, n=n, b=b), r)


def load_demo_non_monotone(rng: np.random.Generator,
                           multiply: Callable[[int], int],
                           params,
                           **kwargs) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random modular, non_monotone function
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_non_monotone' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = list(map(lambda t: map(multiply, t), params.benchmark.nbr))

    for (n, b, r) in nbr:
        yield (DemoNonMonotone(rng, n=n, b=b), r)


def load_facility_location(rng: np.random.Generator,
                           basedir: str,
                           params,
                           **kwargs) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random integer-lattice submodular, monotone function that models
    the Facility Location problem.
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_facility_location' dictionary entry in conf/config.yaml
    """
    print(f'Loading Brunson Revolution...')
    G = dataset_utils.import_brunson_revolution(rng, basedir)
    print(f'...Brunson Revolution successfully loaded')
    br: List[Tuple[int, int]] = params.benchmark.br
    
    for (b, r) in br:
        yield (FacilityLocation(G=G, b=b), r)


def load_budget_allocation(rng: np.random.Generator,
                           basedir: str,
                           params,
                           **kwargs) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random integer-lattice DR-submodular, monotone function that models
    the Budget Allocation problem.
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_facility_location' dictionary entry in conf/config.yaml
    """
    print(f'Loading Brunson Revolution...')
    G = dataset_utils.import_brunson_revolution(rng, basedir)
    print(f'...Brunson Revolution successfully loaded')
    utils.send_telegram(f'...Brunson Revolution successfully loaded')
    br: List[Tuple[int, int]] = params.benchmark.br
    
    for b, r in br:
        yield (BudgetAllocation(G=G, b=b), r)


def get_objective(rng: np.random.Generator,
                  basedir: str,
                  cfg: DictConfig) -> Iterator[Tuple[Objective, int]]:
    """
    Return an instance of the selected set-submodular objective
    :param rng: numpy random generator instance
    :param basedir: base directory relative to main.py
    :param cfg: Hydra configuration dictionary
    """
    objective_name = cfg.obj.objective
    multiplier = cfg.runtime.multiplier

    def multiply(a: int):
        return int(multiplier * a)

    print(f'Loading f: {objective_name}\n')
    utils.send_telegram(f'Loading f: {objective_name}')
    return OBJ_MAP[objective_name](rng=rng,
                                   multiply=multiply,
                                   params=cfg.obj,
                                   basedir=basedir)
