from typing import Iterator, Tuple, List, Callable
import numpy as np
import networkx as nx
from omegaconf import DictConfig
from objective import Objective, DemoMonotone, DemoNonMonotone, FacilityLocation, BudgetAllocation
import utils


OBJ_MAP = {
    'demo_monotone': lambda *args: load_demo_monotone(*args),
    'demo_non_monotone': lambda *args: load_demo_non_monotone(*args),
    'facility_location': lambda *args: load_facility_location(*args),
    'budget_allocation': lambda *args: load_budget_allocation(*args),
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


def load_facility_location(rng: np.random.Generator,
                           multiply: Callable[[int], int],
                           params) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random integer-lattice submodular, monotone function that models
    the Facility Location problem.
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_facility_location' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = list(map(lambda t: map(multiply, t), params.benchmark.nbr))

    for (n, b, r) in nbr:

        # p is the probability that any two u, v \in V are connected
        p = 1

        # m is the number of target users
        m = n * 3

        # generate a random graph weighted bipartite graph 
        G: nx.Graph = nx.bipartite.random_graph(n, m, p, directed=False, seed=utils.get_seed())

        weights = rng.integers(low=0, high=10, size=(len(G.edges), ))
        for (u, v), weight in zip(G.edges(), weights):
            G.edges[u, v]['weight'] = weight

        yield (FacilityLocation(G=G, b=b), r)


def load_budget_allocation(rng: np.random.Generator,
                           multiply: Callable[[int], int],
                           params) -> Iterator[Tuple[Objective, int]]:
    """
    Generate a random integer-lattice DR-submodular, monotone function that models
    the Budget Allocation problem.
    :param rng: numpy random generator instance
    :param multiply: function to apply to n, b, r
    :param params: 'params.demo_facility_location' dictionary entry in conf/config.yaml
    """
    nbr: List[Tuple[int, int, int]] = list(map(lambda t: map(multiply, t), params.benchmark.nbr))

    for (n, b, r) in nbr:

        # p is the probability that any two u, v \in V are connected
        p = 0.8

        # m is the number of target customers
        m = n * 3

        # generate a random graph weighted bipartite graph 
        G: nx.Graph = nx.bipartite.random_graph(n, m, p, directed=False, seed=utils.get_seed())

        weights = rng.uniform(low=0.0, high=0.4, size=(len(G.edges), ))
        for (u, v), weight in zip(G.edges(), weights):
            G.edges[u, v]['weight'] = weight

        yield (BudgetAllocation(G=G, b=b), r)


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
