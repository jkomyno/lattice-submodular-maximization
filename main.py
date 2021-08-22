import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import conf_utils
import utils
import timeout
import algo

TIMEOUTS = [
    # 1 second
    1,

    # 10 seconds
    10,

    # 1 minute
    60,
]


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # numpy random generator instance
    rng: np.random.Generator = np.random.default_rng(2021)

    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # basedir w.r.t. main.py
    basedir = f'{Path(__file__).parent}'

    # objective submodular function
    f = conf_utils.get_objective(rng, cfg=cfg)

    #######################
    #  Run the maximizer  #
    #######################

    # - We want to have different cardinality constraints r to test
    # - We should run each experiment n_samples times
    # - We should store a list of the time results in a dataframe

    for t in TIMEOUTS:
        r = 10
        S = timeout.break_after(seconds=t)(
            algo.stochastic_greedy_norm
        )(rng, f, r)
        print(f'{t}s timeout \t r: {r}')
        print(f'f(S): {f.value(S)}')


if __name__ == '__main__':
    run()
