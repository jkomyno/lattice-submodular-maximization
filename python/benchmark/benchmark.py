import os
import hydra
import time
from pathlib import Path
from omegaconf import DictConfig
from pathlib import Path
from . import conf_utils
from .df_utils import BenchmarkDF
from ..rng import rng


@hydra.main(config_path='../conf', config_name='config')
def benchmark(cfg: DictConfig) -> None:
    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)

    # submodular function name
    f_name = cfg.obj.name

    # dataset basedir
    dataset_dir = f'{basedir}/python/benchmark/'

    # folder where to save benchmarks
    output_benchmarks = f'{basedir}/out/{f_name}'
    print(f'output_benchmark: {output_benchmarks}')

    # ensure output folder exists
    Path(output_benchmarks).mkdir(parents=True, exist_ok=True)

    # run deterministic algorithms only once
    n_samples = cfg.runtime.n_samples if cfg.algo.is_randomized else 1

    ########################
    #  Run the maximizers  #
    ########################

    out_csv_filename = os.path.join(output_benchmarks, f'{cfg.algo.algorithm}.csv')
    print(f'Creating {out_csv_filename}...')

    with open(out_csv_filename, 'w+') as out_csv:
        for f, r in conf_utils.get_objective(rng=rng, dataset_dir=dataset_dir, cfg=cfg):

            # import the selected algorithm to maximize f w.r.t. the cardinality constraint r
            maximizer = conf_utils.get_algo(rng, f, r, cfg=cfg)

            # initialize BenchmarkDF for current batch
            with BenchmarkDF(f=f, r=r, out_csv=out_csv, verbose=True) as benchmark_df:
                for n_sample in range(1, n_samples + 1):

                    t_start = time.time_ns()
                    x, approx = maximizer()
                    time_ns = time.time_ns() - t_start

                    # n_calls is the number of oracle calls
                    n_calls = f.n_calls

                    benchmark_df.add(i=n_sample, approx=approx, n_calls=n_calls,
                                     time_ns=time_ns)

                    # reset the counter of oracle calls for f
                    f.reset()

    print(f'OK')
