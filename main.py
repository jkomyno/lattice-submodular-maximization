from df_utils import BenchmarkDF
import os
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import time
import conf_utils
import utils


# numpy random generator instance
rng: np.random.Generator = utils.get_rng()


@hydra.main(config_path='conf', config_name='config')
def benchmark(cfg: DictConfig) -> None:
    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent)

    # folder where to save benchmarks
    output_benchmarks = os.path.join(basedir, cfg.output.benchmarks)
    print(f'output_benchmark: {output_benchmarks}')

    ########################
    #  Run the maximizers  #
    ########################

    # run deterministic algorithms once
    n_samples = cfg.runtime.n_samples if cfg.algo.is_randomized else 1

    out_csv_filename = os.path.join(output_benchmarks, f'{cfg.obj.objective}-{cfg.algo.algorithm}.csv')
    print(f'Creating {out_csv_filename}...')

    with open(out_csv_filename, 'w+') as out_csv:
        for f, r in conf_utils.get_objective(rng=rng, basedir=basedir, cfg=cfg):

            # import the selected algorithm to maximize f w.r.t. the cardinality constraint r
            maximizer = conf_utils.get_algo(rng, f, r, cfg=cfg)

            # initialize BenchmarkDF for current batch
            with BenchmarkDF(n=f.n, B_range=f.B_range, r=r, out_csv=out_csv, verbose=True) as benchmark_df:
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


if __name__ == '__main__':
    benchmark()
