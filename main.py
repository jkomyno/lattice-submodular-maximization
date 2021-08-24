from df_utils import BenchmarkDF
import os
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import conf_utils
import utils
import timeout
import algo


@hydra.main(config_path='conf', config_name='config')
def benchmark(cfg: DictConfig) -> None:
    # numpy random generator instance
    rng: np.random.Generator = np.random.default_rng(2021)

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent)

    # folder where to save benchmarks
    output_benchmarks = os.path.join(basedir, cfg.output.benchmarks)
    print(f'output_benchmark: {output_benchmarks}')

    #######################
    #  Run the maximizer  #
    #######################

    # - We want to have different cardinality constraints r to test
    # - We should run each experiment n_samples times
    # - We should store a list of the time results in a dataframe

    out_csv_filename = os.path.join(output_benchmarks, f'{cfg.obj.objective}.csv')
    print(f'Creating {out_csv_filename}...')

    with open(out_csv_filename, 'w+') as out_csv:

        for f, r in conf_utils.get_objective(rng, cfg=cfg):
            print(f'Computing exact maximum for n={f.n}, b={f.b}, r={r}...')
            x_star, exact_max = max(((x, f.value(x)) for x in utils.powerset(f) if np.sum(x) <= r), key=utils.snd)
            print(f'max(x) s.t. |x| <= {r}')
            print(f'x*: {x_star}; f(x*): {exact_max}') 

            # initialize BenchmarkDF for current batch
            benchmark_df = BenchmarkDF(n=f.n, b=f.b, r=r, opt=exact_max, out_csv=out_csv)

            for timeout_s in cfg.runtime.timeouts:
                print(f'{timeout_s}s timeout')

                for n_sample in range(1, cfg.runtime.n_samples + 1):
                    print(f'samples: {n_sample}/{cfg.runtime.n_samples}')
                    x = timeout.break_after(seconds=timeout_s)(
                        algo.stochastic_greedy_norm
                    )(rng, f, r)
                    approx_max = f.value(x)
                    approx_ratio = approx_max / exact_max
                    print(f'x: {x}')
                    print(f'f(x): {approx_max}')
                    print(f'f(x) / f(x*): {approx_ratio}')
                    print('')

                    benchmark_df.add(i=n_sample, timeout_s=timeout_s, approx=approx_max)
                
                print('')

            # append batch to the out_csv file
            benchmark_df.write()


if __name__ == '__main__':
    benchmark()
