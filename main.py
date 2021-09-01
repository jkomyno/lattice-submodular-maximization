from df_utils import BenchmarkDF, BenchmarkWithTimeoutDF
import os
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import time
import conf_utils
import utils
from utils import bridge
import timeout
import algo
import set_algo


# numpy random generator instance
rng: np.random.Generator = utils.get_rng()


@hydra.main(config_path='conf', config_name='config')
def benchmark_with_timeout(cfg: DictConfig) -> None:
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

            # initialize BenchmarkWithTimeoutDF for current batch
            benchmark_df = BenchmarkWithTimeoutDF(n=f.n, b=f.b, r=r, opt=exact_max, out_csv=out_csv)

            for timeout_s in cfg.runtime.timeouts:
                print(f'{timeout_s}s timeout')

                for n_sample in range(1, cfg.runtime.n_samples + 1):
                    print(f'samples: {n_sample}/{cfg.runtime.n_samples}')
                    x = timeout.break_after(seconds=timeout_s)(
                        algo.stochastic_greedy_norm_it
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


@hydra.main(config_path='conf', config_name='config')
def bridge_benchmark(cfg: DictConfig) -> None:
    for f, r in conf_utils.get_objective(rng, cfg=cfg):
        f_prime = bridge.to_set_objective(f)

        print(f'Computing exact maximum for n={f.n}, b={f.b}, r={r}...')
        x_star, exact_max = max(((x, f.value(x)) for x in utils.powerset(f) if np.sum(x) <= r), key=utils.snd)
        print(f'max(x) s.t. |x| <= {r}')
        print(f'x*: {x_star}; f(x*): {exact_max}')

        for timeout_s in cfg.runtime.timeouts:
            print(f'{timeout_s}s timeout')

            for n_sample in range(1, cfg.runtime.n_samples + 1):
                print(f'samples: {n_sample}/{cfg.runtime.n_samples}')
                x = timeout.break_after(seconds=timeout_s)(
                    algo.stochastic_greedy_norm_it
                )(rng, f, r)
                approx_max = f.value(x)
                approx_ratio = approx_max / exact_max
                print(f'x: {x}')
                print(f'f(x): {approx_max}')

                S = timeout.break_after(seconds=timeout_s)(
                    set_algo.stochastic_greedy_it
                )(rng, f_prime, r)
                print(f'S: {S}')
                x_prime = bridge.to_integer_lattice(f_prime, S)
                set_approx_max = f_prime.value(S)
                set_approx_ratio = set_approx_max / exact_max

                print(f'x\': {x_prime}')
                print(f'f\'(x\'): {set_approx_max}')

                print(f'f(x) / f(x*):  {approx_ratio}')
                print(f'f(x\') / f(x*): {set_approx_ratio}')
                print('')

            print('')


@hydra.main(config_path='conf', config_name='config')
def bridge_comparison(cfg: DictConfig) -> None:
    for f, r in conf_utils.get_objective(rng, cfg=cfg):
        print(f'n={f.n}, b={f.b}, r={r}...')

        f_prime = bridge.to_set_objective(f)

        t_start = time.time()
        x = algo.stochastic_greedy_norm(rng, f, r)
        time_sgn = time.time() - t_start

        t_start = time.time()
        S = set_algo.stochastic_greedy(rng, f_prime, r)
        time_sg = time.time() - t_start

        max_sgn = f.value(x)
        max_sg = f_prime.value(S)

        obj_ratio = max_sgn / max_sg
        time_ratio = time_sgn / time_sg

        print(f'f(x):  {max_sgn}')
        print(f'f\'(S): {max_sg}')
        print(f'time SGN: {time_sgn}')
        print(f'time SG:  {time_sg}')
        print(f'f(x) / f\'(S): {obj_ratio}')
        print(f'time f(x) / time f\'(S): {time_ratio}')
        print('')


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
            with BenchmarkDF(n=f.n, b=f.b, r=r, out_csv=out_csv, verbose=True) as benchmark_df:
                for n_sample in range(1, n_samples + 1):

                    t_start = time.time_ns()
                    approx = maximizer()
                    time_ns = time.time_ns() - t_start

                    # n_calls is the number of oracle calls
                    n_calls = f.n_calls

                    benchmark_df.add(i=n_sample, approx=approx, n_calls=n_calls, time_ns=time_ns)

                    # reset the counter of oracle calls for f
                    f.reset()

    print(f'OK')


if __name__ == '__main__':
    # benchmark_with_timeout()
    # bridge_benchmark()
    # bridge_comparison()
    benchmark()
