import os
import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path
from glob import glob
from .. import common
from ..generate_nbr import generate_nbr
from . import utils


pd.options.mode.chained_assignment = None


def import_csvs_by_f_name(input_folder: str, f_name: str):
    dtype = dict([
        ('i', np.int8),
        ('n', np.int32),
        ('b', np.int32),
        ('r', np.int32),
        ('approx', np.float64),
        ('n_calls', np.int64),
        ('time_ms', np.int64),
    ])
    df_list = []

    for obj_algo_csv in glob(os.path.join(input_folder, f_name, '*.csv')):
        filename = os.path.splitext(os.path.basename(obj_algo_csv))[0]

        # find algorithm name
        algo = '-'.join(filename.split('-'))

        curr_df = common.read_csv(obj_algo_csv, dtype=dtype)
        curr_df = curr_df.assign(algo=algo)
        df_list.append(curr_df)

    df = pd.concat(df_list)
    df.attrs['obj'] = f_name
    df.sort_values(by=['algo', 'n', 'r', 'b_low', 'b_high', 'i'], inplace=True)

    return df


@hydra.main(config_path="../conf", config_name="config")
def plotter(cfg: DictConfig) -> None:

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)
        
    ###################
    #  demo_monotone  #
    ###################

    demo_monotone_df = import_csvs_by_f_name(f'{basedir}/out/', f_name='demo_monotone')
    df = demo_monotone_df
    plots_folder = f'{basedir}/out/{df.attrs["obj"]}/plots'
    Path(plots_folder).mkdir(parents=True, exist_ok=True)
    utils.plot_synthetic(demo_monotone_df, plots_folder)
