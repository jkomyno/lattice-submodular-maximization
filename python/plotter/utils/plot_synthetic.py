import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from . import boxplot_by_algo
from . import pointplot_by_algo_hue
from ...generate_nbr import generate_nbr


def plot_synthetic(df: pd.DataFrame, plots_folder: str):
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.7)
    fig, ax = plt.subplots(1, figsize=(8, 8))

    for n, r, B in sorted(set(generate_nbr())):
        b_low, b_high = B
    
        data = df \
            .query(f'n == {n} & b_low == {b_low} & b_high == {b_high} & r == {r}')
        ########################
        #  Algorithm vs Value  #
        ########################
        title = f'Algorithm vs Value (n: {n}; B: {[b_low, b_high]}; r: {r})'
        filename = f'algo_vs_value_boxplot-n_{n}-B_{b_low}_{b_high}-r_{r}'
        boxplot_by_algo(data=data, y='approx', ylabel='Value', ax=ax,
                        title=title, plots_folder=plots_folder, filename=filename)

        ###############################
        #  Algorithm vs Oracle Calls  #
        ###############################
        title = f'Algorithm vs Oracle Calls (n: {n}; B: {[b_low, b_high]}; r: {r})'
        filename = f'algo_vs_n_calls_boxplot-n_{n}-B_{b_low}_{b_high}-r_{r}'
        boxplot_by_algo(data=data, y='n_calls', ylabel='Oracle Calls', ax=ax,
                        title=title, plots_folder=plots_folder, filename=filename)

    pairs = set(((n, r) for n, r, _ in sorted(set(generate_nbr()))))
    for n, r in pairs:
        data = df \
            .query(f'n == {n} & r == {r}')
        
        # average between b_low and b_high
        data['B'] = (data['b_low'] + data['b_high']) / 2

        ################
        #  B vs Value  #
        ################
        title = f'B vs Value (n: {n}; r: {r})'
        filename = f'B_vs_value_pointplot-n_{n}-r_{r}'
        pointplot_by_algo_hue(data=data, y='approx', ylabel='Value', ax=ax,
                              title=title, plots_folder=plots_folder, filename=filename)

        #######################
        #  B vs Oracle Calls  #
        #######################
        title = f'B vs Oracle Calls (n: {n}; r: {r})'
        filename = f'B_vs_n_calls_pointplot-n_{n}-r_{r}'
        pointplot_by_algo_hue(data=data, y='n_calls', ylabel='Oracle Calls', ax=ax,
                              title=title, plots_folder=plots_folder, filename=filename)
    
    plt.close(fig)
