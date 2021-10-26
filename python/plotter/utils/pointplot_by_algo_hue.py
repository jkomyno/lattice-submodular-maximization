import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pointplot_by_algo_hue(data: pd.DataFrame, y: str, ylabel: str, title: str,
                          plots_folder: str, filename: str, ax):
    MARKERS = ['o', 'v', 'x', 's', '+', 'd', '<', '>']
    LINESTYLES = [
    (0, (1, 10)),
    (0, (1, 1)),
    (0, (5, 5)),
    (0, (3, 5, 1, 5)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (2, 10)),
    (0, (2, 1)),
    (0, (5, 1)),
    ]
    sns.pointplot(x='B', y=y, hue='algo', data=data,
                       markers=MARKERS, dodge=False,
                       linestyles=LINESTYLES, markeredgecolor=None,
                       ci='sd', legend=True, ax=ax)
    ax.get_legend().set_title('Algorithm')
    ax.set_xlabel('Avg. B')
    ax.set_ylabel(ylabel, labelpad=10)
    output = f'{plots_folder}/{filename}.pdf'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved {output}')
    sns.despine()
    plt.cla()
