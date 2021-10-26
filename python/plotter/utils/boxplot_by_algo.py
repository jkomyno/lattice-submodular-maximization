import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_by_algo(data: pd.DataFrame, y: str, ylabel: str, title: str,
                    plots_folder: str, filename: str, ax):
    PROPS = {
        'boxprops': {'facecolor':'none', 'edgecolor':'black'},
        'medianprops': {'color':'black'},
        'whiskerprops': {'color':'black'},
        'capprops': {'color':'black'}
    }

    sns.boxplot(x='algo', y=y, data=data, ax=ax, **PROPS)
    # ax.set_title(title)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(ylabel, labelpad=10)
    output = f'{plots_folder}/{filename}.pdf'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved {output}')
    sns.despine()
    plt.cla()
