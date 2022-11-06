from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from config import data_dir, result_dir, fig_dir


def plot_flows(df: pd.DataFrame, n_cols: int = 2, single_figsize: tuple = (10, 10), samples_per_tick: int = 24*7, save_path = None):
    """
    Plot all flow columns along time in subplots.
    """
    flow_cols = [col for col in df.columns if 'flow' in col]
    n_rows = math.ceil(len(flow_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * single_figsize[0], n_rows * single_figsize[1]), squeeze=False)
    for i, col in enumerate(flow_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(range(len(df)), df[col])
        ax.set_title(col)
        ax.set_xlabel('time')
        ax.set_ylabel('flow')
        # add ticks
        ax.set_xticks(np.arange(0, len(df), samples_per_tick))
        ax.set_xticklabels(df['time'][::samples_per_tick].dt.strftime('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=80)
    fig.tight_layout()
    fig.savefig(save_path)


if __name__ == "__main__":
    daily_data = pd.read_csv(data_dir / 'daily_dataset.csv')
    plot_flows(daily_data, samples_per_tick=30, save_path=fig_dir / 'daily_flows_raw.png')

    hourly_data = pd.read_csv(data_dir / 'hourly_dataset.csv')
    plot_flows(hourly_data, samples_per_tick=24*7, save_path=fig_dir / 'figs/hourly_flows_raw.png')

    per5min_data = pd.read_csv(data_dir / 'per5min_dataset.csv')
    plot_flows(per5min_data, samples_per_tick=12*24*7, save_path=fig_dir / 'figs/per5min_flows_raw.png')
