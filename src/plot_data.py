from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from config import data_dir, result_dir, fig_dir
from pathlib import Path


def plot_flows(df: pd.DataFrame, figsize: tuple = (20, 10), samples_per_tick: int = 24*7, save_dir: Path = None):
    """
    Plot all flow columns along time.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    flow_cols = [col for col in df.columns if 'flow' in col]
    for i, col in enumerate(flow_cols):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(df)), df[col])
        ax.set_title(col)
        ax.set_xlabel('time')
        ax.set_ylabel('flow')
        # add ticks
        ax.set_xticks(np.arange(0, len(df), samples_per_tick))
        ax.set_xticklabels(pd.to_datetime(df['time'])[::samples_per_tick].dt.strftime('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=80)
        fig.tight_layout()
        fig.savefig(save_dir / f'{col}.png')
        plt.close(fig)
