import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import fig_dir


def MSLE(y_true, y_pred):
    return np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true)))


if __name__ == "__main__":
    y_true_range = np.arange(0, 50, 5)
    fig, axes = plt.subplots(len(y_true_range), 1, figsize=(6, 2 * len(y_true_range)), sharex=True)
    for y_true, ax in zip(y_true_range, axes):
        y_pred = np.linspace(0, 100, 1000)
        loss = np.square(np.log1p(y_pred) - np.log1p(y_true))
        ax.plot(y_pred, loss)
        ax.set_title(f'y_true={y_true}')
    fig.tight_layout()
    plt.savefig(fig_dir / 'loss_distr.png')
