import pandas as pd
import numpy as np
from config import data_dir, time_range, result_dir, fig_dir
from plot_data import plot_flows
from functools import partial


def load_raw(path, time_col: str = 'time'):
    """
    Load raw data.
    """
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df[time_col])
    if time_col != 'time':
        df = df.drop(columns=time_col)
    return df


def split_data(df: pd.DataFrame, time_range: pd.DataFrame):
    """
    Split data into train and test sets.
    """
    data_dict = {}
    for col in time_range.columns:
        data_dict[col] = df[(df['time'] >= time_range[col][0]) & (df['time'] <= time_range[col][1])].drop(columns='train or test').copy()
    return data_dict


def detect_outlier_by_zscore(df: pd.DataFrame, threshold: float = 3):
    """
    Detect outliers by z-score.
    """
    df = df.copy()
    is_outlier = (df - df.mean()).abs() > threshold * df.std()
    return is_outlier


def detect_outlier_by_iqr(df: pd.DataFrame, threshold: float = 1.5):
    """
    Detect outliers by IQR.
    """
    df = df.copy()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    is_outlier = (df < (q1 - threshold * iqr)) | (df > (q3 + threshold * iqr))
    return is_outlier


def clean_data(df: pd.DataFrame, outlier_detect_func: callable = detect_outlier_by_zscore, fillna: bool = True):
    """
    Clean data by replacing abnormal values(negatives or outliers) with NaNs, and then filling them groupby weekday.
    """
    df = df.copy()
    flow_cols = [col for col in df.columns if 'flow' in col]

    # replace abnormal values with NaNs
    # samples before NaN are abnormal
    df[df[flow_cols].shift(periods=-1, fill_value=0).isna()] = np.nan

    df[df[flow_cols] < 0] = np.nan
    if outlier_detect_func:
        is_outlier = outlier_detect_func(df[flow_cols])
        print(is_outlier.sum())
        df[is_outlier] = np.nan

    if fillna:
        # forward fill and then back fill NaNs
        # try to group by weekday+time
        df[flow_cols] = df[flow_cols].groupby(df['time'].dt.strftime('%u%H%M')).apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        # then group by time
        df[flow_cols] = df[flow_cols].groupby(df['time'].dt.strftime('%H%M')).apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    return df


if __name__ == "__main__":
    # Read the data
    data = load_raw(data_dir / 'hourly_dataset.csv')

    # Split the data into train and test sets
    data_dict = split_data(data, time_range)

    for idx in range(1, 5):
        split = f'train{idx}'
        train = data_dict[split]

        # Clean the data
        train_clean = clean_data(train, partial(detect_outlier_by_zscore, threshold=4))

        plot_flows(train, samples_per_tick=24*7, n_cols=1, save_path=fig_dir / f'hourly_flows_{split}_raw.png')
        plot_flows(train_clean, samples_per_tick=24*7, n_cols=1, save_path=fig_dir / f'hourly_flows_{split}_clean.png')
