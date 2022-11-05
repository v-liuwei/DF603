import pandas as pd
import numpy as np
from config import data_dir, time_range, result_dir, fig_dir
from plot_data import plot_flows


def load_raw(path):
    """
    Load raw data.
    """
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df


def split_data(df: pd.DataFrame, time_range: pd.DataFrame):
    """
    Split data into train and test sets.
    """
    data_dict = {}
    for col in time_range.columns:
        data_dict[col] = df[(df['time'] >= time_range[col][0]) & (df['time'] <= time_range[col][1])].copy()
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


def clean_data(df: pd.DataFrame, outlier_detect_func: callable = detect_outlier_by_zscore, fill_nan_groupby: str = 'hour'):
    """
    Clean data by replacing abnormal values(negatives or outliers) with NaNs, and then filling them groupby time freqency.

    fill_nan_groupby: 'hour', 'day', 'minute'
    """
    df = df.copy()
    flow_cols = [col for col in df.columns if 'flow' in col]

    # replace abnormal values with NaNs
    df[df[flow_cols] < 0] = np.nan
    df[outlier_detect_func(df[flow_cols])] = np.nan

    # forward fill and then back fill NaNs, groupby hour
    df[flow_cols] = df[flow_cols].groupby(getattr(df['time'].dt, fill_nan_groupby)).apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    return df



if __name__ == "__main__":
    # Read the data
    data = load_raw(data_dir / 'hourly_dataset.csv')

    # Split the data into train and test sets
    data_dict = split_data(data, time_range)

    train1 = data_dict['train1']

    # Clean the data
    train1_clean = clean_data(train1)

    plot_flows(train1, samples_per_tick=24*7, n_cols=1, save_path=fig_dir / 'hourly_flows_train1_raw.png')
    plot_flows(train1_clean, samples_per_tick=24*7, n_cols=1, save_path=fig_dir / 'hourly_flows_train1_clean.png')
