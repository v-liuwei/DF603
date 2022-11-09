import pandas as pd
import numpy as np
from config import data_dir, time_range, result_dir, fig_dir
from plot_data import plot_flows
from preprocess import load_raw, split_data, clean_data, detect_outlier_by_zscore
from functools import partial
import time
from scipy.optimize import fmin


def agg_last_n_days(data_dict: dict[str, pd.DataFrame], n: int = 7, agg_fn = partial(np.mean, axis=0)):
    """
    Let's copy and paste last n days' flows!
    """
    tests = []
    for idx in range(1, 5):
        train = data_dict['train' + str(idx)]
        test = data_dict['test' + str(idx)]
        flow_cols = [col for col in train.columns if 'flow' in col]
        # copy last n days' flows in train to test
        test[flow_cols] = np.tile(
            agg_fn(train[flow_cols].iloc[-(24*n):].values.reshape(min(n, len(train) // 24), 24, len(flow_cols))), 
        (7, 1))
        tests.append(test)
    submission = pd.concat(tests)
    return submission


def agg_normal(arr: np.ndarray):
    """
    Aggregate candidates based on loss function np.square(np.log1p(y_pred) - np.log1p(y_true))
    """
    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    k = 50
    samples = np.clip(np.random.normal(mean, std, size=(k, *mean.shape)), 0, None)
    loss_fn = lambda y_pred: np.square(np.log1p(y_pred) - np.log1p(samples.reshape(k, -1))).mean()
    # find minimum of function loss_fn
    y_pred = fmin(loss_fn, samples.mean(axis=0), disp=False).reshape(mean.shape)
    # print(loss_fn(y_true.mean()), loss_fn(y_pred))
    print('Done.')
    return y_pred


if __name__ == '__main__':
    # get current timestamp in format of 'MMDDHHMM'
    timestamp = time.strftime('%m%d%H%M', time.localtime())

    # Read the data
    data = load_raw(data_dir / 'hourly_dataset.csv')

    # Split the data into train and test sets
    data_dict = split_data(data, time_range)

    # clean data
    for split in data_dict:
        if split.startswith('train'):
            data_dict[split] = clean_data(data_dict[split], partial(detect_outlier_by_zscore, threshold=3), fillna=True)

    submission = agg_last_n_days(data_dict, n=7, agg_fn=agg_normal)

    # save submission
    submission.to_csv(result_dir / f'submission_{timestamp}.csv', index=False)

    # plot flows
    plot_flows(submission, save_dir=fig_dir / f'submission_{timestamp}')
