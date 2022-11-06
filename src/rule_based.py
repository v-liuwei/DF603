import pandas as pd
import numpy as np
from config import data_dir, time_range, result_dir, fig_dir
from plot_data import plot_flows
from preprocess import load_raw, split_data, clean_data, detect_outlier_by_zscore
from functools import partial
import time


def copy_last_week(data_dict: dict):
    """
    Let's copy and paste last week's flows!
    """
    tests = []
    for split in range(1, 5):
        train = data_dict['train' + str(split)]
        test = data_dict['test' + str(split)]
        flow_cols = [col for col in train.columns if 'flow' in col]
        # copy last week's flows in train to test
        test[flow_cols] = train[flow_cols].iloc[-len(test):].values
        tests.append(test)
    submission = pd.concat(tests)
    return submission


def copy_last_4weeks(data_dict: dict):
    """
    Let's copy and paste mean of last 4 weeks' flows!
    """
    tests = []
    for split in range(1, 5):
        train = data_dict['train' + str(split)]
        test = data_dict['test' + str(split)]
        flow_cols = [col for col in train.columns if 'flow' in col]           
        test[flow_cols] = np.mean([candidate for i in range(4) if len(candidate := train[flow_cols].iloc[-len(test) * (i+1):-len(test) * i].values) == len(test)], axis=0)
        tests.append(test)
    submission = pd.concat(tests)
    return submission


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
            data_dict[split] = clean_data(data_dict[split], None)

    # copy last week's flows
    submission = copy_last_4weeks(data_dict)

    # save submission
    submission.to_csv(result_dir / f'submission_{timestamp}.csv', index=False)

    # plot flows
    plot_flows(submission, save_path=fig_dir / f'submission_{timestamp}.png')
