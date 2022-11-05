from pathlib import Path
import pandas as pd


data_dir = Path('../data/training_dataset')
result_dir = Path('../results')
fig_dir = result_dir / 'figs'
result_dir.mkdir(exist_ok=True)
fig_dir.mkdir(exist_ok=True)


time_range = pd.DataFrame(
    {
        'train1': [pd.Timestamp.min, pd.to_datetime("2022-05-01 00:00:00")],
        'test1': [pd.to_datetime("2022-05-01 01:00:00"), pd.to_datetime("2022-05-08 00:00:00")],
        'train2': [pd.to_datetime("2022-05-08 01:00:00"), pd.to_datetime("2022-06-01 00:00:00")],
        'test2': [pd.to_datetime("2022-06-01 01:00:00"), pd.to_datetime("2022-06-08 00:00:00")],
        'train3': [pd.to_datetime("2022-06-08 01:00:00"), pd.to_datetime("2022-07-21 00:00:00")],
        'test3': [pd.to_datetime("2022-07-21 01:00:00"), pd.to_datetime("2022-07-28 00:00:00")],
        'train4': [pd.to_datetime("2022-07-28 01:00:00"), pd.to_datetime("2022-08-21 00:00:00")],
        'test4': [pd.to_datetime("2022-08-21 01:00:00"), pd.to_datetime("2022-08-28 00:00:00")],
    }
)
