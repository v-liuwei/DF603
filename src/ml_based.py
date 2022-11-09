import time
from functools import partial

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import CatBoostModel, ExponentialSmoothing, LightGBMModel, TransformerModel
from tqdm import tqdm

from config import data_dir, fig_dir, result_dir, time_range
from plot_data import plot_flows
from preprocess import (clean_data, detect_outlier_by_zscore, load_raw,
                        split_data)
from utils import score


class NaiveModel:
    """rule based model"""
    def fit(self, series: TimeSeries):
        self.series = series

    def predict(self, n):
        # predict next n // 24 days
        forecast = np.tile((self.series[-(24*7):].values().reshape(7, 24, len(self.series.columns))).mean(axis=0), (n // 24, 1))
        timeindex = pd.date_range(self.series.end_time(), periods=n + 1, freq='H', inclusive='right')
        return TimeSeries.from_times_and_values(timeindex, forecast, columns=self.series.columns)

    def __str__(self):
        return f"NaiveModel()"


def MSLE_objective(y_true, y_pred):
    # np.square(np.log1p(y_pred) - np.log1p(y_true))
    # return gradient and hessian
    log1p_error = np.log1p(y_pred) - np.log1p(y_true)
    grad = 2 * log1p_error / (y_pred + 1)
    hess = 2 * (1 - log1p_error) / (y_pred + 1) ** 2
    return grad, hess


def get_series(df: pd.DataFrame):
    series = TimeSeries.from_dataframe(df, 'time')
    series = series.add_datetime_attribute('weekday')
    series = series.add_holidays('CN')
    return series


def get_submission(data_dict, model):
    known = None
    tests = []
    for idx in tqdm(range(1, 5)):
        train = data_dict['train' + str(idx)]
        test = data_dict['test' + str(idx)]
        if known is None:
            known = train
        else:
            known = pd.concat([known, train])
        target = get_series(known)[flow_cols + ['weekday', 'holidays']]
        model.fit(target)
        forecast = model.predict(24 * 7)
        test[flow_cols] = forecast[flow_cols].values()
        tests.append(test)
        known = pd.concat([known, test])
    return pd.concat(tests)


if __name__ == '__main__':
    timestamp = time.strftime('%m%d%H%M', time.localtime())

    # Read the data
    data = load_raw(data_dir / 'hourly_dataset.csv')
    epidemic_cols = 'glzl,xzqz,xzcy'.split(',')
    weather_cols = 'R,fx,T,U,fs,V,P'.split(',')
    epidemic = load_raw(data_dir / 'epidemic.csv', 'jzrq')[['time'] + epidemic_cols]
    weather = load_raw(data_dir / 'weather.csv')[['time'] + weather_cols]
    data = pd.merge_asof(data, epidemic, on='time')
    data = pd.merge(data, weather, on='time', how='left')

    # Split the data into train and test sets
    data_dict = split_data(data, time_range)

    # clean data
    for split in data_dict:
        if split.startswith('train'):
            data_dict[split] = clean_data(data_dict[split], partial(detect_outlier_by_zscore, threshold=3), fillna=True)

    flow_cols = [col for col in data_dict['train1'].columns if 'flow' in col]

    # train1, test1 = data_dict['train1'], data_dict['test1']

    # target = get_series(train1)[flow_cols + ['weekday', 'holidays']]
    # train, val = target[:-24*7], target[-24*7:]

    # model = NaiveModel()
    # model.fit(train)
    # forecast = model.predict(len(val))
    # print(forecast[flow_cols].values().shape)
    # print("model {} obtains score: {:.2f}%".format(model, score(val[flow_cols].values(), forecast[flow_cols].values())))

    model = LightGBMModel(lags=24*7)
    # model = CatBoostModel(lags=24*7)
    # model = NBEATSModel(input_chunk_length=24*7, output_chunk_length=24, pl_trainer_kwargs={'accelerator': 'gpu', 'devices': [0]})
    # def eval_model(model, train: TimeSeries, val: TimeSeries):
    #     pred = None
    #     for _ in tqdm(range(7)):
    #         model.fit(series=train)
    #         print('fit done.')
    #         forecast = model.predict(24)
    #         train = train.append(forecast)
    #         if pred is None:
    #             pred = forecast
    #         else:
    #             pred = pred.append(forecast)
    #     print(forecast[flow_cols].values().shape)
    #     print("model {} obtains score: {:.2f}%".format(model, score(val[flow_cols].values(), pred[flow_cols].values())))
    # eval_model(model, train, val)

    submission = get_submission(data_dict, model)
    # save submission
    submission.to_csv(result_dir / f'submission_{timestamp}.csv', index=False)

    # plot flows
    plot_flows(submission, save_dir=fig_dir / f'submission_{timestamp}')
