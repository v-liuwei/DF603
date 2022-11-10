import time

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel, ExponentialSmoothing
from tqdm import tqdm

from preprocess import prepare_all
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
        target = get_series(known)[col_dict['flow']]
        model.fit(target)
        forecast = model.predict(24 * 7)
        test[col_dict['flow']] = forecast[col_dict['flow']].values()
        tests.append(test)
        known = pd.concat([known, test])
    return pd.concat(tests)


if __name__ == '__main__':
    timestamp = time.strftime('%m%d%H%M', time.localtime())

    data_dict, col_dict = prepare_all()

    train1, test1 = data_dict['train1'], data_dict['test1']

    target = get_series(train1)
    train, val = target[:-24*7], target[-24*7:]

    model = NaiveModel()
    model.fit(train)
    forecast = model.predict(len(val))
    print(forecast[col_dict['flow']].values().shape)
    print("model {} obtains score: {:.2f}%".format(model, score(val[col_dict['flow']].values(), forecast[col_dict['flow']].values())))

    model = LightGBMModel(lags=24*7)
    # model = CatBoostModel(lags=24*7)
    # model = NBEATSModel(input_chunk_length=24*7, output_chunk_length=24, pl_trainer_kwargs={'accelerator': 'gpu', 'devices': [0]})
    def eval_model(model, train: TimeSeries, val: TimeSeries):
        model.fit(train[col_dict['flow']])
        pred = model.predict(len(val))
        print("model {} obtains score: {:.2f}%".format(model, score(val[col_dict['flow']].values(), pred[col_dict['flow']].values())))
    eval_model(model, train, val)

    # submission = get_submission(data_dict, model)
    # # save submission
    # submission.to_csv(result_dir / f'submission_{timestamp}.csv', index=False)

    # # plot flows
    # plot_flows(submission, save_dir=fig_dir / f'submission_{timestamp}')
