import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from config import fig_dir, result_dir
from plot_data import plot_flows
from preprocess import feature_engineering, prepare_all
from utils import score


def MSLE_objective(preds, train_data):
    # np.square(np.log1p(y_pred) - np.log1p(y_true))
    # return gradient and hessian
    labels = train_data.get_label()
    log1p_error = np.log1p(preds) - np.log1p(labels)
    grad = 2 * log1p_error / (preds + 1)
    hess = 2 * (1 - log1p_error) / (preds + 1) ** 2
    return grad, hess


def MSLE_eval(preds, eval_data):
    labels = eval_data.get_label()
    return 'Score', score(labels, preds), True


def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:num])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp


def evaluation(data_dict, col_dict):
    train = data_dict['train1']
    train = feature_engineering(train, col_dict)
    train, val = train[:-7*24], train[-7*24:]

    feature_cols = train.columns.difference(sum(col_dict.values(), ['time']))
    target_cols = col_dict['flow']

    val_preds = []
    scores = {}
    for target_col in tqdm(target_cols):
        train_x, train_y = train[feature_cols], train[target_col]
        val_x, val_y = val[feature_cols], val[target_col]

        train_set = lgb.Dataset(train_x, train_y)
        val_set = lgb.Dataset(val_x, val_y, reference=train_set)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
        }

        gbm = lgb.train(params, train_set, valid_sets=[train_set, val_set], feval=MSLE_eval)

        y_pred_val = gbm.predict(val_x, num_iteration=gbm.best_iteration)
        val_preds.append(y_pred_val)
        scores[target_col] = score(val_y.values, y_pred_val)
    val_preds = np.stack(val_preds, axis=1)
    print(scores)
    print(score(val[target_cols], val_preds))
    feat_imp = plot_lgb_importances(gbm, plot=True, num=30)
    return feat_imp


def get_submission(data_dict, col_dict):
    train = None
    tests = []
    for idx in range(1, 5):
        train_split, test_split = data_dict[f'train{idx}'], data_dict[f'test{idx}']
        if train is None:
            train = train_split
        else:
            train = pd.concat([train, train_split], axis=0)

        train_featured = feature_engineering(train, col_dict)

        feature_cols = train_featured.columns.difference(sum(col_dict.values(), ['time']))
        target_cols = col_dict['flow']

        models = []
        for target_col in tqdm(target_cols):
            train_x, train_y = train_featured[feature_cols], train_featured[target_col]
            train_set = lgb.Dataset(train_x, train_y)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'verbose': -1,
            }
            gbm = lgb.train(params, train_set)
            models.append(gbm)

        # for i in tqdm(range(len(test_split))):
        #     train = pd.concat([train, test_split.iloc[i:i+1]], axis=0).reset_index(drop=True)
        #     train_featured = feature_engineering(train.iloc[-169:], col_dict)
        #     sample = train_featured.iloc[-1:][feature_cols]
        #     pred = np.array([model.predict(sample) for model in models]).reshape(-1)
        #     test_split.loc[test_split.index[i], target_cols] = pred
        #     train.loc[train.index[-1], target_cols] = pred
        train = pd.concat([train, test_split], axis=0)
        train_featured = feature_engineering(train, col_dict)
        sample = train_featured.iloc[-len(test_split):][feature_cols]
        pred = np.stack([model.predict(sample) for model in models], axis=1)
        test_split[target_cols] = pred
        train.loc[test_split.index, target_cols] = pred
        tests.append(test_split)
    return pd.concat(tests, axis=0)


if __name__ == "__main__":
    timestamp = time.strftime('%m%d%H%M', time.localtime())

    data_dict, col_dict = prepare_all()

    evaluation(data_dict, col_dict)

    # submission = get_submission(data_dict, col_dict)
    # # save submission
    # submission.to_csv(result_dir / f'submission_{timestamp}.csv', index=False)

    # # plot flows
    # plot_flows(submission, save_dir=fig_dir / f'submission_{timestamp}')
