#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


class Scaler:
    scalars = {"minmax": MinMaxScaler, "robust": RobustScaler, "standard": StandardScaler, "none": None}

    def __init__(self, scalar_type="standard"):
        assert scalar_type in Scaler.scalars, (
            f"The scalar type {scalar_type} is not supported. " f"Please choose from {Scaler.scalars.keys()}."
        )
        scaler_class = Scaler.scalars[scalar_type]
        self.scaler = None if scaler_class is None else scaler_class()

    def fit(self, x):
        if self.scaler is not None:
            self.scaler.fit(x)
        return self

    def transform(self, x):
        if self.scaler is None:
            return x.values if isinstance(x, pd.DataFrame) else x
        else:
            return self.scaler.transform(x)


def normalize_data(train_data, test_data=None, normalizer="none"):
    normalizer = Scaler(normalizer).fit(train_data)
    train_data = normalizer.transform(train_data)
    if test_data is not None:
        test_data = normalizer.transform(test_data)
    return train_data, test_data


def remove_outliers(df, scale=5.0):
    data = df.values
    medians = np.median(data, axis=0)
    a = np.percentile(data, 99, axis=0)
    b = np.percentile(data, 1, axis=0)
    max_value = (a - medians) * scale + medians
    min_value = (b - medians) * scale + medians

    indices = []
    for i in range(data.shape[0]):
        x = data[i]
        f = np.sum((x > max_value).astype(int) + (x < min_value).astype(int))
        if f == 0:
            indices.append(i)
    data = data[indices, :]
    return pd.DataFrame(data, columns=df.columns)


def discretize(df, percentile=95):
    x = df.values
    thres = np.percentile(x, percentile, axis=0)
    x = (x > thres).astype(int)
    return pd.DataFrame(x, index=df.index, columns=df.columns)


def timeseries_window(df, begin_date, end_date):
    begin, end = None, None
    if begin_date is not None:
        begin = np.datetime64(begin_date)
    if end_date is not None:
        end = np.datetime64(end_date)

    if begin and end:
        return df.loc[(df.index >= begin) & (df.index <= end)]
    elif begin:
        return df.loc[df.index >= begin]
    elif end:
        return df.loc[df.index <= end]
    else:
        return df


def estimate_thresholds(df, sigmas, default_sigma=4, win_size=5, reduce="mean", return_mean_std=False):
    x = df.values
    x = np.array([np.mean(x[max(0, i - win_size) : i + 1, :], axis=0) for i in range(x.shape[0])])
    a = np.percentile(x, 0.1, axis=0)
    b = np.percentile(x, 99.9, axis=0)
    x = np.maximum(np.minimum(x, b), a)

    if reduce == "mean":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
    elif reduce == "median":
        mean = np.median(x, axis=0)
        std = np.median(np.abs(x - mean), axis=0)
    else:
        raise ValueError(f"Unknown reduce function: {reduce}")

    lowers = np.zeros(mean.shape[0])
    uppers = np.zeros(mean.shape[0])
    for i, col in enumerate(df.columns):
        lowers[i] = mean[i] - sigmas.get(col, default_sigma) * std[i]
        uppers[i] = sigmas.get(col, default_sigma) * std[i] + mean[i]

    max_values = np.max(df.values, axis=0)
    min_values = np.min(df.values, axis=0)
    for i, col in enumerate(df.columns):
        lowers[i] = max(min_values[i], lowers[i])
        uppers[i] = min(max_values[i], uppers[i])
        # If it is a constant value
        if lowers[i] == uppers[i]:
            uppers[i] += 1e-5

    if not return_mean_std:
        return lowers, uppers
    else:
        return lowers, uppers, mean, std
