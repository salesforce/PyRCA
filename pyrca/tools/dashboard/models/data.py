#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import sys
import logging
import pandas as pd
from collections import OrderedDict
from ..pages.utils import create_emtpy_figure
from ..utils.log import DashLogger
from ..utils.plot import data_table, plot_timeseries

from pyrca.utils.utils import estimate_thresholds

dash_logger = DashLogger(stream=sys.stdout)


class DataAnalyzer:
    def __init__(self, folder):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)
        self.folder = folder
        self.df = None

    def load_data(self, file_name):
        df = pd.read_csv(os.path.join(self.folder, file_name))
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        self.df = df.set_index("Timestamp")
        return self.df

    @staticmethod
    def get_stats(df):
        stats = {
            "@global": OrderedDict(
                {
                    "NO. of Variables": len(df.columns),
                    "Time Series Length": df.shape[0],
                    "Has NaNs": bool(df.isnull().values.any()),
                }
            ),
            "@columns": list(df.columns),
        }
        for col in df.columns:
            info = df[col].describe()
            data = OrderedDict(zip(info.index.values, info.values))
            stats[col] = data
        return stats

    @staticmethod
    def get_data_table(df):
        return data_table(df)

    @staticmethod
    def get_data_figure(df):
        if df is None:
            return create_emtpy_figure()
        else:
            return plot_timeseries(df)

    def estimate_threshold(self, column, sigma):
        df = self.df[[column]].dropna()
        lowers, uppers = estimate_thresholds(df, sigmas={}, default_sigma=sigma, win_size=5, reduce="mean")
        lower_df = pd.DataFrame(lowers.tolist() * len(df), columns=["Lower"], index=df.index)
        upper_df = pd.DataFrame(uppers.tolist() * len(df), columns=["Upper"], index=df.index)
        return plot_timeseries([df, lower_df, upper_df])

    def manual_threshold(self, column, lower, upper):
        df = self.df[[column]]
        lower_df = pd.DataFrame([lower] * len(df), columns=["Lower"], index=df.index)
        upper_df = pd.DataFrame([upper] * len(df), columns=["Upper"], index=df.index)
        return plot_timeseries([df, lower_df, upper_df])
