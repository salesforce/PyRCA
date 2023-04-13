#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import pandas as pd


def load_data(directory, filename):
    if isinstance(filename, (list, tuple)):
        dfs = [load_data(directory, filename=p) for p in filename]
        return pd.concat(dfs, axis=0)

    df = pd.read_csv(os.path.join(directory, filename))
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")

    new_df = None
    for metric in df.columns:
        x = df[[metric]].dropna()
        x = x.resample("Min").agg("ffill").dropna()
        x.index = pd.to_datetime(x.index.values).floor("Min")
        x = x[~x.index.duplicated(keep="first")]
        new_df = x if new_df is None else new_df.join(x, how="inner")
    return new_df.dropna()
