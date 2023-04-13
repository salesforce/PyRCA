#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The statistical-based anomaly detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from copy import deepcopy
from dataclasses import dataclass

from pyrca.base import BaseConfig
from pyrca.outliers.base import BaseDetector, DetectorMixin, DetectionResults
from pyrca.utils.utils import estimate_thresholds


@dataclass
class StatsDetectorConfig(BaseConfig):
    """
    The configuration class for the stats anomaly detector.

    :param default_sigma: The default sigma value for computing the threshold,
        e.g., abs(x - mean) > sigma * std.
    :param thres_win_size: The size of the smoothing window for computing bounds.
    :param thres_reduce_func: The reduction function for bounds, i.e., "mean" uses
        the mean value and standard deviation, "median" uses the median value and median
        absolute deviation.
    :param score_win_size: The default window size for computing anomaly scores.
    :param anomaly_threshold: The default anomaly detection threshold, e.g., what percentage of points
        in a small window (with size `score_win_size`) that violates abs(x - mean) <= sigma * std
        is considered as an anomaly.
    :param sigmas: Variable-specific sigmas other than default for certain variables.
    :param manual_thresholds: Manually specified lower and upper thresholds, e.g., {"lower": 0, "upper": 10}.
    :param custom_win_sizes: Variable-specific window sizes other than default for certain variables.
    :param custom_anomaly_thresholds: Variable-specific anomaly detection thresholds other than
        default for certain variables.
    """

    default_sigma: float = 4.0
    thres_win_size: int = 5
    thres_reduce_func: str = "mean"
    score_win_size: int = 3
    anomaly_threshold: float = 0.5

    sigmas: dict = None
    manual_thresholds: dict = None
    custom_win_sizes: dict = None
    custom_anomaly_thresholds: dict = None


class StatsDetector(BaseDetector, DetectorMixin):
    """
    The statistics-based anomaly detector. During training, it will estimate the mean and std of the
    training time series. During prediction/detection, for each timestamp t, it will consider a small window
    around t, and compute the anomaly score based on the percentage of the points in this small window such
    that abs(x - mean) > sigma * std. If this percentage is greater than a certain threshold, the timestamp t
    is considered as an anomaly.
    """

    config_class = StatsDetectorConfig

    def __init__(self, config: StatsDetectorConfig):
        super(StatsDetector, self).__init__()
        self.config = config
        self.bounds = {}
        self.mean_stds = {}

    def to_dict(self) -> Dict:
        """
        Converts a trained detector into a python dictionary.
        """
        return {"config": self.config.to_dict(), "bounds": deepcopy(self.bounds), "mean_stds": deepcopy(self.mean_stds)}

    @classmethod
    def from_dict(cls, d: Dict) -> StatsDetector:
        """
        Creates a ``StatsDetector`` from a python dictionary.
        """
        self = StatsDetector(StatsDetectorConfig.from_dict(d["config"]))
        self.bounds = deepcopy(d["bounds"])
        self.mean_stds = deepcopy(d["mean_stds"])
        return self

    def _train(self, df, **kwargs):
        if df.isnull().values.any():
            self.logger.warning("The training data contains NaN.")
            df = df.dropna()

        sigmas = {} if self.config.sigmas is None else self.config.sigmas
        manual_thresholds = {} if self.config.manual_thresholds is None else self.config.manual_thresholds
        lowers, uppers, means, stds = estimate_thresholds(
            df=df,
            sigmas=sigmas,
            default_sigma=self.config.default_sigma,
            win_size=self.config.thres_win_size,
            reduce=self.config.thres_reduce_func,
            return_mean_std=True,
        )
        for i, col in enumerate(df.columns):
            lower_bound = lowers[i]
            upper_bound = uppers[i]
            if col in manual_thresholds:
                lower_bound = manual_thresholds[col].get("lower", lower_bound)
                upper_bound = manual_thresholds[col].get("upper", upper_bound)
            self.bounds[col] = (lower_bound, upper_bound)
            self.mean_stds[col] = (means[i], stds[i])
            if self.logger is not None:
                self.logger.debug(f"Stats bounds for {col}: {self.bounds[col]}")
                self.logger.debug(f"Mean and std for {col}: {self.mean_stds[col]}")

    def _get_anomaly_scores(self, df):
        all_scores = []
        for i in range(len(df)):
            scores = []
            for col in df.columns:
                w = (
                    self.config.custom_win_sizes.get(col, self.config.score_win_size)
                    if self.config.custom_win_sizes
                    else self.config.score_win_size
                )
                x = df[col].values[max(0, i - w) : i + w]
                y = (x < self.bounds[col][0]).astype(int) + (x > self.bounds[col][1]).astype(int)
                scores.append(y.sum() / len(y))
            all_scores.append(scores)
        return np.array(all_scores)

    def _predict(self, df, **kwargs):
        if df.isnull().values.any():
            self.logger.warning("The test data contains NaN.")
            df = df.dropna()

        timestamps = df.index.values
        all_scores = self._get_anomaly_scores(df)
        max_scores = np.max(all_scores, axis=0)

        anomalous_metrics = []
        for metric, score in zip(df.columns, max_scores):
            thres = (
                self.config.custom_anomaly_thresholds.get(metric, self.config.anomaly_threshold)
                if self.config.custom_anomaly_thresholds
                else self.config.anomaly_threshold
            )
            if score > thres:
                anomalous_metrics.append(metric)

        anomaly_timestamps = {}
        anomaly_labels = {}
        anomaly_info = {}
        for i, col in enumerate(df.columns):
            if col not in anomalous_metrics:
                continue
            # Anomaly labels and timestamps
            x = df[col].values
            y = (x < self.bounds[col][0]).astype(int) + (x > self.bounds[col][1]).astype(int)
            anomaly_labels[col] = pd.DataFrame((y > 0).astype(int), columns=[col], index=df.index)
            anomaly_timestamps[col] = [timestamps[t] for t in range(len(y)) if y[t] > 0]

            # More details about anomalous data points
            anomaly_info[col] = {
                "normal_range": self.bounds[col],
                "mean_std": self.mean_stds.get(col, None),
                "anomalies": [],
            }
            for t in range(len(y)):
                if y[t] > 0:
                    anomaly_info[col]["anomalies"].append(
                        {
                            "timestamp": timestamps[t],
                            "value": x[t],
                            "absolute_deviation": np.abs(x[t] - self.mean_stds[col][0])
                            if col in self.mean_stds
                            else None,
                            "z_score": np.abs(x[t] - self.mean_stds[col][0]) / (self.mean_stds[col][1] + 1e-5)
                            if col in self.mean_stds
                            else None,
                        }
                    )

        return DetectionResults(
            anomalous_metrics=anomalous_metrics,
            anomaly_timestamps=anomaly_timestamps,
            anomaly_labels=anomaly_labels,
            anomaly_info=anomaly_info,
        )

    def update_bounds(self, d: Dict):
        """
        Updates the bounds manually.

        :param d: The bounds of certain metrics, e.g., {"metric A": (0, 1)}.
        """
        for key, values in d.items():
            self.bounds[key] = values
