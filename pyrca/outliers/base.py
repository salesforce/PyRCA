#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
Base classes for all outliers.
"""
import itertools
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Dict
from dataclasses import dataclass, field, asdict

from pyrca.base import BaseModel
from pyrca.utils.logger import get_logger


class BaseDetector(BaseModel):
    """
    Base class for Outlier (Anomaly) Detectors.
    This class should not be used directly, Use dervied class instead.
    """

    config_class = None

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = None

    def train(self, df: pd.DataFrame, **kwargs):
        """
        Train method for fitting the data.

        :param df: The training dataset.
        :param kwargs: Parameters needed for training.
        :return:
        """
        self.logger.info("Training for detector: %s", self.__class__.__name__)
        # self.logger.info(self._get_init_params())
        return self._train(df, **kwargs)

    def predict(self, df: pd.DataFrame, **kwargs):
        """
        Predict anomalies/outliers given the input data.

        :param df: The test dataset.
        :param kwargs: Parameters needed for prediction.
        :return: The detection results.
        """
        self.logger.info("Detecting anomalies with detector: %s", self.__class__.__name__)
        return self._predict(df, **kwargs)

    @abstractmethod
    def _train(self, df: pd.DataFrame, **kwargs):
        """
        Derived Class to implement the _train procedure based on the algorithm.

        :param df: The training dataset.
        :param kwargs: Kwargs needed for training.
        """

    @abstractmethod
    def _predict(self, df: pd.DataFrame, **kwargs):
        """
        Derived Class to implement the _predict procedure based on the algorithm.

        :param df: The test dataset.
        :param kwargs: Kwargs needed for prediction.
        """

    def update_config(self, d: Dict):
        """
        Updates the configurations (hyperparameters).

        :param d: The new parameters in a dict format.
        """
        config = self.config.to_dict()
        for key, value in d.items():
            if key not in config:
                self.logger.warning(f"The config object has no parameter called {key}.")
                continue
            if not isinstance(value, dict) or config[key] is None:
                config[key] = value
            else:
                for k, v in value.items():
                    config[key][k] = v
        self.config = self.config_class.from_dict(config)


class DetectorMixin:
    """
    Check data quality and train
    """

    @staticmethod
    def _check_nan(df, **kwargs):
        assert not bool(df.isnull().values.any()), "The input dataframe contains NaNs."

    @staticmethod
    def _check_column_names(df, **kwargs):
        for col in df.columns:
            assert isinstance(col, str), f"The column name must be a string instead of {type(col)}."
            assert " " not in col, f"The column name cannot contains a SPACE: {col}."
            assert "#" not in col, f"The column name cannot contains #: {col}."

    @staticmethod
    def _check_length(df, min_length=3000, **kwargs):
        assert len(df) >= min_length, f"The number of data points is less than {min_length}."

    @staticmethod
    def _check_data_type(df, **kwargs):
        for t in df.dtypes:
            assert t in [
                np.int,
                np.int32,
                np.int64,
                np.float,
                np.float32,
                np.float64,
            ], f"The data type {t} is not int or float."

    def check_data_and_train(self, df, **kwargs):
        """
        Data quality check and training.

        :param df: The training dataset.
        :param kwargs: Additional parameters.
        """
        # Check data format and quality
        for checker in dir(DetectorMixin):
            if checker.startswith("_check"):
                getattr(DetectorMixin, checker)(df, **kwargs)
        # Run training
        self.train(df, **kwargs)


@dataclass
class DetectionResults:
    """
    The class for storing anomaly detection results.
    """

    anomalous_metrics: list = field(default_factory=lambda: [])
    anomaly_timestamps: dict = field(default_factory=lambda: {})
    anomaly_labels: dict = field(default_factory=lambda: {})
    anomaly_info: dict = field(default_factory=lambda: {})

    def to_dict(self) -> dict:
        """
        Converts the anomaly detection results into a dict.
        """
        return asdict(self)

    @classmethod
    def merge(cls, results: list):
        """
        Merges multiple detection results.

        :param results: A list of ``DetectionResults`` objects.
        :return: The merged ``DetectionResults`` object.
        """
        res = DetectionResults()
        res.anomalous_metrics = list(itertools.chain(*[r.anomalous_metrics for r in results]))
        res.anomaly_timestamps = dict(itertools.chain(*[r.anomaly_timestamps.items() for r in results]))
        res.anomaly_labels = dict(itertools.chain(*[r.anomaly_labels.items() for r in results]))
        res.anomaly_info = dict(itertools.chain(*[r.anomaly_info.items() for r in results]))
        return res
