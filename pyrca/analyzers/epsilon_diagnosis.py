#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The epsilon-Diagnosis algorithm.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from itertools import combinations

from pyrca.base import BaseConfig
from pyrca.analyzers.base import BaseRCA, RCAResults


@dataclass
class EpsilonDiagnosisConfig(BaseConfig):
    """
    The configuration class for the epsilon-diagnosis algorithm for Root Cause Analysis.

    :param alpha: The desired significance level (float) in (0, 1). Default: 0.05.
    :param bootstrap_time: Bootstrap times.
    :param root_cause_top_k: The maximum number of root causes in the results.
    """

    alpha: float = 0.05
    bootstrap_time: int = 200
    root_cause_top_k: int = 3


class EpsilonDiagnosis(BaseRCA):
    """
    The epsilon-diagnosis method for Root Cause Analysis. If using this method, please cite the original work:
    `epsilon-Diagnosis: Unsupervised and Real-time Diagnosis of Small window Long-tail Latency in Large-scale Microservice Platforms`.
    """

    config_class = EpsilonDiagnosisConfig

    def __init__(self, config: EpsilonDiagnosisConfig):
        super().__init__()
        self.config = config

    def train(self, normal_df: pd.DataFrame, **kwargs):
        """
        Two variable correlation analysis given the training time series.

        :param normal_df: A pandas dataframe of normal data.
        """
        self.normal_df = normal_df

        def _samples(array, times=50):
            return np.random.choice(array, (array.shape[0], times))

        # bootstrapping to calculate the p-value
        normal_df_msample = np.apply_along_axis(_samples, 0, normal_df.values, times=self.config.bootstrap_time)
        normal_correlations = np.empty(
            (int((normal_df_msample.shape[1] * (normal_df_msample.shape[1] - 1) / 2)), normal_df_msample.shape[2])
        )
        for k in range(normal_df_msample.shape[2]):
            cov_matrix = np.cov(normal_df_msample[:, :, k].T)
            idx = 0
            for i, j in combinations(np.arange(normal_df_msample.shape[1]), 2):
                if cov_matrix[i, i] == 0 or cov_matrix[j, j] == 0:
                    normal_correlations[idx, k] = 0
                else:
                    normal_correlations[idx, k] = np.square(cov_matrix[i, j]) / (cov_matrix[i, i] * cov_matrix[j, j])
                idx += 1
        self.statistics = dict(
            zip(normal_df.columns, np.apply_along_axis(np.quantile, 0, normal_correlations, q=1 - self.config.alpha))
        )

    def find_root_causes(self, abnormal_df: pd.DataFrame, **kwargs):
        """
        Finds the root causes given the abnormal dataset.

        :param abnormal_df: A pandas dataFrame of abnormal data.
        :return: A list of the found root causes.
        """
        root_cause_nodes = []
        self.correlations = {}
        for colname in abnormal_df.columns:
            if np.var(self.normal_df[colname].values) == 0 or np.var(abnormal_df[colname].values) == 0:
                self.correlations[colname] = 0
            else:
                self.correlations[colname] = np.square(
                    np.cov(self.normal_df[colname].values, abnormal_df[colname].values)[0, 1]
                ) / (np.var(self.normal_df[colname].values) * np.var(abnormal_df[colname].values))
                if self.correlations[colname] > self.statistics[colname]:
                    root_cause_nodes.append((colname, self.correlations[colname]))
        root_cause_nodes = sorted(root_cause_nodes, key=lambda r: r[1], reverse=True)[: self.config.root_cause_top_k]
        return RCAResults(root_cause_nodes=root_cause_nodes)
