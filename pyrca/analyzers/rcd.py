#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The Root Cause Discovery (RCD) algorithm
"""

from dataclasses import dataclass
import pandas as pd

from pyrca.base import BaseConfig
from pyrca.analyzers.base import BaseRCA, RCAResults

from pyrca.thirdparty.causallearn.utils.cit import chisq
from pyrca.thirdparty.causallearn.utils.cit import CIT
from pyrca.thirdparty.rcd import rcd


@dataclass
class RCDConfig(BaseConfig):
    """
    The configuration class for the RCD algorithm for Root Cause Analysis

    :param start_alpha: The desired start significance level (float) in (0, 1) for search.
    :param alpha_step: The search step for alpha.
    :param alpha_limit: The maximum alpha for search.
    :param localized: Whether use local method.
    :param gamma: Chunk size.
    :param bins: The number of bins to discretize data.
    :param K: Top-k root causes.
    :param f_node: The name of anomaly variable.
    :param verbose: True iff verbose output should be printed. Default: False.
    """

    start_alpha: float = 0.01
    alpha_step: float = 0.1
    alpha_limit: float = 1
    localized: bool = True
    gamma: int = 5
    bins: int = 5
    k: int = 3
    f_node: str = "F-node"
    verbose: bool = False
    ci_test: CIT = chisq


class RCD(BaseRCA):
    """
    The RCD algorithm for Root Cause Analysis. If using this explainer, please cite the original work:
    `Root Cause Analysis of Failures in Microservices through Causal Discovery`.
    """

    config_class = RCDConfig

    def __init__(self, config: RCDConfig):
        super().__init__()
        self.config = config

    def train(self, **kwargs):
        """
        model training is implemented in find_root_causes function.
        """
        pass

    def find_root_causes(self, normal_df: pd.DataFrame, abnormal_df: pd.DataFrame, **kwargs):
        """
        Finds the root causes given the abnormal dataset.

        :return: A list of the found root causes.
        """
        result, _ = rcd.run_multi_phase(normal_df, abnormal_df, self.config.to_dict())
        root_cause_nodes = [(key, None) for key in result[: self.config.k]]
        return RCAResults(root_cause_nodes=root_cause_nodes)
