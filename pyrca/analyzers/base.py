#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""Base classes for all RCA algorithms"""
from abc import abstractmethod
from dataclasses import dataclass, field, asdict

from pyrca.base import BaseModel
from pyrca.utils.logger import get_logger


@dataclass
class RCAResults:
    """
    The class for storing root cause analysis results.

    :param root_cause_nodes: A list of potential root causes, e.g.,
        [(metric_a, score_a), (metric_b, score_b), ...].
    :param root_cause_paths: A dict of root cause paths, where the key
        is the metric name and the value is a list of paths. Each path has the
        following format: (path_score, [(path_node_a, score_a), (path_node_b, score_b), ...]).
        If ``path_node_a`` has no score, ``score_a`` is set to None.
    """

    root_cause_nodes: list = field(default_factory=lambda: [])
    root_cause_paths: dict = field(default_factory=lambda: {})

    def to_dict(self) -> dict:
        """
        Converts the RCA results into a dict.
        """
        return asdict(self)

    def to_list(self) -> list:
        """
        Converts the RCA results into a list.
        """
        results = []
        for node, score in self.root_cause_nodes:
            results.append({"root_cause": node, "score": score, "paths": self.root_cause_paths.get(node, None)})
        return results


class BaseRCA(BaseModel):
    """
    Base class for RCA algorithms.
    This class should not be used directly, Use derived class instead.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def train(self, **kwargs):
        """
        The training procedure for learning model parameters.
        :param kwargs: Parameters needed for training.
        """

    @abstractmethod
    def find_root_causes(self, **kwargs) -> RCAResults:
        """
        Finds the root causes given the observed anomalous metrics.
        :param kwargs: Additional parameters.
        """
