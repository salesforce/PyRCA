#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The Hypothesis Testing (HT) algorithm
"""
from dataclasses import dataclass
import pickle
import pandas as pd
from typing import Dict, Union, List
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from pyrca.base import BaseConfig
from pyrca.analyzers.base import BaseRCA, RCAResults


@dataclass
class HTConfig(BaseConfig):
    """
    The configuration class of the HT method for Root Cause Analysis

    :param graph: The adjacency matrix of the causal graphs,
        which can be a pandas dataframe or a file path of a CSV file or a pickled file.
    :param aggregator: The function for aggregating the node score from all the abnormal data.
    :param root_cause_top_k: The maximum number of root causes in the results.
    """

    graph: Union[pd.DataFrame, str]
    aggregator: str = "max"
    root_cause_top_k: int = 3


class HT(BaseRCA):
    """
    Regression-based Hypothesis Testing method for Root Cause Analysis.
    If using this explainer, please cite the original work:
    `Causal Inference-Based Root Cause Analysis for Online Service Systems with Intervention Recognition`.
    """

    config_class = HTConfig

    def __init__(self, config: HTConfig):
        super().__init__()
        self.config = config
        if isinstance(config.graph, str):
            if config.graph.endswith(".csv"):
                graph = pd.read_csv(config.graph)
            elif config.graph.endswith(".pkl"):
                with open(config.graph, "rb") as f:
                    graph = pickle.load(f)
            else:
                raise RuntimeError("The graphs file format is not supported, " "please choose a csv or pickle file.")
        else:
            graph = config.graph
        self.adjacency_mat = graph
        self.graph = nx.from_pandas_adjacency(graph, create_using=nx.DiGraph())
        self.regressors_dict: Dict[str, List[LinearRegression, StandardScaler]] = {}

    @staticmethod
    def _get_aggregator(name):
        if name == "max":
            return max
        elif name == "min":
            return min
        elif name == "sum":
            return sum
        else:
            raise f"Unknown aggregator {name}"

    def train(self, normal_df: pd.DataFrame, **kwargs):
        """
        Train regression model for each node based on its parents. Build the score functions.

        :param normal_df: A pandas dataFrame of normal data.
        """
        assert self.graph is not None, "The graphs is not set."

        for node in list(self.graph):
            parents = list(self.graph.predecessors(node))
            normal_x = normal_df[parents].values
            normal_y = normal_df[node].values
            if normal_x.shape[1] > 0:
                regressor = _regressor = LinearRegression()
                regressor.fit(normal_x, normal_y)
                normal_err = normal_y - _regressor.predict(normal_x)
                scaler = StandardScaler().fit(normal_err.reshape(-1, 1))
                self.regressors_dict[node] = [regressor, scaler]
            else:
                scaler = StandardScaler().fit(normal_y.reshape(-1, 1))
                self.regressors_dict[node] = [None, scaler]

    def find_root_causes(
        self, abnormal_df: pd.DataFrame, anomalous_metrics: str = None, adjustment: bool = False, **kwargs
    ):
        """
        Finds the root causes given the abnormal dataset.

        :param abnormal_df: A pandas dataFrame of abnormal data.
        :param anomalous_metrics: The name of detected anomalous metrics, it is used to print the path from root nodes.
        :param adjustment: Whether to perform descendant adjustment.
        :return: A list of the found root causes.
        """
        node_scores = {}
        for node in list(self.graph):
            parents = list(self.graph.predecessors(node))
            abnormal_x = abnormal_df[parents].values
            abnormal_y = abnormal_df[node].values
            if abnormal_x.shape[1] > 0:
                abnormal_err = abnormal_y - self.regressors_dict[node][0].predict(abnormal_x)
                scores = self.regressors_dict[node][1].transform(abnormal_err.reshape(-1, 1))[:, 0]
            else:
                scores = self.regressors_dict[node][1].transform(abnormal_y.reshape(-1, 1))[:, 0]
            score = self._get_aggregator(self.config.aggregator)(abs(scores))
            conf = 1 - 2 * norm.cdf(-abs(score))
            node_scores[node] = [score, conf]
        if adjustment:
            # start from node with 0 children
            H = self.graph.reverse(copy=True)
            topological_sort = list(nx.topological_sort(H))
            child_nodes = {}
            for node in topological_sort:
                child_nodes[node] = list(self.graph.successors(node))
                for child in child_nodes[node]:
                    if node_scores[child][0] < 3:
                        child_nodes[node] = list(set(child_nodes[node]).union(set(child_nodes[child])))
            for node in list(self.graph):
                if node_scores[node][0] > 3:
                    candidate_scores = [node_scores[child_node][0] for child_node in child_nodes[node]]
                    if len(candidate_scores) == 0:
                        candidate_scores.append(0)
                    node_scores[node][0] = node_scores[node][0] + max(candidate_scores)

        # node_scores[key][1] indicates the confidence
        root_cause_nodes = [(key, node_scores[key][0]) for key in node_scores]
        root_cause_nodes = sorted(root_cause_nodes, key=lambda r: r[1], reverse=True)[: self.config.root_cause_top_k]

        root_cause_paths = {}
        if anomalous_metrics is not None:
            for idx in range(len(root_cause_nodes)):
                try:
                    path = nx.shortest_path(self.graph, source=root_cause_nodes[idx][0], target=anomalous_metrics)
                except nx.exception.NetworkXNoPath:
                    path = None
                root_cause_paths[root_cause_nodes[idx][0]] = path
        return RCAResults(root_cause_nodes=root_cause_nodes, root_cause_paths=root_cause_paths)
