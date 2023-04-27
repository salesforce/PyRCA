#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The RCA method based on random walk
"""
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from typing import Union, Dict, List
from collections import defaultdict
from dataclasses import dataclass

from pyrca.base import BaseConfig
from pyrca.analyzers.base import BaseRCA, RCAResults


@dataclass
class RandomWalkConfig(BaseConfig):
    """
    The configuration class for the random walk based RCA.

    :param graph: The adjacency matrix of the causal graph,
        which can be a pandas dataframe or a file path of a CSV file or a pickled file.
    :param use_partial_corr: Whether to use partial correlation when computing edge weights.
    :param rho: The weight from a "cause" node to a "result" node.
    :param num_steps: The number of random walk steps in each run.
    :param num_repeats: The number of random walk runs.
    :param root_cause_top_k: The maximum number of root causes in the results.
    """

    graph: Union[pd.DataFrame, str]
    use_partial_corr: bool = False
    rho: float = 0.1
    num_steps: int = 10
    num_repeats: int = 1000
    root_cause_top_k: int = 5


class RandomWalk(BaseRCA):
    """
    The RCA method based on random walk on the topology/causal graph.
    """

    config_class = RandomWalkConfig

    def __init__(self, config: RandomWalkConfig):
        super().__init__()
        self.config = config
        if isinstance(config.graph, str):
            if config.graph.endswith(".csv"):
                graph = pd.read_csv(config.graph)
            elif config.graph.endswith(".pkl"):
                with open(config.graph, "rb") as f:
                    graph = pickle.load(f)
            else:
                raise RuntimeError("The graph file format is not supported, " "please choose a csv or pickle file.")
        else:
            graph = config.graph
        self.adjacency_mat = graph
        self.graph = nx.from_pandas_adjacency(graph, create_using=nx.DiGraph())
        self.use_partial_corr = config.use_partial_corr

    @staticmethod
    def _partial_correlation(df: pd.DataFrame, x, y, z, add_noise=True):
        import pingouin as pg

        if add_noise:
            noise = np.random.normal(0, 1e-8, size=df.shape)
            df = df.add(pd.DataFrame(noise, columns=df.columns, index=df.index))
        corr = pg.partial_corr(data=df, x=x, y=y, covar=z)
        p_val = corr["p-val"].values[0]
        if not np.isnan(p_val):
            return corr["r"].values[0]
        else:
            raise RuntimeError("The p-value for partial correlation is NaN, " "please add more data points.")

    @staticmethod
    def _correlation(df: pd.DataFrame, x, y):
        corr = df[x].corr(df[y])
        return 0.0 if np.isnan(corr) else corr

    def _compute_weight(self, df, anomaly, metric):
        if anomaly == metric:
            return 1
        if metric in list(self.graph.successors(anomaly)):
            return 0
        if self.use_partial_corr:
            z = list(self.graph.predecessors(anomaly)) + list(self.graph.predecessors(metric))
            ps = list(set([p for p in z if p != metric and p != anomaly]))
            weight = (
                abs(self._partial_correlation(df, anomaly, metric, ps))
                if len(ps) > 0
                else abs(self._correlation(df, anomaly, metric))
            )
        else:
            weight = abs(self._correlation(df, anomaly, metric))
        return weight

    def _node_weight(self, df, anomalies):
        q = defaultdict(list)
        for anomaly in anomalies:
            for metric in df.columns:
                q[metric].append(self._compute_weight(df, anomaly, metric))
        return q

    def _self_weight(self, df, anomalies, node_weights):
        q = defaultdict(list)
        for i, anomaly in enumerate(anomalies):
            for metric in df.columns:
                pb = self.graph.predecessors(metric)
                ps = [node_weights[p][i] for p in pb if p in node_weights]
                q_max = 0 if len(ps) == 0 else max(ps)
                w = max(0, node_weights[metric][i] - q_max)
                q[metric].append(w if metric not in anomalies else 0)
        return q

    def _build_weighted_graph(self, df, anomalies, rho):
        metrics = df.columns
        node_weights = self._node_weight(df, anomalies)
        self_weights = self._self_weight(df, anomalies, node_weights)
        node_ws = {metric: max(values) for metric, values in node_weights.items()}
        self_ws = {metric: max(values) for metric, values in self_weights.items()}
        graph = {m: {"nodes": [], "weights": [], "probs": None} for m in metrics}

        for metric in metrics:
            # To cause
            for p in self.graph.predecessors(metric):
                graph[metric]["nodes"].append(p)
                graph[metric]["weights"].append(node_ws[p])
            # To result
            for p in self.graph.successors(metric):
                graph[metric]["nodes"].append(p)
                graph[metric]["weights"].append(node_ws[p] * rho)
            # To self
            graph[metric]["nodes"].append(metric)
            graph[metric]["weights"].append(self_ws[metric])

        for metric in graph.keys():
            w = np.array(graph[metric]["weights"])
            graph[metric]["probs"] = w / sum(w)
        return graph

    @staticmethod
    def _random_walk(graph, start, num_steps, num_repeats, random_seed=0):
        if random_seed is not None:
            np.random.seed(random_seed)
        counts = {}
        for _ in range(num_repeats):
            path, node = [], start
            for _ in range(num_steps):
                probs = graph[node]["probs"]
                index = np.random.choice(list(range(len(probs))), p=probs)
                node = graph[node]["nodes"][index]
                path.append(node)
            for node in path:
                counts[node] = counts.get(node, 0) + 1
        return counts

    @staticmethod
    def _find_all_paths(graph, u, v):
        from collections import deque

        q, paths = deque([(u, [])]), []
        while q:
            node, path = q.popleft()
            path.append(node)
            if node == v:
                paths.append(path)
            else:
                for k in graph.successors(node):
                    q.append((k, path[:]))
        return paths

    @staticmethod
    def _get_node_levels(graph):
        queue, visited = [], {}
        for i in range(graph.shape[0]):
            if sum(graph.values[:, i]) == 0:
                queue.append(i)
                visited[i] = True
        levels = {0: [graph.columns[k] for k in queue]}
        while len(queue) > 0:
            new_queue = []
            for i in queue:
                for j in range(graph.shape[0]):
                    if i != j and graph.values[i, j] > 0 and not visited.get(j, False):
                        new_queue.append(j)
                        visited[j] = True
            if new_queue:
                levels[len(levels)] = [graph.columns[k] for k in new_queue]
            queue = new_queue
        return levels

    def _get_root_cause_paths(self, root, anomaly, root_cause_scores):
        paths = self._find_all_paths(self.graph, root, anomaly)
        scores = [sum(root_cause_scores[node] for node in path) / len(path) for path in paths]
        return sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)

    def train(self, **kwargs):
        """
        Random walks needs no training.
        """
        pass

    def find_root_causes(self, anomalous_metrics: Union[List, Dict], df: pd.DataFrame, **kwargs) -> RCAResults:
        """
        Finds the root causes given the observed anomalous metrics.

        :param anomalous_metrics: A list of anomalous metrics. ``anomalous_metrics`` is either a list
            ['metric_A', 'metric_B', ...] or a dict {'metric_A': 1, 'metric_B': 1}.
        :param df: The time series dataframe in the incident window.
        :return: A list of the found root causes.
        """
        if isinstance(anomalous_metrics, dict):
            anomalous_metrics = [key for key, value in anomalous_metrics.items() if value > 0]
        levels = self._get_node_levels(self.adjacency_mat)
        graph = self._build_weighted_graph(df, anomalous_metrics, self.config.rho)
        counts = {
            anomaly: self._random_walk(
                graph,
                anomaly,
                self.config.num_steps,
                self.config.num_repeats,
                random_seed=kwargs.get("random_seed", None),
            )
            for anomaly in anomalous_metrics
        }
        merged_counts = {}
        for anomaly in anomalous_metrics:
            for key, value in counts[anomaly].items():
                merged_counts[key] = merged_counts.get(key, 0) + value

        for metric in df.columns:
            if metric not in merged_counts:
                merged_counts[metric] = 0
        node_scores = {m: v / sum(merged_counts.values()) for m, v in merged_counts.items()}

        root_cause_nodes = []
        root_cause_paths = {}
        scores = [node_scores[m] for m in levels[0]]
        for root, s in sorted(zip(levels[0], scores), key=lambda x: x[1], reverse=True)[:4]:
            root_cause_nodes.append((root, s))
            paths = []
            for anomaly in anomalous_metrics:
                path_scores = self._get_root_cause_paths(root, anomaly, node_scores)
                for nodes, path_score in path_scores:
                    paths.append((path_score, [(node, None) for node in nodes]))
            root_cause_paths[root] = paths

        return RCAResults(root_cause_nodes=root_cause_nodes, root_cause_paths=root_cause_paths)
