#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The RCA method based on Bayesian inference.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union, List
from collections import deque
from dataclasses import dataclass

from pyrca.thirdparty.pgmpy.models import BayesianModel
from pyrca.thirdparty.pgmpy.estimators import MaximumLikelihoodEstimator
from pyrca.thirdparty.pgmpy.inference import VariableElimination
from pyrca.thirdparty.pgmpy.factors.discrete import TabularCPD
from pyrca.thirdparty.pgmpy.readwrite import BIFWriter, BIFReader

from pyrca.utils.utils import estimate_thresholds
from pyrca.base import BaseConfig, BaseModel
from pyrca.analyzers.base import BaseRCA, RCAResults


@dataclass
class BayesianNetworkConfig(BaseConfig):
    """
    The configuration class for the Bayesian network based RCA.

    :param graph: The adjacency matrix of the causal graph,
        which can be a pandas dataframe or a file path of a CSV file or a pickled file.
    :param sigmas: Specific sigmas other than ``default_sigma`` for certain variables. This parameter
        is for constructing training data (only used when ``detector`` in the ``train`` function is not set).
    :param default_sigma: The default sigma value for computing the detection. This parameter is
        for constructing training data (only used when ``detector`` in the ``train`` function is not set).
    :param thres_win_size: The size of the smoothing window for computing the detection threshold.
        This parameter is for constructing training data (only used when ``detector`` in the ``train``
        function is not set).
    :param thres_reduce_func: The reduction function for threshold, i.e., "mean" uses the mean value
        and standard deviation, "median" uses the median value and median absolute deviation. This parameter
        is for constructing training data (only used when ``detector`` in the ``train`` function is not set).
    :param infer_method: Use "posterior" or "likelihood" when doing Bayesian inference.
    :param root_cause_top_k: The maximum number of root causes in the results.
    """

    graph: Union[pd.DataFrame, str]
    sigmas: Dict = None
    default_sigma: float = 4.0
    thres_win_size: int = 5
    thres_reduce_func: str = "mean"
    infer_method: str = "posterior"
    root_cause_top_k: int = 3


class BayesianNetwork(BaseRCA):
    """
    The RCA method based on Bayesian inference.
    """

    config_class = BayesianNetworkConfig

    def __init__(self, config: BayesianNetworkConfig):
        super().__init__()
        self.config = config
        if isinstance(config.graph, str):
            if config.graph.endswith(".csv"):
                self.graph = pd.read_csv(config.graph)
            elif config.graph.endswith(".pkl"):
                with open(config.graph, "rb") as f:
                    self.graph = pickle.load(f)
            else:
                raise RuntimeError("The graph file format is not supported, " "please choose a csv or pickle file.")
        else:
            self.graph = config.graph
        self.bayesian_model = self._build_bayesian_network(self.graph)
        self.root_nodes = []

    @staticmethod
    def _build_bayesian_network(graph):
        if graph is not None:
            edges = []
            adj_mat = graph.values
            node_names = graph.columns
            node2index = {c: i for i, c in enumerate(graph.columns)}
            for node in graph.columns:
                parents = [node_names[p] for p, v in enumerate(adj_mat[:, node2index[node]]) if v > 0]
                for p in parents:
                    edges.append((p, node))
            return BayesianModel(ebunch=edges)
        return None

    def train(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]], detector: BaseModel = None, **kwargs):
        """
        Estimates Bayesian network parameters given the training time series.

        :param dfs: One or multiple training time series.
        :param detector: The detector used to construct the training dataset, i.e., the training dataset
            includes the anomaly labels detected by ``detector``. If ``detector`` is None, a default
            stats-based detector will be applied.
        """
        assert self.graph is not None, "The graph is not set."
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        if detector is None:
            sigmas = {} if self.config.sigmas is None else self.config.sigmas
            all_scores = []
            for df in dfs:
                lowers, uppers = estimate_thresholds(
                    df=df,
                    sigmas=sigmas,
                    default_sigma=self.config.default_sigma,
                    win_size=self.config.thres_win_size,
                    reduce=self.config.thres_reduce_func,
                )
                scores = (df.values > uppers).astype(int) + (df.values < lowers).astype(int)
                all_scores.append(scores)
            all_scores = np.concatenate(all_scores, axis=0)
            score_df = pd.DataFrame(all_scores, columns=dfs[0].columns)
        else:
            all_scores = []
            for df in dfs:
                all_scores.append(detector.predict(df)["anomaly_labels"])
            score_df = pd.concat(all_scores, axis=0)

        ignored_columns = [c for c in score_df.columns if c not in self.graph.columns]
        score_df = score_df.drop(columns=ignored_columns)
        self.bayesian_model.fit(data=score_df, estimator=MaximumLikelihoodEstimator)
        self._refine_parameters(lower_bound=0.01)

    def _refine_parameters(self, lower_bound):
        # Set lower and upper bounds
        for node in self.bayesian_model.nodes():
            cpd = self.bayesian_model.get_cpds(node)
            assert cpd.values.shape[0] == 2, f"The cardinality of the variable {cpd.variable} should be = 2."
            cpd.values = np.clip(cpd.values, lower_bound, 1.0 - lower_bound)

        # Make sure that P(m=1|Sa) >= P(m=1|Sb) when Sa >= Sb
        def _refine(_index, _values, _num_vars, _mem, _inc=0.1):
            if _index in _mem:
                return _mem[_index]
            _v = _values[1, _index]
            for i in range(_num_vars):
                _x = (~(1 << i)) & _index
                if _x != _index:
                    _v = max(_v, _refine(_x, _values, _num_vars, _mem) + _inc)
            _v = min(_v, 1.0 - lower_bound)
            _mem[_index] = _v
            return _v

        for node in self.bayesian_model.nodes():
            cpd = self.bayesian_model.get_cpds(node)
            values = cpd.values.reshape((2, -1))
            if values.shape[1] > 2:
                num_vars, mem = len(cpd.variables) - 1, {}
                _refine(2**num_vars - 1, values, num_vars, mem)
                new_values = np.zeros_like(values)
                new_values[1, :] = [mem[k] for k in range(values.shape[1])]
                new_values[0, :] = 1.0 - new_values[1, :]
                self.bayesian_model.add_cpds(
                    TabularCPD(
                        variable=cpd.variable,
                        variable_card=2,
                        values=new_values,
                        evidence=cpd.variables[1:],
                        evidence_card=cpd.cardinality[1:],
                    )
                )

    def _infer(self, variables, evidence):
        model_infer = VariableElimination(self.bayesian_model)
        return model_infer.query(variables=variables, evidence=evidence)

    def _compute_joint(self, variables, evidence):
        model_infer = VariableElimination(self.bayesian_model)
        res = model_infer.query(variables=list(variables.keys()), evidence=evidence, joint=True)
        return res.get_value(**variables)

    def _add_root_cause(self, root_cause_name, metric_name, root_cause_probs, root_prob):
        """
        Adds a root cause node into the graph.

        :param root_cause_name: the name of the root cause, e.g., IO
        :param metric_name: the name of the first affected metric, e.g., USIO
        :param root_cause_probs: [P(metric=0 | root=0), P(metric=0 | root=1)]
        :param root_prob: P(root=1)
        """
        assert len(
            root_cause_probs
        ), "root_cause_probs should contain two values: P(metric=0 | root=0), P(metric=0 | root=1)"
        if metric_name not in self.bayesian_model.nodes():
            print(f"WARNING: Metric {metric_name} is not in the Bayesian network.")
            self.bayesian_model.add_node(metric_name)
        if root_cause_name not in self.bayesian_model.nodes():
            self.bayesian_model.add_node(root_cause_name)
            self.root_nodes.append(root_cause_name)
        self.bayesian_model.add_edge(root_cause_name, metric_name)
        self.bayesian_model.add_cpds(
            TabularCPD(variable=root_cause_name, variable_card=2, values=[[1 - root_prob], [root_prob]])
        )

        cpd = self.bayesian_model.get_cpds(metric_name)
        if cpd is None or cpd.values.size == 2:
            self.bayesian_model.add_cpds(
                TabularCPD(
                    variable=metric_name,
                    variable_card=2,
                    values=[root_cause_probs, [1 - root_cause_probs[0], 1 - root_cause_probs[1]]],
                    evidence=[root_cause_name],
                    evidence_card=[2],
                )
            )
        else:
            v = cpd.values.reshape((2, -1))
            u = np.zeros(v.shape, dtype=float)
            u[0, :] = root_cause_probs[1]
            u[1, :] = 1 - root_cause_probs[1]
            evidence = [root_cause_name] + cpd.variables[1:]
            self.bayesian_model.add_cpds(
                TabularCPD(
                    variable=metric_name,
                    variable_card=2,
                    values=np.concatenate([v, u], axis=1),
                    evidence=evidence,
                    evidence_card=[2] * len(evidence),
                )
            )

    def add_root_causes(self, root_causes: List):
        """
        Adds additional root cause nodes into the graph based on domain knowledge.

        :param root_causes: A list of root causes.
        """
        if root_causes is None:
            return
        for r in root_causes:
            for metric in r["metrics"]:
                self._add_root_cause(
                    root_cause_name=r["name"],
                    metric_name=metric["name"],
                    root_cause_probs=[metric.get("P(m=0|r=0)", 0.99), metric.get("P(m=0|r=1)", 0.01)],
                    root_prob=r["P(r=1)"],
                )

    def update_probability(self, target_node: str, parent_nodes: List, prob: float):
        """
        Updates the Bayesian network parameters. For example, if we want to set `P(A=1 | B=0, C=1, D=1) = p`,
        we set `target_node = A`, `parent_nodes = [C, D]` and `prob = p`.

        :param target_node: The child/effect node to be modified.
        :param parent_nodes: The parent nodes whose values are ones.
        :param prob: The probability that the value of target node is one given these parent nodes.
        """
        cpd = self.bayesian_model.get_cpds(target_node)
        variables = cpd.variables[1:]
        values_0, values_1 = cpd.values[0], cpd.values[1]
        for v in variables[:-1]:
            idx = int(v in parent_nodes)
            values_0 = values_0[idx]
            values_1 = values_1[idx]
        idx = int(variables[-1] in parent_nodes)
        values_0[idx] = 1 - prob
        values_1[idx] = prob
        self.bayesian_model.add_cpds(cpd)

    def _get_root_cause_score(self, root_node, evidence, mode):
        if mode == "posterior":
            return self._infer(variables=[root_node], evidence=evidence).values[1]
        elif mode == "likelihood":
            return self._compute_joint(variables=evidence, evidence={root_node: 1})
        else:
            raise NotImplementedError

    def _get_paths(self, node, root=True):
        q, paths = deque([(node, [])]), []
        while q:
            node, path = q.popleft()
            nodes = self.bayesian_model.get_parents(node) if root else self.bayesian_model.get_children(node)
            path.append(node)
            if len(nodes) == 0:
                paths.append(path[::-1] if root else path)
            else:
                for v in nodes:
                    q.append((v, path[:]))
        return paths

    def _get_all_paths(self, node):
        root_paths = self._get_paths(node, root=True)
        leaf_paths = self._get_paths(node, root=False)
        all_paths = [p + q[1:] for p in root_paths for q in leaf_paths]

        paths, flags = [], {}
        for path in all_paths:
            p = "_".join(path)
            if p not in flags:
                paths.append(path)
                flags[p] = True
        return paths

    def _get_path_root_cause_scores(self, paths, evidence, node_scores, overwrite_scores=None):
        if overwrite_scores is None:
            overwrite_scores = {}
        for path in paths:
            for node in path:
                if node in node_scores:
                    continue
                if node in evidence:
                    prob = evidence[node]
                elif node in overwrite_scores:
                    prob = overwrite_scores[node]
                else:
                    prob = self._infer(variables=[node], evidence=evidence).values[1]
                node_scores[node] = prob

        score_paths = []
        for path in paths:
            score = np.mean([node_scores[node] for node in path])
            path_scores = [(node, node_scores[node]) for node in path]
            score_paths.append((score, path_scores))
        score_paths = sorted(score_paths, key=lambda x: x[0], reverse=True)
        return score_paths

    def _argument_root_nodes(self):
        existing_roots = [str(node).replace("ROOT_", "") for node in self.root_nodes]

        nodes = []
        for i, values in enumerate(self.graph.values.T):
            if np.sum(values) == 0:
                name = str(self.graph.columns[i])
                if name not in existing_roots:
                    nodes.append(self.graph.columns[i])

        if len(nodes) > 0:
            root_nodes = [
                {
                    "name": f"ROOT_{node}",
                    "P(r=1)": 0.5,
                    "metrics": [{"name": node, "P(m=0|r=0)": 0.99, "P(m=0|r=1)": 0.01}],
                }
                for node in nodes
            ]
            self.add_root_causes(root_nodes)

    def _post_process(self, all_paths):
        paths, flags = [], {}
        for path_score, path in all_paths:
            filtered_path = []
            for node, node_score in path:
                if node_score > 0:
                    filtered_path.append((node, node_score))
            signature = "->".join(str(node) for node, _ in filtered_path)
            if signature in flags:
                continue
            flags[signature] = True
            paths.append((path_score, filtered_path))
        return paths

    def find_root_causes(
        self,
        anomalous_metrics: Union[List, Dict],
        set_zero_path_score_for_normal_metrics: bool = False,
        remove_zero_score_node_in_path: bool = True,
        **kwargs,
    ) -> List:
        """
        Finds the root causes given the observed anomalous metrics.

        :param anomalous_metrics: A list of anomalous metrics. ``anomalous_metrics`` is either a list
            ['metric_A', 'metric_B', ...] or a dict {'metric_A': 1, 'metric_B': 1}.
        :param set_zero_path_score_for_normal_metrics: Whether to set the scores of normal metrics
            (metrics that are not in ``anomalous_metrics``) to zeros when computing root cause path scores.
        :param remove_zero_score_node_in_path: Whether to remove the nodes with zero scores from the paths.
        :return: A list of the found root causes.
        """
        self._argument_root_nodes()

        if isinstance(anomalous_metrics, Dict):
            evidence = {metric: v for metric, v in anomalous_metrics.items() if metric in self.bayesian_model.nodes()}
        else:
            evidence = {metric: 1 for metric in anomalous_metrics if metric in self.bayesian_model.nodes()}

        # Pick the paths which contain anomalous node
        valid_paths = {}
        for root in self.root_nodes:
            try:
                paths = self._get_all_paths(node=root)
                for path in paths:
                    for node in path:
                        if evidence.get(node, 0) == 1:
                            if root not in valid_paths:
                                valid_paths[root] = []
                            valid_paths[root].append(path)
                            break
            except Exception as e:
                print(e)

        overwrite_scores = {}
        if set_zero_path_score_for_normal_metrics:
            for node in self.bayesian_model.nodes():
                if node not in anomalous_metrics and node not in self.root_nodes:
                    overwrite_scores[node] = 0

        # Compute the root cause scores
        root_scores = []
        for root in valid_paths.keys():
            score = self._get_root_cause_score(root, evidence, mode=self.config.infer_method)
            root_scores.append((root, score))
        root_scores = sorted(root_scores, key=lambda x: x[1], reverse=True)

        results, node_scores = [], {}
        for root, score in root_scores:
            res = {"root_cause": root, "score": score, "paths": []}
            paths = valid_paths[root]
            res["paths"] = self._get_path_root_cause_scores(paths, evidence, node_scores)[
                : self.config.root_cause_top_k
            ]
            results.append(res)
        results = sorted(results, key=lambda r: (r["score"], r["paths"][0][0]), reverse=True)

        root_cause_nodes = []
        root_cause_paths = {}
        for entry in results:
            root_cause_nodes.append((entry["root_cause"], entry["score"]))
            root_cause_paths[entry["root_cause"]] = entry["paths"]
        return RCAResults(root_cause_nodes=root_cause_nodes, root_cause_paths=root_cause_paths)

    def save(self, directory, filename="bn", **kwargs):
        writer = BIFWriter(self.bayesian_model)
        writer.write_bif(os.path.join(directory, f"{filename}.bif"))
        state = self.__getstate__()
        state.pop("bayesian_model", None)
        with open(os.path.join(directory, f"{filename}_info.pkl"), "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, directory, filename="bn", **kwargs):
        model = cls.__new__(cls)
        reader = BIFReader(os.path.join(directory, f"{filename}.bif"))
        model.bayesian_model = reader.get_model()
        with open(os.path.join(directory, f"{filename}_info.pkl"), "rb") as f:
            model.__setstate__(pickle.load(f))
        return model

    def print_probabilities(self):
        for node in self.bayesian_model.nodes():
            print(self.bayesian_model.get_cpds(node))
