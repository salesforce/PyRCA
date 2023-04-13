#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The PC algorithm.
"""
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from pyrca.graphs.causal.base import CausalModel, CausalModelConfig


@dataclass
class PCConfig(CausalModelConfig):
    """
    The configuration class for the PC algorithm.

    :param domain_knowledge_file: The file path of the domain knowledge file.
    :param run_pdag2dag: Whether to convert a partial DAG to a DAG.
    :param max_num_points: The maximum number of data points in causal discovery.
    :param alpha: The p-value threshold for independent test.
    """

    domain_knowledge_file: str = None
    run_pdag2dag: bool = True
    max_num_points: int = 5000000
    alpha: float = 0.01


class PC(CausalModel):
    """
    The standard PC algorithm.
    """

    config_class = PCConfig

    def __init__(self, config: PCConfig):
        self.config = config

    def _train(self, df: pd.DataFrame, forbids: List, requires: List, **kwargs):
        from pyrca.thirdparty.causallearn.search.ConstraintBased.PC import pc
        from pyrca.thirdparty.causallearn.utils.BackgroundKnowledge import BackgroundKnowledge

        var_names, n = df.columns, df.shape[1]
        graph = np.zeros((n, n))
        column_name2index = {str(name): i for i, name in enumerate(var_names)}

        prior = BackgroundKnowledge()
        if forbids is not None:
            for a, b in forbids:
                prior.add_forbidden_by_node(a, b)
        if requires is not None:
            for a, b in requires:
                prior.add_required_by_node(a, b)

        res = pc(df.values, alpha=self.config.alpha, node_names=list(df.columns), background_knowledge=prior)

        for edge in res.G.get_graph_edges():
            edge = str(edge)
            if edge == "":
                continue
            items = edge.split()
            assert len(items) == 3
            a, b = str(items[0]), str(items[2])
            if items[1] == "-->":
                graph[column_name2index[a], column_name2index[b]] = 1
            elif items[1] == "---":
                graph[column_name2index[a], column_name2index[b]] = 1
                graph[column_name2index[b], column_name2index[a]] = 1
            else:
                raise ValueError("Unknown direction: {}".format(items[1]))

        adjacency_mat = graph.astype(int)
        np.fill_diagonal(adjacency_mat, 0)
        adjacency_df = pd.DataFrame({var_names[i]: adjacency_mat[:, i] for i in range(len(var_names))}, index=var_names)
        return adjacency_df
