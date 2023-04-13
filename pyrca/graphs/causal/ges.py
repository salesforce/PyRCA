#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The greedy equivalence search (GES) algorithm.
"""
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from pyrca.graphs.causal.base import CausalModel, CausalModelConfig


@dataclass
class GESConfig(CausalModelConfig):
    """
    The configuration class for the GES algorithm.

    :param domain_knowledge_file: The file path of the domain knowledge file.
    :param run_pdag2dag: Whether to convert a partial DAG to a DAG.
    :param max_num_points: The maximum number of data points in causal discovery.
    :param max_degree: The allowed maximum number of parents when searching the graph.
    :param penalty_discount: The penalty discount (a regularization parameter).
    """

    domain_knowledge_file: str = None
    run_pdag2dag: bool = True
    max_num_points: int = 5000000
    max_degree: int = 5
    penalty_discount: int = 100


class GES(CausalModel):
    """
    The greedy equivalence search (GES) algorithm for causal discovery.
    """

    config_class = GESConfig

    def __init__(self, config: GESConfig):
        self.config = config

    def _train(self, df: pd.DataFrame, forbids: List, requires: List, **kwargs):
        from pyrca.thirdparty.causallearn.search.ScoreBased.GES import ges
        from pyrca.thirdparty.causallearn.utils.BackgroundKnowledge import BackgroundKnowledge

        var_names, n = df.columns, df.shape[1]
        graph = np.zeros((n, n))
        column_name2index = {str(name): i for i, name in enumerate(var_names)}

        prior = BackgroundKnowledge()
        if forbids is not None:
            for a, b in forbids:
                i, j = column_name2index[a], column_name2index[b]
                prior.add_forbidden_by_node(f"X{i+1}", f"X{j+1}")
        if requires is not None:
            for a, b in requires:
                i, j = column_name2index[a], column_name2index[b]
                prior.add_required_by_node(f"X{i+1}", f"X{j+1}")

        res = ges(
            df.values,
            maxP=self.config.max_degree,
            parameters={"kfold": 10, "lambda": self.config.penalty_discount},
            background_knowledge=prior,
            verbose=False,
        )
        for edge in res["G"].get_graph_edges():
            edge = str(edge)
            if edge == "":
                continue
            items = edge.split()
            assert len(items) == 3
            a = int(str(items[0]).lower().replace("x", "")) - 1
            b = int(str(items[2]).lower().replace("x", "")) - 1
            if items[1] == "-->":
                graph[a, b] = 1
            elif items[1] == "---":
                graph[a, b] = 1
                graph[b, a] = 1
            else:
                raise ValueError("Unknown direction: {}".format(items[1]))

        adjacency_mat = graph.astype(int)
        np.fill_diagonal(adjacency_mat, 0)
        adjacency_df = pd.DataFrame({var_names[i]: adjacency_mat[:, i] for i in range(len(var_names))}, index=var_names)
        return adjacency_df
