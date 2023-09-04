#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The fast greedy equivalence search (FGES) algorithm.
"""
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from pyrca.graphs.causal.base import CausalModel, CausalModelConfig

from pyrca.thirdparty.pytetrad.tools.TetradSearch import TetradSearch


@dataclass
class FGESConfig(CausalModelConfig):
    """
    The configuration class for the FGES algorithm.

    :param domain_knowledge_file: The file path of the domain knowledge file.
    :param max_num_points: The maximum number of data points in causal discovery.
    :param max_degree: The allowed maximum number of parents when searching the graph.
    :param penalty_discount: The penalty discount (a regularization parameter).
    :param score_id: The score function name, e.g., "sem_bic_score".
    """

    domain_knowledge_file: str = None
    max_num_points: int = 5000000
    max_degree: int = 10
    penalty_discount: int = 80
    score_id: str = "sem_bic_score"


class FGES(CausalModel):
    """
    The fast greedy equivalence search (FGES) algorithm for causal discovery.
    """

    config_class = FGESConfig
    causal = None

    def __init__(self, config: FGESConfig):
        self.config = config

    @staticmethod
    def initialize():
        TetradSearch.start_vm()

    @staticmethod
    def finish():
        TetradSearch.stop_vm()

    def _train(self, df: pd.DataFrame, forbids: List, requires: List, start_vm: bool = True, **kwargs):
        var_names, n = df.columns, df.shape[1]

        graph = np.zeros((n, n))

        res = TetradSearch(df)
        res.add_knowledge(forbiddirect=forbids, requiredirect=requires)

        # set score function
        res.use_sem_bic(penalty_discount=self.config.penalty_discount)

        # run fges algorithm
        res.run_fges(max_degree=self.config.max_degree,
                     faithfulness_assumed=True,
                     symmetric_first_step=False,
                     parallelized=False,
                     meek_verbose=False)

        causal_learn_graph = res.get_causal_learn().graph

        for source in range(n):
            for target in range(source+1, n):
                if (causal_learn_graph[source, target] == -1) & (causal_learn_graph[target, source] == 1):
                    graph[source, target] = 1
                elif (causal_learn_graph[source, target] == 1) & (causal_learn_graph[target, source] == -1):
                    graph[target, source] = 1
                elif (causal_learn_graph[source, target] == -1) & (causal_learn_graph[target, source] == -1):
                    graph[source, target] = 1
                    graph[target, source] = 1
                elif (causal_learn_graph[source, target] == 0) & (causal_learn_graph[target, source] == 0):
                    continue
                else:
                    raise ValueError("Unknown direction: source {}, target {}".format(graph[source, target], graph[target, source]))
        if start_vm:
            TetradSearch.stop_vm()

        adjacency_mat = graph.astype(int)
        np.fill_diagonal(adjacency_mat, 0)
        adjacency_df = pd.DataFrame({var_names[i]: adjacency_mat[:, i] for i in range(len(var_names))}, index=var_names)
        return adjacency_df
