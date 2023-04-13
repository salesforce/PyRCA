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


@dataclass
class FGESConfig(CausalModelConfig):
    """
    The configuration class for the FGES algorithm.

    :param domain_knowledge_file: The file path of the domain knowledge file.
    :param run_pdag2dag: Whether to convert a partial DAG to a DAG.
    :param max_num_points: The maximum number of data points in causal discovery.
    :param max_degree: The allowed maximum number of parents when searching the graph.
    :param penalty_discount: The penalty discount (a regularization parameter).
    :param score_id: The score function name, e.g., "sem-bic-score".
    """

    domain_knowledge_file: str = None
    run_pdag2dag: bool = True
    max_num_points: int = 5000000
    max_degree: int = 10
    penalty_discount: int = 80
    score_id: str = "sem-bic-score"


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
        from pycausal.pycausal import pycausal as pc

        FGES.causal = pc()
        FGES.causal.start_vm()

    @staticmethod
    def finish():
        FGES.causal.stop_vm()

    def _train(self, df: pd.DataFrame, forbids: List, requires: List, start_vm: bool = True, **kwargs):
        from ...utils.misc import is_pycausal_available

        assert is_pycausal_available(), (
            "pycausal is not installed. Please install it from github repo: " "https://github.com/bd2kccd/py-causal."
        )

        from pycausal import search, prior
        from pycausal.pycausal import pycausal as pc

        var_names, n = df.columns, df.shape[1]
        if start_vm and FGES.causal is None:
            causal = pc()
            causal.start_vm()

        column_name2idx = {str(name): i for i, name in enumerate(var_names)}
        graph = np.zeros((n, n))
        prior_knowledge = prior.knowledge(forbiddirect=forbids, requiredirect=requires)

        tetrad = search.tetradrunner()
        tetrad.run(
            algoId="fges",
            dfs=df,
            priorKnowledge=prior_knowledge,
            scoreId=self.config.score_id,
            dataType="continuous",
            maxDegree=self.config.max_degree,
            faithfulnessAssumed=True,
            symmetricFirstStep=False,
            penaltyDiscount=self.config.penalty_discount,
            verbose=False,
        )
        for edge in tetrad.getEdges():
            if edge == "":
                continue
            items = edge.split()
            assert len(items) == 3
            a, b = str(items[0]), str(items[2])
            if items[1] == "-->":
                graph[column_name2idx[a], column_name2idx[b]] = 1
            elif items[1] == "---":
                graph[column_name2idx[a], column_name2idx[b]] = 1
                graph[column_name2idx[b], column_name2idx[a]] = 1
            else:
                raise ValueError("Unknown direction: {}".format(items[1]))
        if start_vm and FGES.causal is None:
            causal.stop_vm()

        adjacency_mat = graph.astype(int)
        np.fill_diagonal(adjacency_mat, 0)
        adjacency_df = pd.DataFrame({var_names[i]: adjacency_mat[:, i] for i in range(len(var_names))}, index=var_names)
        return adjacency_df
