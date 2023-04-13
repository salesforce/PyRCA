#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The non-gaussian linear causal models (LiNGAM).
"""
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from pyrca.graphs.causal.base import CausalModel, CausalModelConfig


@dataclass
class LiNGAMConfig(CausalModelConfig):
    """
    The configuration class for the LiNGAM algorithm.

    :param domain_knowledge_file: The file path of the domain knowledge file.
    :param run_pdag2dag: Whether to convert a partial DAG to a DAG.
    :param max_num_points: The maximum number of data points in causal discovery.
    :param lower_limit: The lower limit of the causal effect scores for constructing causal graphs.
    :param n_sampling: The number of bootstrapping samples (for bootstrapping only). If ``n_sampling`` > 0,
        bootstrapping will be applied.
    :param min_causal_effect: The threshold for detecting causal direction (for bootstrapping only).
    """

    domain_knowledge_file: str = None
    run_pdag2dag: bool = True
    max_num_points: int = 5000000
    lower_limit: float = 0.1
    n_sampling: int = -1
    min_causal_effect: float = 0.01


class LiNGAM(CausalModel):
    """
    The non-gaussian linear causal models (LiNGAM): https://github.com/cdt15/lingam.
    """

    config_class = LiNGAMConfig

    def __init__(self, config: LiNGAMConfig):
        self.config = config

    def _train(self, df: pd.DataFrame, forbids: List, requires: List, **kwargs):
        import lingam
        from lingam.utils import make_prior_knowledge

        no_paths, paths = [], []
        column_name2idx = {str(name): i for i, name in enumerate(df.columns)}
        if forbids is not None:
            for a, b in forbids:
                no_paths.append((column_name2idx[a], column_name2idx[b]))
        if requires is not None:
            for a, b in requires:
                paths.append((column_name2idx[a], column_name2idx[b]))
        prior = make_prior_knowledge(n_variables=df.shape[1], paths=paths, no_paths=no_paths)

        var_names = df.columns
        model = lingam.DirectLiNGAM(prior_knowledge=prior)
        if self.config.n_sampling <= 0:
            model.fit(df)
            adjacency_mat = (np.abs(model.adjacency_matrix_) >= self.config.lower_limit).astype(int).T
        else:
            result = model.bootstrap(df, n_sampling=self.config.n_sampling)
            prob = result.get_probabilities(min_causal_effect=self.config.min_causal_effect)
            adjacency_mat = (prob >= self.config.lower_limit).astype(int).T

        np.fill_diagonal(adjacency_mat, 0)
        adjacency_df = pd.DataFrame({var_names[i]: adjacency_mat[:, i] for i in range(len(var_names))}, index=var_names)
        return adjacency_df
