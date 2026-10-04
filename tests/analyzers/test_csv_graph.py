#
# Copyright (c) 2026 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import pandas as pd
import pytest

from pyrca.analyzers.bayesian import BayesianNetwork
from pyrca.analyzers.ht import HT
from pyrca.analyzers.random_walk import RandomWalk


@pytest.mark.parametrize("analyzer", [BayesianNetwork, HT, RandomWalk])
@pytest.mark.parametrize("index_name,write_index", [(None, True), ("metric", True), (None, False)])
def test_analyzer_loads_csv_adjacency_matrix(tmp_path, analyzer, index_name, write_index):
    graph = pd.DataFrame([[0, 1], [0, 0]], index=["cause", "effect"], columns=["cause", "effect"])
    graph.index.name = index_name
    path = tmp_path / "graph.csv"
    graph.to_csv(path, index=write_index)

    model = analyzer(analyzer.config_class(graph=str(path)))
    if analyzer is BayesianNetwork:
        pd.testing.assert_frame_equal(model.graph, graph)
        actual_graph = model.bayesian_model
    else:
        pd.testing.assert_frame_equal(model.adjacency_mat, graph)
        actual_graph = model.graph
    assert set(actual_graph.nodes) == {"cause", "effect"}
    assert set(actual_graph.edges) == {("cause", "effect")}
