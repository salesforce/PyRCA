#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
import numpy as np
import pandas as pd
from pyrca.analyzers.bayesian import BayesianNetwork, BayesianNetworkConfig


class TestBayesianNetwork(unittest.TestCase):
    def setUp(self) -> None:
        columns = ["a", "b", "c", "d"]
        self.graph = pd.DataFrame(
            [[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], columns=columns, index=columns
        )
        np.random.seed(0)
        self.df = pd.DataFrame(np.random.randn(100, 4), columns=columns)

    def test(self):
        model = BayesianNetwork(config=BayesianNetworkConfig(graph=self.graph))
        model.train(self.df)
        results = model.find_root_causes({"d": 1}).to_list()
        self.assertEqual(results[0]["root_cause"], "ROOT_a")


if __name__ == "__main__":
    unittest.main()
