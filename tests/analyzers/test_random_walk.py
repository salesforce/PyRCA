#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
import numpy as np
import pandas as pd
from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig


class TestRandomWalk(unittest.TestCase):
    def setUp(self) -> None:
        columns = ["a", "b", "c", "d"]
        self.graph = pd.DataFrame(
            [[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], columns=columns, index=columns
        )
        np.random.seed(0)
        self.df = pd.DataFrame(np.random.randn(100, 4), columns=columns)

    def test(self):
        model = RandomWalk(config=RandomWalkConfig(graph=self.graph))
        results = model.find_root_causes(anomalous_metrics=["d"], df=self.df).to_list()
        print(results)


if __name__ == "__main__":
    unittest.main()
