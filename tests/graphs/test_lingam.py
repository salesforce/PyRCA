#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
import numpy as np
import pandas as pd
from pyrca.graphs.causal.lingam import LiNGAM


class TestLINGAM(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        sample_size = 1000
        columns = ["x0", "x1", "x2", "x3", "x4", "x5"]
        x3 = np.random.uniform(size=sample_size)
        x0 = 3.0 * x3 + np.random.uniform(size=sample_size)
        x2 = 6.0 * x3 + np.random.uniform(size=sample_size)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=sample_size)
        x5 = 4.0 * x0 + np.random.uniform(size=sample_size)
        x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=sample_size)
        self.df = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=columns)
        self.graph = pd.DataFrame(
            [
                [0, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            columns=columns,
            index=columns,
        )

    def test(self):
        try:
            model = LiNGAM(LiNGAM.config_class())
            graph = model.train(self.df)
        except ImportError as e:
            print(str(e))
            return
        diff = np.sum(np.abs(graph.values - self.graph.values))
        self.assertLessEqual(diff, 2)


if __name__ == "__main__":
    unittest.main()
