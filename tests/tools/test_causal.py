#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
import pandas as pd
from pyrca.tools.dashboard.models.causal import CausalDiscovery


class TestCausal(unittest.TestCase):
    def test_1(self):
        graph = pd.DataFrame(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            columns=["a", "b", "c", "d"],
            index=["a", "b", "c", "d"],
        )
        levels, cycles = CausalDiscovery.causal_order(graph)
        self.assertEqual(levels, None)
        self.assertSetEqual(set(cycles[0]), set(["d", "c"]))

    def test_2(self):
        graph = pd.DataFrame(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            columns=["a", "b", "c", "d"],
            index=["a", "b", "c", "d"],
        )
        levels, cycles = CausalDiscovery.causal_order(graph)
        self.assertEqual(cycles, None)
        self.assertSetEqual(set(levels[0]), set(["a", "c"]))
        self.assertSetEqual(set(levels[1]), set(["b", "d"]))

    def test_3(self):
        graph = pd.DataFrame(
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
            columns=["a", "b", "c", "d", "e"],
            index=["a", "b", "c", "d", "e"],
        )
        levels, cycles = CausalDiscovery.causal_order(graph)
        self.assertEqual(levels, None)
        self.assertSetEqual(set(cycles[0]), set(["d", "c", "e"]))


if __name__ == "__main__":
    unittest.main()
