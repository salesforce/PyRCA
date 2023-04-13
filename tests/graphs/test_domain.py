#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import unittest
from pyrca.utils.domain import DomainParser


class TestDomain(unittest.TestCase):
    def test(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        parser = DomainParser(os.path.join(directory, "../data/domain_knowledge.yaml"))

        results = [
            ["A", "B"],
            ["APT", "DB_CPU"],
            ["APT", "APP_GC"],
            ["APP_GC", "DB_CPU"],
            ["APT", "DB_CPU"],
            ["DB_CPU", "APP_GC"],
            ["APT", "APP_GC"],
        ]
        forbids = parser.get_forbid_links(graph_nodes=["DB_CPU", "APP_GC", "APT"])
        self.assertEqual(len(results), len(forbids))
        for a, b in zip(results, forbids):
            self.assertListEqual(a, b)
        self.assertEqual(parser.get_require_links(), None)

        causes = parser.get_root_causes()[0]
        self.assertEqual(causes["name"], "Root_APP_GC")
        self.assertEqual(causes["P(r=1)"], 0.5)
        self.assertEqual(causes["metrics"][0], {"name": "APP_GC", "P(m=0|r=0)": 0.99, "P(m=0|r=1)": 0.01})


if __name__ == "__main__":
    unittest.main()
