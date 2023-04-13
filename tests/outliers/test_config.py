#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import unittest
from typing import Dict, List
from dataclasses import dataclass, field
from pyrca.base import BaseConfig
from pyrca.outliers.stats import StatsDetectorConfig


@dataclass
class TestConfig(BaseConfig):
    A: float = 1
    B: List = field(default_factory=lambda: [0, 1, 2])
    C: Dict = field(default_factory=lambda: {"a": 1})


class TestBaseConfig(unittest.TestCase):
    def test_dict(self):
        config = TestConfig()
        self.assertDictEqual(config.to_dict(), {"A": 1, "B": [0, 1, 2], "C": {"a": 1}})
        config = TestConfig.from_dict({"A": 2, "B": [1, 2], "C": {"b": 1}})
        self.assertDictEqual(config.to_dict(), {"A": 2, "B": [1, 2], "C": {"b": 1}})

    def test_json(self):
        config = TestConfig()
        self.assertEqual(config.to_json(), '{"A": 1, "B": [0, 1, 2], "C": {"a": 1}}')
        config = TestConfig.from_json('{"A": 2, "B": [1, 2], "C": {"b": 1}}')
        self.assertDictEqual(config.to_dict(), {"A": 2, "B": [1, 2], "C": {"b": 1}})

    def test_yaml(self):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/outlier_configs.yaml")
        config = StatsDetectorConfig.from_yaml(filepath)
        self.assertDictEqual(
            config.to_dict(),
            {
                "default_sigma": 4,
                "thres_win_size": 5,
                "thres_reduce_func": "mean",
                "score_win_size": 3,
                "anomaly_threshold": 0.5,
                "sigmas": None,
                "manual_thresholds": None,
                "custom_win_sizes": None,
                "custom_anomaly_thresholds": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
