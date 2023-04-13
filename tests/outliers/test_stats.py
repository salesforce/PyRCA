#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
import numpy as np
import pandas as pd
from pyrca.outliers.stats import StatsDetector


class TestStatsDetector(unittest.TestCase):
    def test(self):
        np.random.seed(0)
        x = np.random.randn(50) * 0.1
        y = np.random.randn(50) * 0.1
        y[20:30] = 10

        df = pd.DataFrame(x)
        df.index = pd.to_datetime(df.index * 60, unit="s")
        df.index.rename("timestamp", inplace=True)

        detector = StatsDetector(StatsDetector.config_class())
        detector.train(df)

        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index * 60, unit="s")
        df.index.rename("timestamp", inplace=True)

        results = detector.predict(df).to_dict()
        self.assertListEqual(results["anomalous_metrics"], [0])
        self.assertEqual(results["anomaly_timestamps"][0][0], np.datetime64("1970-01-01T00:20:00.000000000"))

        detector = StatsDetector.from_dict(detector.to_dict())
        results = detector.predict(df).to_dict()
        self.assertListEqual(results["anomalous_metrics"], [0])
        self.assertEqual(results["anomaly_timestamps"][0][0], np.datetime64("1970-01-01T00:20:00.000000000"))

    def test_update_config(self):
        config = {
            "default_sigma": 4.0,
            "thres_win_size": 5,
            "thres_reduce_func": "mean",
            "score_win_size": 3,
            "anomaly_threshold": 0.5,
            "manual_thresholds": {"Connection_Pool_Errors": {"lower": 0.0, "upper": 10.0}},
        }
        detector = StatsDetector(StatsDetector.config_class.from_dict(config))
        detector.update_config({"manual_thresholds": {"Connection_Pool_Errors": {"lower": 1.0, "upper": 9.0}}})
        d = detector.config.to_dict()
        self.assertEqual(d["manual_thresholds"]["Connection_Pool_Errors"]["lower"], 1.0)
        self.assertEqual(d["manual_thresholds"]["Connection_Pool_Errors"]["upper"], 9.0)

    def test_update_bounds(self):
        y = np.ones(50)
        y[20:30] = 10

        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index * 60, unit="s")
        df.index.rename("timestamp", inplace=True)

        detector = StatsDetector(StatsDetector.config_class())
        detector.update_bounds({0: (0, 5)})

        results = detector.predict(df).to_dict()
        self.assertListEqual(results["anomalous_metrics"], [0])
        self.assertEqual(results["anomaly_timestamps"][0][0], np.datetime64("1970-01-01T00:20:00.000000000"))


if __name__ == "__main__":
    unittest.main()
