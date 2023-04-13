#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
import pandas as pd
import numpy as np

from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig


class TestEpsilonDiagnosis(unittest.TestCase):
    def gen_random(self, n: int, d: int, covar: float) -> np.ndarray:
        cov_mat = np.ones((d, d)) * covar
        np.fill_diagonal(cov_mat, 1)
        offset = np.zeros(d)
        return np.random.multivariate_normal(offset, cov_mat, size=n)

    def setUp(self) -> None:
        columns = ["a", "b", "c", "d", "e"]
        correlations = [0.1, 0.1, 0.9, 0.1, 0.1]
        self.normal_data = np.zeros((100, 5))
        self.abnormal_data = np.zeros((100, 5))
        for i in range(5):
            col_data = self.gen_random(100, 2, correlations[i])
            self.normal_data[:, i] = col_data[:, 0]
            self.abnormal_data[:, i] = col_data[:, 1]

        self.normal_data = pd.DataFrame(self.normal_data, columns=columns)
        self.abnormal_data = pd.DataFrame(self.abnormal_data, columns=columns)

    def test(self):
        model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(alpha=0.01))
        model.train(self.normal_data)
        results = model.find_root_causes(self.abnormal_data).to_list()
        print(results)


if __name__ == "__main__":
    unittest.main()
