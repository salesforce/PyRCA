#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import unittest
from pyrca.simulation.data_gen import DAGGenConfig, DataGenConfig, AnomalyDataGenConfig, DAGGen, DataGen, AnomalyDataGen

class Testdatagen(unittest.TestCase):
    def setUp(self):
        dag = DAGGen(DAGGenConfig())
        self.graph = dag.gen()

    def test_dag(self):
        #print(self.graph)
        self.assertEqual(self.graph.shape[0], 20)

    def test_datagen(self):
        try:
            DataGenConfig(dag=self.graph, noise_type='a')
        except AssertionError as error:
            self.assertTrue(len(str(error)) > 0)

        config = DataGenConfig(dag=self.graph, func_type='square', weight_generator='normal')
        self.assertEqual(config.func_type(10), 100)

        data, parent_weights, noise_weights, _, _, = DataGen(config).gen()
        self.assertEqual(data.shape[1], parent_weights.shape[0])

    def test_abnormaldatagen(self):
        config = DataGenConfig(dag=self.graph, func_type='square', weight_generator='normal')
        data, parent_weights, noise_weights, func_type, noise_type = DataGen(config).gen()

        # compute threshold, baseline
        _SLI = 0
        tau = 3
        baseline = data[_SLI, :].mean()
        sli_sigma = data[_SLI, :].std()
        threshold = tau * sli_sigma

        # generate anomaly data
        config = AnomalyDataGenConfig(parent_weights=parent_weights, noise_weights=noise_weights, threshold=threshold,
                                       func_type=func_type, noise_type=noise_type, baseline=baseline, anomaly_type=0)
        anomaly_data, fault = AnomalyDataGen(config).gen()
        self.assertEqual(fault.shape[0], parent_weights.shape[0])


if __name__ == "__main__":
    unittest.main()
