#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import pytest
import unittest
import pandas as pd
import dill as pkl

from pyrca.analyzers.ht import HT, HTConfig


class TestRHT(unittest.TestCase):
    @pytest.mark.skip(reason="pickle issue")
    def test(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(directory, "../data/estimated_dag.pkl"), "rb") as f:
            graph = pkl.load(f)
        path = os.path.join(directory, "../data/synthetic0.pkl")

        # load data and meta configuration
        with open(path, "rb") as input_file:
            data = pkl.load(input_file)

        # get normal and abnormal dataset in pd.DataFrame
        training_samples = data["data"]["num_samples"]
        tot_data = data["data"]["data"]

        names = [("X%d" % (i + 1)) for i in range(tot_data.shape[1])]
        normal_data = tot_data[:training_samples]
        normal_data_pd = pd.DataFrame(normal_data, columns=names)

        abnormal_data = tot_data[training_samples:]
        abnormal_data_pd = pd.DataFrame(abnormal_data, columns=names)

        model = HT(config=HTConfig(graph=graph))
        model.train(normal_data_pd)
        results = model.find_root_causes(abnormal_data_pd, "X1", True).to_list()
        print(results)


if __name__ == "__main__":
    unittest.main()
