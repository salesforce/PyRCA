#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import pytest
import unittest
import pandas as pd
import pickle as pkl
from pyrca.analyzers.psi_pc import PsiPC, PsiPCConfig


class TestEplisonDiagnosis(unittest.TestCase):
    @pytest.mark.skip(reason="pickle issue")
    def test(self):
        # SRC_DIR = '../data/n-10-d-3-an-1-nor-s-1000-an-s-1000/'
        # normal = SRC_DIR + 'normal.csv'
        # anomalous = SRC_DIR + 'anomalous.csv'
        # normal_df = pd.read_csv(normal)
        # anomalous_df = pd.read_csv(anomalous)
        #
        # model = PsiPC(config=PsiPCConfig(verbose=False))
        # results = model.find_root_causes(normal_df, anomalous_df)
        # print(results)

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

        names = [("A%d" % (i + 1)) for i in range(tot_data.shape[1])]
        normal_data = tot_data[:training_samples]
        normal_data_pd = pd.DataFrame(normal_data, columns=names)

        abnormal_data = tot_data[training_samples:]
        abnormal_data_pd = pd.DataFrame(abnormal_data, columns=names)

        model = PsiPC(config=PsiPCConfig(verbose=False))
        results = model.find_root_causes(normal_data_pd, abnormal_data_pd).to_list()
        print(results)


if __name__ == "__main__":
    unittest.main()
