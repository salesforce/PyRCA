#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import pprint
import numpy as np
import pandas as pd
from pyrca.applications.example.dataset import load_data
from pyrca.applications.example.rca import RCAEngine


def build_bayesian_network():
    directory = os.path.dirname(os.path.abspath(__file__))
    df = load_data(directory=os.path.join(directory, "../../data"), filename="example.csv")

    engine = RCAEngine()
    engine.build_causal_graph(df=df, run_pdag2dag=True, max_num_points=5000000, verbose=True)
    bn = engine.train_bayesian_network(
        dfs=[
            pd.DataFrame(
                df.values + np.random.randn(*df.shape) * 1e-5,  # To avoid constant values
                columns=df.columns,
                index=df.index,
            )
        ]
    )
    bn.print_probabilities()


def test_root_causes():
    engine = RCAEngine()
    result = engine.find_root_causes_bn(anomalies=["conn_pool", "apt"])
    pprint.pprint(result)


if __name__ == "__main__":
    build_bayesian_network()
    test_root_causes()
