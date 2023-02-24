import pprint
import numpy as np
import pandas as pd
from pyrca.applications.SDDB.dataset import load_data
from pyrca.applications.SDDB.rca import RCAEngine


def build_bayesian_network():
    directory = "/home/ywz/data/timeseries/SDDB_new"
    df, _, _ = load_data(directory, date_range=("2022-10-01", "2022-10-31"))
    print(df.columns)
    print(df.shape)

    engine = RCAEngine()
    engine.build_causal_graph(
        df=df,
        run_pdag2dag=True,
        max_num_points=5000000,
        penalty_discount=150,
        verbose=True
    )
    df = pd.DataFrame(
        df.values + np.random.randn(*df.shape) * 1e-4,
        columns=df.columns,
        index=df.index
    )
    bn = engine.train_bayesian_network(
        dfs=df
    )
    for node in bn.bayesian_model.nodes():
        cpd = bn.bayesian_model.get_cpds(node)
        print(cpd)


def test_update_probablity():
    from pyrca.analyzers.bayesian import BayesianNetwork

    bayesian_network = BayesianNetwork.load(
        directory="/home/ywz/Dropbox/Program/Salesforce/SDDB/pyrca/applications/SDDB/models"
    )
    bayesian_network.update_probablity(
        target_node="ACT",
        parent_nodes=["AWT", "AvgSL"],
        prob=0.9
    )


def test_root_causes():
    engine = RCAEngine()

    evidence = {
        'USIO': 1,
        'IOWT': 1,
        'AvgSL': 1,
        'DBt': 1,
        'ACT': 1,
        'NWT': 1,
        'COMT': 1,
        'APPL': 1,
        'BGB': 1
    }
    result = engine.find_root_causes(evidence)
    pprint.pprint(result)


if __name__ == "__main__":
    test_root_causes()
