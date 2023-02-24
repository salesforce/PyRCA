import pprint
from pyrca.applications.WardenAIOps.dataset import load_data
from pyrca.applications.WardenAIOps.rca import RCAEngine


def build_bayesian_network():
    pods = ["eu44", "na115", "na116", "na136", "na211"]

    engine = RCAEngine()
    engine.build_causal_graph(
        df=load_data(pod=pods),
        run_pdag2dag=True,
        max_num_points=5000000,
        verbose=True
    )
    engine.train_bayesian_network(
        dfs=[load_data(pod=pod) for pod in pods]
    )
    result = engine.find_root_causes_bn(
        anomalies=["conn_pool_waits_1p", "apt_no_mq_1p"]
    )
    pprint.pprint(result)


def test_root_causes():
    engine = RCAEngine()
    result = engine.find_root_causes_bn(
        anomalies=["conn_pool_waits_1p", "apt_no_mq_1p"]
    )
    pprint.pprint(result)


if __name__ == "__main__":
    build_bayesian_network()
