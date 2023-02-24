import pprint
from pyrca.applications.WardenAIOps.rca import RCAEngine
from pyrca.applications.WardenAIOps.dataset import load_data
from pyrca.utils.utils import timeseries_window

cases = {
    1: ("na170", "2022-08-29T21:00:00Z", "2022-08-29T22:00:00Z"),
    2: ("cs41", "2022-08-27T03:40:00Z", "2022-08-27T05:00:00Z"),
    3: ("cs122", "2022-08-27T02:00:00Z", "2022-08-27T02:50:00Z"),
    5: ("ap16", "2022-08-26T07:10:00Z", "2022-08-26T08:50:00Z"),
    6: ("cs116", "2022-08-26T04:20:00Z", "2022-08-26T05:20:00Z"),
    8: ("cs212", "2022-08-16T02:00:00Z", "2022-08-16T10:00:00Z"),
    9: ("cs24", "2022-08-16T07:20:00Z", "2022-08-16T16:00:00Z"),
    11: ("na152", "2022-08-10T14:30:00Z", "2022-08-10T19:30:00Z"),
    13: ("cs27", "2022-08-08T14:40:00Z", "2022-08-08T15:30:00Z")
}


def train_bayesian_network(rca):
    pods = ["ap16", "cs41", "cs80", "eu33", "eu36", "na110", "na150"]
    dfs = [load_data(pod=pod) for pod in pods]
    rca.train_bayesian_network(dfs)


def test():
    rca = RCAEngine()
    # Train the Bayesian network parameters
    # The Bayesian network can be fixed after training
    # train_bayesian_network(rca)

    pod, begin_date, end_date = cases[13]
    df = load_data(pod=pod)
    # Estimate the stats parameters of the time series
    # The stats can be updated per month
    rca.train_detector(
        df=df,
        additional_config={
            "manual_thresholds": {
                "Connection_Pool_Errors": {"lower": 0, "upper": 5.0}
            }
        }
    )

    # The inference step
    # Given a window of time series, the algorithm finds potential anomalous metrics and the root causes.
    # If we know some metrics are abnormal, e.g., APT, we can set `anomalies` in `find_root_causes` to ["APT"].
    win_df = timeseries_window(df, begin_date, end_date)
    results = rca.find_root_causes(win_df, detector=None, rca_method="bn")

    print("Anomalous metrics:")
    print(results["anomalous_metrics"])
    print("Anomaly timestamps:")
    print(results["anomaly_timestamps"])
    print("Anomaly info:")
    pprint.pprint(results["anomaly_info"])
    print("Root causes:")
    for causes in results["root_causes"]:
        pprint.pprint(causes)


if __name__ == "__main__":
    test()
