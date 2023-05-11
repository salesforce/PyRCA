#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import yaml
import pickle
import logging
import pandas as pd
from typing import Dict, List, Union, Optional
from pyrca.outliers.base import BaseDetector, DetectionResults


class ConfigParser:
    def __init__(self, file_path):
        directory = os.path.dirname(os.path.abspath(__file__))
        if file_path is None:
            file_path = os.path.join(directory, "configs/default.yaml")
        with open(file_path, "r") as f:
            self.config = yaml.safe_load(f)
        if "sigmas" not in self.config["bayesian"]:
            self.config["bayesian"]["sigmas"] = {}
        if "sigmas" not in self.config["stats_detector"]:
            self.config["stats_detector"]["sigmas"] = {}

    def get_parameters(self, name):
        return self.config[name]


class RCAEngine:
    def __init__(self, model_dir=None, logger=None):
        self.adjacency_df_filename = "adjacency_df.pkl"
        self.bn_filename = "bn"

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self._rca_methods = {
            "bn": self._find_root_causes_bn,
            "bayesian": self._find_root_causes_bn,
            "rw": self._find_root_causes_rw,
            "random_walk": self._find_root_causes_rw,
        }

    def build_causal_graph(
            self,
            df,
            domain_knowledge_file=None,
            run_pdag2dag=True,
            max_num_points=5000000,
            method_class=None,
            verbose=False,
            **kwargs,
    ):
        """
        Builds the causal graph via causal discovery methods given the observed time series data.

        :param df: The input time series data in a pandas dataframe.
        :param domain_knowledge_file: The domain knowledge file.
        :param run_pdag2dag: Whether to run the "partial DAG to DAG" function (sometimes the output of
            a causal discovery is a partial DAG instead of DAG).
        :param max_num_points: The maximum number of the training timestamps.
        :param method_class: The class of a causal discovery method, e.g., FGES, PC defined in
            `pyrca.graphs.causal`.
        :param verbose: Whether to print debugging messages.
        """
        from pyrca.graphs.causal.pc import PC

        df = df.iloc[:max_num_points] if max_num_points is not None else df
        if verbose:
            self.logger.info(f"The shape of the training data for infer_causal_graph: {df.shape}")
        if domain_knowledge_file is None:
            domain_knowledge_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "configs/domain_knowledge.yaml"
            )

        if method_class is None:
            method_class = PC
        model = method_class(
            method_class.config_class(
                domain_knowledge_file=domain_knowledge_file,
                run_pdag2dag=run_pdag2dag,
                max_num_points=max_num_points,
                **kwargs,
            )
        )
        adjacency_df = model.train(df)
        adjacency_df.to_pickle(os.path.join(self.model_dir, self.adjacency_df_filename))
        PC.dump_to_tetrad_json(adjacency_df, self.model_dir)
        return adjacency_df

    def train_bayesian_network(
            self,
            dfs,
            domain_knowledge_file=None,
            config_file=None,
            verbose=False
    ):
        """
        Trains the Bayesian network parameters given the causal graph and the time series data.

        :param dfs: The time series data for training (either a pandas dataframe or a list of pandas dataframe).
        :param domain_knowledge_file: The domain knowledge file.
        :param config_file: The configuration file for Bayesian network.
        :param verbose: Whether to print debugging messages.
        """
        from pyrca.utils.domain import DomainParser
        from pyrca.analyzers.bayesian import BayesianNetwork, BayesianNetworkConfig

        if isinstance(dfs, pd.DataFrame):
            assert dfs.shape[0] > 10000, "The length of df is less than 10000."
            if verbose:
                self.logger.info(f"The training data shape for the Bayesian network: {dfs.shape}")
        else:
            n = sum([df.shape[0] for df in dfs])
            assert n > 10000, "The total length of df is less than 10000."
            if verbose:
                self.logger.info(f"The training data shape for the Bayesian network: {(n, dfs[0].shape[1])}")

        if domain_knowledge_file is None:
            domain_knowledge_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "configs/domain_knowledge.yaml"
            )
        domain = DomainParser(domain_knowledge_file)
        config = ConfigParser(config_file)

        graph = pd.read_pickle(os.path.join(self.model_dir, self.adjacency_df_filename))
        params = config.get_parameters("bayesian")
        if verbose:
            self.logger.info(f"The training parameters for the Bayesian network: {params}")

        bayesian_network = BayesianNetwork(
            BayesianNetworkConfig(
                graph=graph,
                default_sigma=params["default_sigma"],
                thres_win_size=params["thres_win_size"],
                thres_reduce_func=params["thres_reduce_func"],
                sigmas=params.get("sigmas", {}),
            )
        )
        bayesian_network.train(dfs=dfs)
        bayesian_network.add_root_causes(domain.get_root_causes())
        bayesian_network.save(self.model_dir, name=self.bn_filename)
        return bayesian_network

    def train_detector(
            self,
            df: Union[pd.DataFrame, Dict],
            config_file: Optional[str] = None,
            use_separate_models=True,
            additional_config: Dict = None,
    ) -> Dict:
        """
        Trains the detector(s) given the time series data.

        :param df: The training dataset which can be a pandas dataframe or a dict with format
            discussed in https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html.
        :param config_file: The configuration file path.
        :param use_separate_models: Whether to train a separate model for each metric.
        :param additional_config: Additional configurations other than the yaml config file.
        :return: The train model for each metric.
        """
        from pyrca.outliers.stats import StatsDetector, StatsDetectorConfig

        if isinstance(df, dict):
            df = pd.DataFrame.from_dict(df)
        assert df.shape[0] > 5000, "The length of df is less than 5000."
        self.logger.info(f"######################The training data shape for the anomaly detector: {df.shape}")

        config = ConfigParser(config_file)
        params = config.get_parameters("stats_detector")
        self.logger.debug(f"The training parameters for the anomaly detector: {params}")

        if use_separate_models:
            trained_model = {}
            for col in df.columns:
                model = StatsDetector(StatsDetectorConfig(**params))
                if additional_config is not None:
                    model.update_config(additional_config)
                model.train(df=df[[col]])
                trained_model[col] = model
                self.logger.debug(f"Model for column {col}")
                self.logger.debug(model.bounds)
        else:
            trained_model = StatsDetector(StatsDetectorConfig(**params))
            if additional_config is not None:
                trained_model.update_config(additional_config)
            trained_model.train(df=df)
            self.logger.info(trained_model.bounds)

        with open(os.path.join(self.model_dir, "stats_detector.pkl"), "wb") as f:
            pickle.dump(trained_model, f)
        self.logger.info("Training anomaly finished ##################################")
        return trained_model

    def _find_root_causes_rw(self, df, anomalies, **kwargs):
        from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig

        graph = pd.read_pickle(os.path.join(self.model_dir, self.adjacency_df_filename))
        model = RandomWalk(RandomWalkConfig(graph=graph))
        return model.find_root_causes(anomalies, df=df, **kwargs)

    def _find_root_causes_bn(self, df, anomalies, **kwargs):
        from pyrca.analyzers.bayesian import BayesianNetwork

        try:
            model = BayesianNetwork.load(self.model_dir, self.bn_filename)
        except:
            model = BayesianNetwork.load(self.model_dir)
        return model.find_root_causes(anomalies, **kwargs)

    def find_root_causes_bn(self, anomalies: List, **kwargs):
        """
        Finds the potential root causes via Bayesian inference.

        :param anomalies: A list of anomalous metrics found by any anomaly detector.
        :return: The root cause analysis results.
        """
        return self._find_root_causes_bn(None, anomalies, **kwargs).to_list()

    def find_root_causes(
            self,
            df: Union[pd.DataFrame, Dict],
            detector: Union[Dict, BaseDetector],
            rca_method: Optional[str] = None,
            known_anomalies: List = None,
            **kwargs,
    ):
        """
        Finds the potential root causes given an incident window.

        :param df: The training dataset which can be a pandas dataframe or a dict with format
            discussed in https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html.
        :param detector: The anomaly detector (e.g., StatsDetector). It can either be a single model
            for all metrics or a dict with format `{metric_1: model_1, metric_2: model_2, ...}`.
        :param rca_method: The selected RCA method. If it is None, no RCA methods such as Bayesian inference
            or random walk will run.
        :param known_anomalies: The known anomalies based on domain knowledge or other anomaly alerts.
        :return: The root cause analysis results.
        """
        if detector is None:
            with open(os.path.join(self.model_dir, "stats_detector.pkl"), "rb") as f:
                detector = pickle.load(f)
        if isinstance(df, dict):
            df = pd.DataFrame.from_dict(df)

        if isinstance(detector, BaseDetector):
            anomaly_info = detector.predict(df).to_dict()
        else:
            results = []
            for col in df.columns:
                assert col in detector, f"The detector for metric {col} is not specified."
                results.append(detector[col].predict(df[[col]]))
            anomaly_info = DetectionResults.merge(results).to_dict()

        if rca_method is not None:
            anomalies = (
                anomaly_info["anomalous_metrics"]
                if known_anomalies is None
                else list(set(anomaly_info["anomalous_metrics"] + known_anomalies))
            )
            anomaly_info["root_causes"] = self._rca_methods[rca_method](df=df, anomalies=anomalies, **kwargs)
        return anomaly_info
