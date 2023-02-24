import os
import yaml
import logging
import pandas as pd


class ConfigParser:

    def __init__(self, file_path):
        directory = os.path.dirname(os.path.abspath(__file__))
        if file_path is None:
            file_path = os.path.join(directory, "configs/default.yaml")
        with open(file_path, "r") as f:
            self.config = yaml.safe_load(f)
        if "sigmas" not in self.config["bayesian"]:
            self.config["bayesian"]["sigmas"] = {}

    def get_parameters(self, name):
        return self.config[name]


class RCAEngine:

    def __init__(self, model_dir=None, logger=None):
        self.adjacency_df_filename = "adjacency_df.pkl"
        self.bn_filename = "bn"

        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

    def build_causal_graph(
            self,
            df,
            domain_knowledge_file=None,
            run_pdag2dag=True,
            max_num_points=5000000,
            method_class=None,
            verbose=False,
            **kwargs
    ):
        from pyrca.graphs.causal.fges import FGES

        df = df.iloc[:max_num_points] if max_num_points is not None else df
        if verbose:
            self.logger.info(
                f"The shape of the training data for infer_causal_graph: {df.shape}")
        if domain_knowledge_file is None:
            domain_knowledge_file = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "configs/domain_knowledge.yaml")

        if method_class is None:
            method_class = FGES
        model = method_class(method_class.config_class(
            domain_knowledge_file=domain_knowledge_file,
            run_pdag2dag=run_pdag2dag,
            max_num_points=max_num_points,
            **kwargs))
        adjacency_df = model.train(df)
        adjacency_df.to_pickle(os.path.join(self.model_dir, self.adjacency_df_filename))
        FGES.dump_to_tetrad_json(adjacency_df, self.model_dir)
        return adjacency_df

    def train_bayesian_network(
            self,
            dfs,
            domain_knowledge_file=None,
            config_file=None,
            verbose=False
    ):
        from pyrca.utils.domain import DomainParser
        from pyrca.analyzers.bayesian import BayesianNetwork, BayesianNetworkConfig
        if isinstance(dfs, pd.DataFrame):
            assert dfs.shape[0] > 10000, "The length of df is less than 10000."
            if verbose:
                self.logger.info(
                    f"The training data shape for the Bayesian network: {dfs.shape}")
        else:
            n = sum([df.shape[0] for df in dfs])
            assert n > 10000, "The total length of df is less than 10000."
            if verbose:
                self.logger.info(
                    f"The training data shape for the Bayesian network: {(n, dfs[0].shape[1])}")

        if domain_knowledge_file is None:
            domain_knowledge_file = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "configs/domain_knowledge.yaml")
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
                sigmas=params.get("sigmas", {})
            ))
        bayesian_network.train(dfs=dfs)
        bayesian_network.add_root_causes(domain.get_root_causes())
        bayesian_network.save(self.model_dir, name=self.bn_filename)
        return bayesian_network

    def find_root_causes(self, anomalies, **kwargs):
        """
        Finds the potential root causes via Bayesian inference.

        :param anomalies: A list of anomalous metrics found by any anomaly detector.
        :return: The root cause analysis results.
        """
        from pyrca.analyzers.bayesian import BayesianNetwork
        try:
            model = BayesianNetwork.load(self.model_dir, self.bn_filename)
        except:
            model = BayesianNetwork.load(self.model_dir)
        return model.find_root_causes(anomalies, **kwargs).to_list()
