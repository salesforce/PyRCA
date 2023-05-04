#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import sys
import json
import yaml
import logging
import inspect
import importlib
import numpy as np
import pandas as pd
import networkx as nx
from enum import Enum
from collections import OrderedDict, defaultdict

from ..utils.log import DashLogger
from pyrca.graphs.causal.base import CausalModel, BaseConfig
from pyrca.utils.domain import DomainParser

dash_logger = DashLogger(stream=sys.stdout)


class CausalDiscovery:
    def __init__(self, folder):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)
        self.folder = folder
        self.supported_methods = None

    def load_data(self, file_name):
        df = pd.read_csv(os.path.join(self.folder, file_name))
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        return df.set_index("Timestamp")

    def get_supported_methods(self):
        if self.supported_methods is not None:
            return self.supported_methods

        method_names = []
        method_classes = []
        config_classes = []
        module = importlib.import_module("pyrca.graphs.causal")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if issubclass(obj, CausalModel):
                    method_names.append(name)
                    method_classes.append(obj)
                elif issubclass(obj, BaseConfig):
                    config_classes.append(obj)

        method_names = sorted(method_names)
        method_classes = sorted(method_classes, key=lambda x: x.__name__)
        config_classes = sorted(config_classes, key=lambda x: x.__name__)
        self.supported_methods = {
            name: {"class": method, "config_class": config}
            for name, method, config in zip(method_names, method_classes, config_classes)
        }
        return self.supported_methods

    @staticmethod
    def get_default_method():
        return "PC"

    def get_parameter_info(self, algorithm):
        config_class = self.supported_methods[algorithm]["config_class"]
        param_info = self._param_info(config_class.__init__)
        return param_info

    @staticmethod
    def _param_info(function):
        def is_enum(t):
            return isinstance(t, type) and issubclass(t, Enum)

        def is_valid_type(t):
            return t in [int, float, str, bool, list, tuple, dict] or is_enum(t)

        param_info = OrderedDict()
        signature = inspect.signature(function).parameters
        for name, param in signature.items():
            if name in ["self", "domain_knowledge_file"]:
                continue
            value = param.default
            if value == param.empty:
                value = ""
            if is_valid_type(type(param.default)):
                value = value.name if isinstance(value, Enum) else value
                param_info[name] = {"type": type(param.default), "default": value}
            elif is_valid_type(param.annotation):
                value = value.name if isinstance(value, Enum) else value
                param_info[name] = {"type": param.annotation, "default": value}
        return param_info

    @staticmethod
    def parse_parameters(param_info, params):
        for key in params.keys():
            assert key in param_info, f"{key} is not in `param_info`."

        kwargs = {}
        for name, value in params.items():
            info = param_info[name]
            value_type = info["type"]
            if value.lower() in ["none", "null"]:
                kwargs[name] = None
            elif value_type in [int, float, str]:
                kwargs[name] = value_type(value)
            elif issubclass(value_type, Enum):
                valid_enum_values = value_type.__members__.keys()
                assert value in valid_enum_values, f"The value of {name} should be in {valid_enum_values}"
                kwargs[name] = value_type[value]
            elif value_type == bool:
                assert value.lower() in ["true", "false"], f"The value of {name} should be either True or False."
                kwargs[name] = value.lower() == "true"
            elif info["type"] in [list, tuple, dict]:
                value = value.replace(" ", "").replace("\t", "")
                value = value.replace("(", "[").replace(")", "]").replace(",]", "]")
                kwargs[name] = json.loads(value)
        return kwargs

    @staticmethod
    def _extract_relations(graph_df):
        relations = {}
        names = list(graph_df.columns)
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                if graph_df.values[i, j] > 0:
                    if graph_df.values[j, i] == 0:
                        relations[(i, j)] = "-->"
                    else:
                        if (j, i) not in relations:
                            relations[(i, j)] = "---"
        relations = {f"{names[i]}<split>{names[j]}": v for (i, j), v in relations.items()}
        return relations

    def run(self, df, algorithm, params, constraints=None, domain_file=None):
        if constraints is None:
            constraints = {}
        df = df.dropna()
        if domain_file:
            domain = DomainParser(os.path.join(self.folder, domain_file))
            metrics = domain.get_metrics()
            if metrics is not None:
                df = df[metrics]

        method_class = self.get_supported_methods()[algorithm]["class"]
        config_class = self.get_supported_methods()[algorithm]["config_class"]
        method = method_class(config_class.from_dict(params))
        graph_df = method.train(
            df=df, forbids=constraints.get("forbidden", []), requires=constraints.get("required", [])
        )
        relations = self._extract_relations(graph_df)
        nx_graph = nx.from_pandas_adjacency(graph_df, create_using=nx.DiGraph())
        return nx_graph, graph_df, relations

    @staticmethod
    def causal_order(graph_df):
        names = list(graph_df.columns)
        cycles = CausalModel.check_cycles(graph_df)
        if len(cycles) > 0:
            return None, cycles

        def _find_root_nodes(graph):
            _roots = []
            for i in range(graph.shape[0]):
                if np.sum(graph[:, i]) == 0:
                    _roots.append(i)
            if len(_roots) == 0:
                _roots.append(0)
            return _roots

        def _assign_level(node, level, graph, level_map):
            level_map[node] = max(level_map[node], level)
            if np.sum(graph[node, :]) == 0:
                return
            for j in range(graph.shape[0]):
                if graph[node, j] == 1:
                    _assign_level(j, level_map[node] + 1, graph, level_map)

        roots = _find_root_nodes(graph_df.values)
        levels = np.zeros(graph_df.shape[0], dtype=int)
        for root in roots:
            _assign_level(root, 0, graph_df.values, levels)

        level_info = defaultdict(list)
        for i, v in enumerate(levels):
            level_info[v].append(names[i])
        return level_info, None

    def dump_results(self, output_dir, graph_df, domain_knowledge):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        graph_df.to_pickle(os.path.join(output_dir, "adjacency_df.pkl"))
        CausalModel.dump_to_tetrad_json(graph_df, output_dir)
        with open(os.path.join(output_dir, "domain_knowledge.yaml"), "w") as outfile:
            yaml.dump(domain_knowledge, outfile, default_flow_style=False)

    def parse_domain_knowledge(self, filename):
        domain = DomainParser(os.path.join(self.folder, filename))
        root_nodes = domain.get_root_nodes()
        leaf_nodes = domain.get_leaf_nodes()
        forbids = domain.get_forbid_links(process_root_leaf=False)
        requires = domain.get_require_links()
        return root_nodes, leaf_nodes, forbids, requires

    def load_graph(self, filename):
        filepath = os.path.join(self.folder, filename)
        if filename.endswith(".pkl"):
            graph_df = pd.read_pickle(filepath)
        elif filename.endswith(".json"):
            graph_df = CausalModel.load_from_tetrad_json(filepath)
        else:
            raise ValueError(f"Unknown file extension for {filename}")
        relations = self._extract_relations(graph_df)
        nx_graph = nx.from_pandas_adjacency(graph_df, create_using=nx.DiGraph())
        return nx_graph, graph_df, relations
