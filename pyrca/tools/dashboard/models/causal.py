import os
import sys
import json
import logging
import inspect
import importlib
import pandas as pd
import networkx as nx
from enum import Enum
from collections import OrderedDict

from ..utils.log import DashLogger
from pyrca.graphs.causal.base import BaseModel, BaseConfig

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
                if issubclass(obj, BaseModel):
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

    def run(self, df, algorithm, params):
        df = df.dropna()
        method_class = self.get_supported_methods()[algorithm]["class"]
        config_class = self.get_supported_methods()[algorithm]["config_class"]
        method = method_class(config_class.from_dict(params))
        graph_df = method.train(df)

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
        relations = {(names[i], names[j]): v for (i, j), v in relations.items()}
        return nx.from_pandas_adjacency(graph_df), relations

    @staticmethod
    def causal_order(graph):
        pass
