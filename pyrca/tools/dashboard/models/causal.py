import os
import sys
import logging
import inspect
import importlib
import pandas as pd
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

        self.df = None
        self.supported_methods = None

    def load_data(self, file_name):
        df = pd.read_csv(os.path.join(self.folder, file_name))
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        self.df = df.set_index("Timestamp")
        return self.df

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
