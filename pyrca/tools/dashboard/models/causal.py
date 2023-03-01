import os
import sys
import logging
import inspect
import importlib
import pandas as pd

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

    def load_data(self, file_name):
        df = pd.read_csv(os.path.join(self.folder, file_name))
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        self.df = df.set_index("Timestamp")
        return self.df

    @staticmethod
    def get_supported_methods():
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
        return {name: {"class": method, "config_class": config}
                for name, method, config in zip(method_names, method_classes, config_classes)}
