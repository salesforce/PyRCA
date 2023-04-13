#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
Base class for all Models.
"""
import os
import json
import yaml
import dill
from copy import deepcopy
from typing import Dict
from inspect import signature
from dataclasses import dataclass, asdict
from pyrca.utils.misc import AutodocABCMeta


@dataclass
class BaseConfig(metaclass=AutodocABCMeta):
    """
    Base class for all model configurations.
    """

    class _DefaultJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, BaseConfig):
                return o.to_dict()
            return super().default(o)

    def to_dict(self) -> Dict:
        """
        Converts the config into a dict.
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Converts the config into a JSON string.
        """
        return json.dumps(self, cls=BaseConfig._DefaultJSONEncoder)

    @classmethod
    def from_dict(cls, d: Dict):
        """
        Loads a config from a dict.

        :param d: The parameters in a dict.
        """
        config = cls()
        for key, value in d.items():
            assert key in config.__dict__, f"Class {cls.__name__} has no field named {key}."
            setattr(config, key, value)
        return config

    @classmethod
    def from_json(cls, s: str):
        """
        Loads a config from a JSON string.

        :param s: The parameters in a JSON string.
        """
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_yaml(cls, filepath: str):
        """
        Loads a config from a YAML file.

        :param filepath: The YAML file path.
        """
        with open(filepath, "r") as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(d)


class BaseModel(metaclass=AutodocABCMeta):
    """
    Base Class for all Models.
    """

    def _get_init_params(self):
        """
        Get the init params for the implementation outlier algorithm.

        :return: dict containing all init params.
        """
        if "config" in self.__dict__:
            return self.config.to_dict()
        else:
            param_values = {}
            init = getattr(self.__init__, "deprecated_original", self.__init__)
            if init is object.__init__:
                # No explicit constructor to introspect
                return []
            init_signature = signature(init)
            # Consider the constructor parameters excluding 'self'
            for p in init_signature.parameters.values():
                if p.name != "self" and p.kind != p.VAR_KEYWORD:
                    param_values[p.name] = {"default": p.default, "actual": self.__dict__[p.name]}
            return param_values

    def __getstate__(self):
        return {k: deepcopy(v) for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def save(self, directory: str, filename: str = None, **kwargs):
        """
        Saves the initialized model.

        :param directory: The folder for the dumped explainer.
        :param filename: The filename (the model class name if it is None).
        """
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f"{type(self).__name__}.pkl"
        state = self.__getstate__()
        if "ignored_attributes" in kwargs:
            for attr in kwargs["ignored_attributes"]:
                state.pop(attr, None)
        with open(os.path.join(directory, filename), "wb") as f:
            dill.dump(state, f)

    @classmethod
    def load(cls, directory: str, filename: str = None, **kwargs):
        """
        Loads the dumped model.

        :param directory: The folder for the dumped model.
        :param filename: The filename (the model class name if it is None).
        """
        if filename is None:
            filename = f"{cls.__name__}.pkl"
        with open(os.path.join(directory, filename), "rb") as f:
            state = dill.load(f)
        self = super(BaseModel, cls).__new__(cls)
        self.__setstate__(state)
        return self
