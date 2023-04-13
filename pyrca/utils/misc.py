#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import sys
import importlib.util
from abc import ABCMeta
from packaging import version

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


class AutodocABCMeta(ABCMeta):
    """
    Metaclass used to ensure that inherited members of an abstract base class
    also inherit docstrings for inherited methods.
    """

    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if member.__doc__ is None:
                for base in bases[::-1]:
                    attr = getattr(base, name, None)
                    if attr is not None:
                        member.__doc__ = attr.__doc__
                        break
        return cls


def is_pycausal_available():
    """
    Checks if the `pycausal` library is installed.
    """
    if importlib.util.find_spec("pycausal") is not None:
        _version = importlib_metadata.version("pycausal")
        if version.parse(_version) != version.parse("1.1.1"):
            raise EnvironmentError(f"pycausal found but with version {_version}. " f"The require version is 1.1.1.")
        return True
    else:
        return False
