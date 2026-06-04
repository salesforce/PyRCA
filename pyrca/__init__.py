#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sfr-pyrca")
except PackageNotFoundError:
    __version__ = "Please install PyRCA with setup.py"

