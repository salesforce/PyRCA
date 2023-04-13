#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("sfr-pyrca")
except DistributionNotFound:
    __version__ = "Please install PyRCA with setup.py"
else:
    __version__ = dist.version
