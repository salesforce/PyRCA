#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import logging


def get_logger(name, level="INFO"):
    logging.basicConfig(level=level)
    return logging.getLogger(name)
