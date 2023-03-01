#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import json
import dash
from dash import Input, Output, State, callback
from ..utils.file_manager import FileManager

file_manager = FileManager()


@callback(
    Output("causal-select-file", "options"),
    Output("causal-select-file", "value"),
    [
        Input("causal-upload-data", "filename"),
        Input("causal-upload-data", "contents")
    ],
)
def upload_file(filenames, contents):
    name = None
    if filenames is not None and contents is not None:
        for name, data in zip(filenames, contents):
            file_manager.save_file(name, data)
    options = []
    files = file_manager.uploaded_files()
    for filename in files:
        options.append({"label": filename, "value": filename})
    return options, name

