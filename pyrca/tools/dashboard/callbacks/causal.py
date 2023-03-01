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
from ..pages.utils import create_param_table
from ..models.causal import CausalDiscovery

file_manager = FileManager()
causal_method = CausalDiscovery(folder=file_manager.data_directory)


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


@callback(
    Output("select-causal-method", "options"),
    Input("select-causal-method-parent", "n_clicks")
)
def update_method_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "select-causal-method-parent":
        methods = sorted(causal_method.get_supported_methods().keys())
        options += [{"label": s, "value": s} for s in methods]
    return options


@callback(
    Output("causal-param-table", "children"),
    Input("select-causal-method", "value")
)
def select_algorithm(algorithm):
    param_table = create_param_table()
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "select-causal-method":
        param_info = causal_method.get_parameter_info(algorithm)
        param_table = create_param_table(param_info)
    return param_table
