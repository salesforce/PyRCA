#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash
from dash import Input, Output, State, callback
from ..utils.file_manager import FileManager
from ..pages.utils import create_param_table
from ..pages.causal import create_graph_figure, create_causal_relation_table
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
    if prop_id == "select-causal-method" and algorithm is not None:
        param_info = causal_method.get_parameter_info(algorithm)
        param_table = create_param_table(param_info)
    return param_table


@callback(
    Output("cytoscape", "elements"),
    Output("causal-relationship-table", "children"),
    Output("causal-exception-modal", "is_open"),
    Output("causal-exception-modal-content", "children"),
    [
        Input("causal-run-btn", "n_clicks"),
        Input("causal-exception-modal-close", "n_clicks"),
    ],
    [
        State("causal-select-file", "value"),
        State("select-causal-method", "value"),
        State("causal-param-table", "children"),
    ],
    running=[
        (Output("causal-run-btn", "disabled"), True, False),
        (Output("causal-cancel-btn", "disabled"), False, True),
    ],
    cancel=[Input("causal-cancel-btn", "n_clicks")],
    background=True,
    manager=file_manager.get_long_callback_manager(),
)
def click_train_test(
    run_clicks,
    modal_close,
    filename,
    algorithm,
    param_table,
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    graph = None
    relations = None

    try:
        if ctx.triggered:
            prop_id = ctx.triggered_id
            if prop_id == "causal-run-btn" and run_clicks > 0:
                assert filename, "The data file is empty!"
                assert algorithm, "Please select a causal discovery algorithm."

                params = causal_method.parse_parameters(
                    param_info=causal_method.get_parameter_info(algorithm),
                    params={p["Parameter"]: p["Value"] for p in param_table["props"]["data"]},
                )
                df = causal_method.load_data(filename)
                graph, graph_df, relations = causal_method.run(df, algorithm, params)
                causal_levels, cycles = causal_method.causal_order(graph_df)

                import pprint
                pprint.pprint(causal_levels)

    except Exception as e:
        modal_is_open = True
        modal_content = str(e)

    return create_graph_figure(graph), \
           create_causal_relation_table(relations), \
           modal_is_open, modal_content
