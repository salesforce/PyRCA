#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import json
import dash
from dash import html, Input, Output, State, \
    callback, no_update
from ..utils.file_manager import FileManager
from ..pages.utils import create_param_table
from ..pages.causal import create_graph_figure, \
    create_causal_relation_table, create_cycle_table
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
    param_table = create_param_table(height=60)
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "select-causal-method" and algorithm is not None:
        param_info = causal_method.get_parameter_info(algorithm)
        param_table = create_param_table(param_info, height=60)
    return param_table


@callback(
    Output("causal-state", "data"),
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
        State("causal-state", "data"),
        State("cytoscape", "elements")
    ],
    running=[
        (Output("causal-run-btn", "disabled"), True, False),
        (Output("causal-cancel-btn", "disabled"), False, True),
    ],
    cancel=[Input("causal-cancel-btn", "n_clicks")],
    background=True,
    manager=file_manager.get_long_callback_manager(),
    prevent_initial_call=True
)
def click_train_test(
    run_clicks,
    modal_close,
    filename,
    algorithm,
    param_table,
    causal_state,
    cyto_elements
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    state = json.loads(causal_state) \
        if causal_state is not None else {}

    try:
        if ctx.triggered:
            prop_id = ctx.triggered_id
            if prop_id == "causal-run-btn" and run_clicks > 0:
                assert filename, "The data file is empty."
                assert algorithm, "Please select a causal discovery algorithm."

                params = causal_method.parse_parameters(
                    param_info=causal_method.get_parameter_info(algorithm),
                    params={p["Parameter"]: p["Value"] for p in param_table["props"]["data"]},
                )
                df = causal_method.load_data(filename)
                graph, graph_df, relations = causal_method.run(df, algorithm, params)
                causal_levels, cycles = causal_method.causal_order(graph_df)
                state["graph"] = create_graph_figure(graph, causal_levels)
                state["relations"] = relations
                state["cycles"] = cycles

                positions = {}
                if len(cyto_elements) > 0:
                    for element in cyto_elements:
                        if "position" in element:
                            positions[element["data"]["id"]] = element["position"]
                state["positions"] = positions

    except Exception as e:
        modal_is_open = True
        modal_content = str(e)

    return json.dumps(state), modal_is_open, modal_content


@callback(
    Output("cytoscape-hover-output", "children"),
    Input("cytoscape", "mouseoverNodeData"),
)
def hover_graph_node(data):
    if data is None:
        return no_update
    return f"Node ID: {data['id']}"


@callback(
    Output("cytoscape", "elements"),
    Output("causal-relationship-table", "children"),
    Output("causal-cycle-table", "children"),
    Input("causal-state", "data"),
    prevent_initial_call=True
)
def update_view(data):
    state = json.loads(data) \
        if data is not None else {}
    graph = state.get("graph", [])
    positions = state.get("positions", {})

    if state.get("cycles", None) is not None:
        cycle_table = html.Div(children=[
            html.B("Cyclic Paths"),
            html.Hr(),
            create_cycle_table(state["cycles"])
        ])
    else:
        cycle_table = None

    for element in graph:
        if "position" in element:
            element["position"] = \
                positions.get(element["data"]["id"], element["position"])

    return graph, \
        create_causal_relation_table(state.get("relations", None)), \
        cycle_table
