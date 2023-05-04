#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import os
import json
import dash
from dash import html, Input, Output, State, callback, no_update, dcc
from ..utils.file_manager import FileManager
from ..pages.utils import create_param_table
from ..pages.causal import (
    create_graph_figure,
    create_causal_relation_table,
    create_cycle_table,
    create_root_leaf_table,
    create_link_table,
)
from ..models.causal import CausalDiscovery

file_manager = FileManager()
causal_method = CausalDiscovery(folder=file_manager.data_directory)


@callback(
    Output("causal-select-file", "options"),
    Output("select-domain", "options"),
    [Input("causal-upload-data", "filename"), Input("causal-upload-data", "contents")],
)
def upload_file(filenames, contents):
    if filenames is not None and contents is not None:
        for name, data in zip(filenames, contents):
            file_manager.save_file(name, data)
    file_options, domain_options = [], []
    files = file_manager.uploaded_files()
    for filename in files:
        if filename.endswith(".csv"):
            file_options.append({"label": filename, "value": filename})
        elif filename.endswith(".yml") or filename.endswith(".yaml"):
            domain_options.append({"label": filename, "value": filename})
    return file_options, domain_options


@callback(Output("select-causal-method", "options"), Input("select-causal-method-parent", "n_clicks"))
def update_method_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "select-causal-method-parent":
        methods = sorted(causal_method.get_supported_methods().keys())
        options += [{"label": s, "value": s} for s in methods]
    return options


@callback(Output("causal-param-table", "children"), Input("select-causal-method", "value"))
def select_algorithm(algorithm):
    param_table = create_param_table(height=80)
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "select-causal-method" and algorithm is not None:
        param_info = causal_method.get_parameter_info(algorithm)
        param_table = create_param_table(param_info, height=80)
    return param_table


def _build_constraints(metrics, root_leaf_table, link_table):
    forbids, requires = [], []
    for p in root_leaf_table["props"]["data"]:
        node = p["Metric"]
        if node:
            if p["Type"] == "root":
                for metric in metrics:
                    if metric != node:
                        forbids.append([metric, node])
            else:
                for metric in metrics:
                    if metric != node:
                        forbids.append([node, metric])

    for p in link_table["props"]["data"]:
        if p["Node A"] and p["Node B"]:
            if p["Type"] == "⇒":
                requires.append([p["Node A"], p["Node B"]])
            else:
                forbids.append([p["Node A"], p["Node B"]])

    constraints = {"forbidden": forbids, "required": requires}
    return constraints


def _dump_results(output_folder, graph_df, root_leaf_table, link_table):
    roots, leaves = [], []
    forbids, requires = [], []
    for p in root_leaf_table["props"]["data"]:
        node = p["Metric"]
        if node:
            if p["Type"] == "root":
                roots.append(node)
            else:
                leaves.append(node)
    for p in link_table["props"]["data"]:
        if p["Node A"] and p["Node B"]:
            if p["Type"] == "⇒":
                requires.append([p["Node A"], p["Node B"]])
            else:
                forbids.append([p["Node A"], p["Node B"]])

    domain_knowledge = {
        "causal-graph": {"root-nodes": roots, "leaf-nodes": leaves, "forbids": forbids, "requires": requires}
    }
    causal_method.dump_results(output_folder, graph_df, domain_knowledge)


@callback(
    Output("causal-state", "data"),
    Output("causal-data-state", "data"),
    Output("causal-exception-modal", "is_open"),
    Output("causal-exception-modal-content", "children"),
    [
        Input("causal-run-btn", "n_clicks"),
        Input("causal-exception-modal-close", "n_clicks"),
        Input("upload-graph", "filename"),
        Input("upload-graph", "contents"),
    ],
    [
        State("causal-select-file", "value"),
        State("select-causal-method", "value"),
        State("causal-param-table", "children"),
        State("causal-state", "data"),
        State("causal-data-state", "data"),
        State("cytoscape", "elements"),
        State("root-leaf-table", "children"),
        State("link-table", "children"),
        State("select-domain", "value"),
    ],
    running=[
        (Output("causal-run-btn", "disabled"), True, False),
        (Output("causal-cancel-btn", "disabled"), False, True),
    ],
    cancel=[Input("causal-cancel-btn", "n_clicks")],
    background=True,
    manager=file_manager.get_long_callback_manager(),
    prevent_initial_call=True,
)
def click_train_test(
    run_clicks,
    modal_close,
    upload_graph_file,
    upload_graph_content,
    filename,
    algorithm,
    param_table,
    causal_state,
    data_state,
    cyto_elements,
    root_leaf_table,
    link_table,
    domain_file,
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    state = json.loads(causal_state) if causal_state is not None else {}
    data_state = json.loads(data_state) if data_state is not None else {}

    def _update_states(graph, graph_df, relations):
        causal_levels, cycles = causal_method.causal_order(graph_df)
        state["graph"] = create_graph_figure(graph, causal_levels)
        state["relations"] = relations
        state["cycles"] = cycles
        data_state["columns"] = list(graph_df.columns)
        positions = {}
        if len(cyto_elements) > 0:
            for element in cyto_elements:
                if "position" in element:
                    positions[element["data"]["id"]] = element["position"]
        state["positions"] = positions

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
                constraints = _build_constraints(list(df.columns), root_leaf_table, link_table)
                graph, graph_df, relations = causal_method.run(df, algorithm, params, constraints, domain_file)

                output_dir = os.path.join(file_manager.model_directory, filename.split(".")[0])
                _dump_results(output_dir, graph_df, root_leaf_table, link_table)
                _update_states(graph, graph_df, relations)

            elif upload_graph_file is not None and upload_graph_content is not None:
                filename = None
                for name, data in zip(upload_graph_file, upload_graph_content):
                    if name.endswith(".json") or name.endswith(".pkl"):
                        file_manager.save_file(name, data)
                        filename = name
                if filename:
                    graph, graph_df, relations = causal_method.load_graph(filename)
                    _update_states(graph, graph_df, relations)

    except Exception as e:
        modal_is_open = True
        modal_content = str(e)

    return json.dumps(state), json.dumps(data_state), modal_is_open, modal_content


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
    prevent_initial_call=True,
)
def update_view(data):
    state = json.loads(data) if data is not None else {}
    graph = state.get("graph", [])
    positions = state.get("positions", {})

    if state.get("cycles", None) is not None:
        cycle_table = html.Div(children=[html.B("Cyclic Paths"), html.Hr(), create_cycle_table(state["cycles"])])
    else:
        cycle_table = None

    for element in graph:
        if "position" in element:
            element["position"] = positions.get(element["data"]["id"], element["position"])

    return graph, create_causal_relation_table(state.get("relations", None)), cycle_table


@callback(
    Output("add-root-leaf-node", "options"),
    Input("add-root-leaf-node-parent", "n_clicks"),
    State("causal-data-state", "data"),
)
def update_root_leaf_dropdown(n_clicks, data_state):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "add-root-leaf-node-parent":
        data = json.loads(data_state) if data_state is not None else {}
        options += [{"label": col, "value": col} for col in data.get("columns", [])]
    return options


@callback(Output("add-node-A", "options"), Input("add-node-A-parent", "n_clicks"), State("causal-data-state", "data"))
def update_node_a_dropdown(n_clicks, data_state):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "add-node-A-parent":
        data = json.loads(data_state) if data_state is not None else {}
        options += [{"label": col, "value": col} for col in data.get("columns", [])]
    return options


@callback(Output("add-node-B", "options"), Input("add-node-B-parent", "n_clicks"), State("causal-data-state", "data"))
def update_node_b_dropdown(n_clicks, data_state):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "add-node-B-parent":
        data = json.loads(data_state) if data_state is not None else {}
        options += [{"label": col, "value": col} for col in data.get("columns", [])]
    return options


@callback(
    Output("root-leaf-table", "children"),
    Input("select-domain", "value"),
    Input("add-root-leaf-btn", "n_clicks"),
    Input("delete-root-leaf-btn", "n_clicks"),
    [State("add-root-leaf-node", "value"), State("root-leaf-check", "value"), State("root-leaf-table", "children")],
)
def add_delete_root_leaf_node(domain_file, add_click, delete_click, metric, is_root, table):
    ctx = dash.callback_context
    metrics = {}
    if table is not None:
        if isinstance(table, list):
            table = table[0]
        metrics = {p["Metric"]: p["Type"] for p in table["props"]["data"] if p["Metric"]}

    if ctx.triggered:
        prop_id = ctx.triggered_id
        if prop_id == "select-domain" and domain_file:
            metrics = {}
            root_nodes, leaf_nodes, _, _ = causal_method.parse_domain_knowledge(domain_file)
            if root_nodes:
                for metric in root_nodes:
                    metrics[metric] = "root"
            if leaf_nodes:
                for metric in leaf_nodes:
                    metrics[metric] = "leaf"
        elif prop_id == "add-root-leaf-btn" and add_click > 0 and metric:
            metrics[metric] = "root" if len(is_root) > 0 else "leaf"
        elif prop_id == "delete-root-leaf-btn" and delete_click > 0 and metric:
            metrics.pop(metric, None)

    metrics = [{"name": key, "type": value} for key, value in metrics.items()]
    return create_root_leaf_table(metrics=metrics, height=80)


@callback(
    Output("link-table", "children"),
    Input("select-domain", "value"),
    Input("add-link-btn", "n_clicks"),
    Input("delete-link-btn", "n_clicks"),
    [
        State("add-node-A", "value"),
        State("add-node-B", "value"),
        State("link_radio_button", "value"),
        State("link-table", "children"),
    ],
)
def add_link(domain_file, add_click, delete_click, node_a, node_b, link_type, table):
    ctx = dash.callback_context
    links = {}
    if table is not None:
        if isinstance(table, list):
            table = table[0]
        links = {(p["Node A"], p["Node B"]): p["Type"] for p in table["props"]["data"] if p["Node A"]}

    if ctx.triggered:
        prop_id = ctx.triggered_id
        if prop_id == "select-domain" and domain_file:
            links = {}
            _, _, forbids, requires = causal_method.parse_domain_knowledge(domain_file)
            if forbids:
                for node_a, node_b in forbids:
                    links[(node_a, node_b)] = "⇏"
            if requires:
                for node_a, node_b in requires:
                    links[(node_a, node_b)] = "⇒"
        elif prop_id == "add-link-btn" and add_click > 0 and node_a and node_b:
            links[(node_a, node_b)] = "⇒" if link_type == "Required" else "⇏"
        elif prop_id == "delete-link-btn" and delete_click > 0 and node_a and node_b:
            links.pop((node_a, node_b), None)

    links = [{"A": a, "B": b, "type": t} for (a, b), t in links.items()]
    return create_link_table(links, height=80)


@callback(
    Output("download-data", "data"),
    Output("data-download-exception-modal", "is_open"),
    Output("data-download-exception-modal-content", "children"),
    [Input("causal-download-btn", "n_clicks"), Input("data-download-exception-modal-close", "n_clicks")],
    State("causal-select-file", "value"),
)
def download(btn_click, modal_close, filename):
    ctx = dash.callback_context
    data = None
    modal_is_open = False
    modal_content = ""

    if ctx.triggered:
        prop_id = ctx.triggered_id
        if prop_id == "causal-download-btn" and btn_click > 0:
            try:
                assert filename, "Please select the dataset name " "to download the generated causal graph."
                name = filename.split(".")[0]
                path = file_manager.get_model_download_path(name)
                data = dcc.send_file(path)
            except Exception as e:
                modal_is_open = True
                modal_content = str(e)
    return data, modal_is_open, modal_content
