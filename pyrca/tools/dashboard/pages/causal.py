#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import networkx as nx
import dash_cytoscape as cyto

from dash import dcc
from dash import html, dash_table
from .utils import create_modal, create_param_table
from ..settings import *


default_stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "opacity": "data(weight)",
            "background-color": "#1f77b4",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-color": "black",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1,
            "line-color": "black",
            "opacity": 0.6,
            "width": 0.5,
        },
    },
]


def build_cyto_graph(graph, levels, positions, max_node_name_length=20):
    scales = (50, 80)
    node2pos = {}
    if levels is not None:
        for key, nodes in levels.items():
            for i, node in enumerate(nodes):
                node2pos[node] = (key * scales[0], i * scales[1])
    else:
        for node, pos in nx.shell_layout(graph, scale=5).items():
            node2pos[node] = (pos[0] * scales[0], pos[1] * scales[1])
    positions = {} if positions is None else positions

    cy_edges = []
    cy_nodes = []
    for node in graph.nodes():
        label = str(node)
        if len(label) > max_node_name_length:
            label = label[:max_node_name_length] + "*"
        data = {
            "data": {"id": node, "label": label},
            "position": positions.get(node, {"x": int(node2pos[node][1]), "y": int(node2pos[node][0])}),
        }
        cy_nodes.append(data)
    for edge in graph.edges():
        cy_edges.append({"data": {"source": edge[0], "target": edge[1]}})
    return cy_nodes + cy_edges


def create_graph_figure(graph=None, levels=None, positions=None):
    if graph is None:
        graph = nx.random_geometric_graph(5, 0.5)
    return build_cyto_graph(graph, levels, positions)


def create_causal_relation_table(relations=None, height=500):
    if relations is None or len(relations) == 0:
        data = [{"Node A": "", "Relation": "", "Node B": ""}]
    else:
        data = []
        for key, val in relations.items():
            i, j = key.split("<split>")
            data.append({"Node A": i, "Relation": val, "Node B": j})

    table = dash_table.DataTable(
        id="causal-relations",
        data=data,
        columns=[
            {"id": "Node A", "name": "Node A"},
            {"id": "Relation", "name": "Relation"},
            {"id": "Node B", "name": "Node B"},
        ],
        editable=False,
        sort_action="native",
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR),
        style_table={"overflowX": "scroll", "overflowY": "scroll", "height": height},
    )
    return table


def create_cycle_table(cycles, height=100):
    if cycles is None or len(cycles) == 0:
        data = [{"Cyclic Path": ""}]
    else:
        data = [{"Cyclic Path": " --> ".join([str(node) for node in path])} for path in cycles]

    table = dash_table.DataTable(
        id="causal-cycles",
        data=data,
        columns=[
            {"id": "Cyclic Path", "name": "Cyclic Path"},
        ],
        editable=False,
        sort_action="native",
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR),
        style_table={"overflowX": "scroll", "overflowY": "scroll", "height": height},
    )
    return table


def create_root_leaf_table(metrics=None, height=80):
    if metrics is None or len(metrics) == 0:
        data = [{"Metric": "", "Type": ""}]
    else:
        data = [{"Metric": metric["name"], "Type": metric["type"]} for metric in metrics]

    table = dash_table.DataTable(
        data=data,
        columns=[
            {"id": "Metric", "name": "Metric"},
            {"id": "Type", "name": "Type"},
        ],
        editable=False,
        sort_action="native",
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR),
        style_table={"overflowX": "scroll", "overflowY": "scroll", "height": height},
    )
    return table


def create_link_table(links=None, height=80):
    if links is None or len(links) == 0:
        data = [{"Node A": "", "Type": "", "Node B": ""}]
    else:
        data = [{"Node A": link["A"], "Type": link["type"], "Node B": link["B"]} for link in links]

    table = dash_table.DataTable(
        data=data,
        columns=[
            {"id": "Node A", "name": "Node A"},
            {"id": "Type", "name": "Type"},
            {"id": "Node B", "name": "Node B"},
        ],
        editable=False,
        sort_action="native",
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR),
        style_table={"overflowX": "scroll", "overflowY": "scroll", "height": height},
    )
    return table


def create_control_panel() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P(id="label", children="Upload Time Series Data / Domain Knowledge File"),
            dcc.Upload(
                id="causal-upload-data",
                children=html.Div(
                    children=[
                        html.Img(src="../assets/upload.svg"),
                        html.Div(),
                    ]
                ),
                style={
                    "height": "50px",
                    "lineHeight": "50px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "5px",
                },
                multiple=True,
            ),
            html.Br(),
            html.P("Select Data File"),
            html.Div(
                id="causal-select-file-parent",
                children=[dcc.Dropdown(id="causal-select-file", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Causal Discovery Algorithm"),
            html.Div(
                id="select-causal-method-parent",
                children=[dcc.Dropdown(id="select-causal-method", options=[], style={"width": "100%"})],
            ),
            html.Div(id="causal-param-table", children=[create_param_table(height=80)]),
            html.Br(),
            html.P("Select Domain Knowledge File"),
            html.Div(
                id="select-domain-parent",
                children=[dcc.Dropdown(id="select-domain", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="causal-run-btn", children="Run", n_clicks=0),
                    html.Button(id="causal-cancel-btn", children="Cancel", style={"margin-left": "10px"}),
                    html.Button(id="causal-download-btn", children="Download", style={"margin-left": "10px"}),
                    dcc.Download(id="download-data"),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.Hr(),
            html.P("Edit Domain Knowledge"),
            html.Label("Root or Leaf"),
            html.Div(
                children=[
                    html.Div(
                        id="add-root-leaf-node-parent",
                        children=[dcc.Dropdown(id="add-root-leaf-node", options=[])],
                        style={"width": "80%"},
                    ),
                    html.Div(
                        id="add-root-leaf-node-parent",
                        children=[
                            dcc.Checklist(
                                id="root-leaf-check",
                                options=[
                                    {"label": " Is Root", "value": "root"},
                                ],
                                value=["root"],
                            )
                        ],
                        style={"width": "20%", "margin-left": "15px"},
                    ),
                ],
                style=dict(display="flex"),
            ),
            html.Div(id="root-leaf-table", children=[create_root_leaf_table()]),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="add-root-leaf-btn", children="Add", n_clicks=0),
                    html.Button(id="delete-root-leaf-btn", children="Delete", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.Label("Forbidden or Required Links"),
            html.Div(
                id="add-node-A-parent",
                children=[dcc.Dropdown(id="add-node-A", options=[], placeholder="Node A", style={"width": "100%"})],
            ),
            html.Div(
                children=dcc.RadioItems(
                    id="link_radio_button",
                    options=[
                        {"label": " ⇒ (Required)", "value": "Required"},
                        {"label": " ⇏ (Forbidden)", "value": "Forbidden"},
                    ],
                    value="Required",
                    inline=True,
                    inputStyle={"margin-left": "20px"},
                )
            ),
            html.Div(
                id="add-node-B-parent",
                children=[dcc.Dropdown(id="add-node-B", options=[], placeholder="Node B", style={"width": "100%"})],
            ),
            html.Div(id="link-table", children=[create_link_table()]),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="add-link-btn", children="Add", n_clicks=0),
                    html.Button(id="delete-link-btn", children="Delete", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.Hr(),
            html.P(id="label", children="Open Causal Graph"),
            dcc.Upload(
                id="upload-graph",
                children=html.Div(
                    children=[
                        html.Img(src="../assets/upload.svg"),
                        html.Div(),
                    ]
                ),
                style={
                    "height": "50px",
                    "lineHeight": "50px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "5px",
                },
                multiple=True,
            ),
            create_modal(
                modal_id="causal-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="causal-exception-modal-content",
                button_id="causal-exception-modal-close",
            ),
            create_modal(
                modal_id="data-download-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="data-download-exception-modal-content",
                button_id="data-download-exception-modal-close",
            ),
        ],
    )


def create_right_column() -> html.Div:
    return html.Div(
        id="right-column-causal",
        children=[
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Causal Graph"),
                    html.Hr(),
                    html.Div(
                        id="causal-graph-plots",
                        children=[
                            cyto.Cytoscape(
                                id="cytoscape",
                                elements=[],
                                stylesheet=default_stylesheet,
                                style={"height": "60vh", "width": "100%"},
                                minZoom=0.5,
                                maxZoom=4.0,
                                layout={"name": "preset"},
                            ),
                            html.B(id="cytoscape-hover-output"),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.Div(id="causal-cycle-table"),
                    html.B("Causal Relationships"),
                    html.Hr(),
                    html.Div(id="causal-relationship-table"),
                ],
            ),
        ],
    )


def create_causal_layout() -> html.Div:
    return html.Div(
        id="causal_views",
        children=[
            # Left column
            html.Div(
                id="left-column-data",
                className="three columns",
                children=[create_control_panel()],
            ),
            # Right column
            html.Div(className="nine columns", children=create_right_column()),
        ],
    )
