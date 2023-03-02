#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import networkx as nx
import dash_cytoscape as cyto

from dash import dcc
from dash import html, dash_table
from .utils import create_modal, create_param_table
from ..settings import *


default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(id)',
            'opacity': 'data(weight)',
            'background-color': '#1f77b4',
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-color': 'black',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1,
            'line-color': 'black',
            'opacity': 0.6,
            'width': 0.5
        }
    },
]


def build_cyto_graph(graph, levels):
    cy_edges = []
    cy_nodes = []
    for node in graph.nodes():
        cy_nodes.append({"data": {"id": node, "label": node}})
    for edge in graph.edges():
        cy_edges.append({"data": {"source": edge[0], "target": edge[1]}})
    return cy_nodes + cy_edges


def create_graph_figure(graph=None, levels=None):
    if graph is None:
        graph = nx.random_geometric_graph(5, 0.5)
    fig = build_cyto_graph(graph, levels)
    return fig


def create_causal_relation_table(relations=None, height=200):
    if relations is None or len(relations) == 0:
        data = [{"Node A": "", "Relation": "", "Node B": ""}]
    else:
        data = [{"Node A": i, "Relation": v, "Node B": j}
                for (i, j), v in relations.items()]

    table = dash_table.DataTable(
        id="causal-relations",
        data=data,
        columns=[
            {"id": "Node A", "name": "Node A"},
            {"id": "Relation", "name": "Relation"},
            {"id": "Node B", "name": "Node B"}
        ],
        editable=False,
        sort_action="native",
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR),
        style_table={
            "overflowX": "scroll",
            "overflowY": "scroll",
            "height": height
        },
    )
    return table


def create_cycle_table(cycles, height=100):
    if cycles is None or len(cycles) == 0:
        data = [{"Cyclic Path": ""}]
    else:
        data = [{"Cyclic Path": " --> ".join([str(node) for node in path])}
                for path in cycles]

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
        style_table={
            "overflowX": "scroll",
            "overflowY": "scroll",
            "height": height
        },
    )
    return table


def create_control_panel() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P(id="label", children="Upload Time Series Data File"),
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
                children=[
                    dcc.Dropdown(
                        id="causal-select-file",
                        options=[],
                        style={"width": "100%"}
                    )]
            ),

            html.Br(),
            html.P("Causal Discovery Algorithm"),
            html.Div(
                id="select-causal-method-parent",
                children=[
                    dcc.Dropdown(
                        id="select-causal-method",
                        options=[],
                        style={"width": "100%"}
                    )]
            ),

            html.Br(),
            html.P("Algorithm Setting"),
            html.Div(
                id="causal-param-table",
                children=[create_param_table()]
            ),

            html.Br(),
            html.Div(
                children=[
                    html.Button(id="causal-run-btn", children="Run", n_clicks=0),
                    html.Button(id="causal-cancel-btn", children="Cancel", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"}
            ),

            create_modal(
                modal_id="causal-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="causal-exception-modal-content",
                button_id="causal-exception-modal-close"
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
                                elements=create_graph_figure(),
                                stylesheet=default_stylesheet,
                                style={"height": "60vh", "width": "100%"}
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.Div(id="causal-cycle-table"),
                    html.B("Causal Relationships"),
                    html.Hr(),
                    html.Div(id="causal-relationship-table")
                ]
            )
        ]
    )


def create_causal_layout() -> html.Div:
    return html.Div(
        id="causal_views",
        children=[
            # Left column
            html.Div(
                id="left-column-data",
                className="three columns",
                children=[
                    create_control_panel()
                ],
            ),
            # Right column
            html.Div(
                className="nine columns",
                children=create_right_column()
            )
        ]
    )
