#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import networkx as nx
import dash_cytoscape as cyto

from dash import dcc
from dash import html
from .utils import create_modal


default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(id)',
            'opacity': 'data(weight)',
            'background-color': 'blue',
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
            'opacity': 0.3,
            'width': 0.5
        }
    },
]


def build_cyto_graph(graph):
    cy_edges = []
    cy_nodes = []
    for node in graph.nodes():
        cy_nodes.append({"data": {"id": node, "label": node}})
    for edge in graph.edges():
        cy_edges.append({"data": {"source": edge[0], "target": edge[1]}})
    return cy_nodes + cy_edges


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
                                elements=build_cyto_graph(nx.random_geometric_graph(5, 0.5)),
                                stylesheet=default_stylesheet,
                                style={'height': '40vh', 'width': '100%'}
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id="result_table_card",
                children=[
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
