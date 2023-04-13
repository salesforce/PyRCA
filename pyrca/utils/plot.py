#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import plotly
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_plot(df, extra_df=None):
    traces, index = [], 0
    color_list = plotly.colors.qualitative.Dark24
    for i in range(df.shape[1]):
        v = df[[df.columns[i]]]
        color = color_list[index % len(color_list)]
        traces.append(
            go.Scatter(
                name=f"{df.columns[i]}",
                x=v.index,
                y=v.values.flatten().astype(float),
                mode="lines",
                line=dict(color=color),
            )
        )
        index += 1

    if extra_df is not None:
        for i in range(extra_df.shape[1]):
            v = extra_df[[extra_df.columns[i]]]
            color = color_list[index % len(color_list)]
            traces.append(
                go.Scatter(
                    name=f"{extra_df.columns[i]}_extra",
                    x=v.index,
                    y=v.values.flatten().astype(float),
                    mode="lines",
                    line=dict(color=color),
                )
            )
            index += 1

    layout = dict(
        showlegend=True,
        xaxis=dict(
            title="Time",
            type="date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
        ),
    )
    fig = make_subplots(figure=go.Figure(layout=layout))
    fig.update_yaxes(title_text="Timeseries")
    for trace in traces:
        fig.add_trace(trace)
    return fig


def plot_causal_graph_networkx(adjacency_df, node_sizes):
    graph = nx.from_pandas_adjacency(adjacency_df, create_using=nx.DiGraph)
    pos = nx.layout.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist=list(node_sizes.keys()), node_size=list(node_sizes.values()))
    nx.draw_networkx_edges(
        graph,
        pos,
        arrowstyle="->",
        arrowsize=15,
        edge_color="c",
        width=1.5,
    )
    nx.draw_networkx_labels(graph, pos, labels={c: c for c in adjacency_df.columns})
    plt.show()
