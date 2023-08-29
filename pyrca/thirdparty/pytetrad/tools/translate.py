## This assumes that you have already started the JVM using JPype. You may
## start the JVM only once per session. Your code should start with the following
## lines:
#
import jpype
import jpype.imports
import os

try:
    tetrad_lib_path = os.path.join(os.path.dirname(__file__), '../resources/tetrad-gui-current-launch.jar')
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx40g", classpath=[tetrad_lib_path])
except OSError:
    print("JVM already started")

import os
import sys

## Some functions wrapping various classes in Tetrad. Feel free to just steal
## the relevant code for your own projects, or 'pip install' this Github directory
## and call these functions. will add more named parameters to help one see which 
## methods for the the searches can be controlled.

# this needs to happen before import pytetrad (otherwise lib cant be found)
BASE_DIR = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from pyrca.thirdparty.causallearn.graph.GeneralGraph import GeneralGraph
from pyrca.thirdparty.causallearn.graph.GraphNode import GraphNode
from pyrca.thirdparty.causallearn.graph.Endpoint import Endpoint
from pyrca.thirdparty.causallearn.graph.Edge import Edge
from pandas import DataFrame

import java.util as util
import edu.cmu.tetrad.data as td


def pandas_data_to_tetrad(df: DataFrame, int_as_cont=False):
    dtypes = ["float16", "float32", "float64"]
    if int_as_cont:
        for i in range(3, 7):
            dtypes.append(f"int{2 ** i}")
            dtypes.append(f"uint{2 ** i}")
    cols = df.columns
    discrete_cols = [col for col in cols if df[col].dtypes not in dtypes]
    category_map = {col: {val: i for i, val in enumerate(df[col].unique())} for col in discrete_cols}
    df = df.replace(category_map)
    values = df.values
    n, p = df.shape

    variables = util.ArrayList()
    for col in cols:
        if col in discrete_cols:
            categories = util.ArrayList()
            for category in category_map[col]:
                categories.add(str(category))
            variables.add(td.DiscreteVariable(str(col), categories))
        else:
            variables.add(td.ContinuousVariable(str(col)))

    if len(discrete_cols) == len(cols):
        databox = td.IntDataBox(n, p)
    elif len(discrete_cols) == 0:
        databox = td.DoubleDataBox(n, p)
    else:
        databox = td.MixedDataBox(variables, n)

    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    return td.BoxDataSet(databox, variables)


def tetrad_data_to_pandas(data: td.DataSet):
    names = data.getVariableNames()
    columns_ = []

    for name in names:
        columns_.append(str(name))

    df: DataFrame = pd.DataFrame(columns=columns_, index=range(data.getNumRows()))

    for row in range(data.getNumRows()):
        for col in range(data.getNumColumns()):
            df.at[row, columns_[col]] = data.getObject(row, col)

    return df


def tetrad_graph_to_pcalg(g):
    endpoint_map = {"NULL": 0,
                    "CIRCLE": 1,
                    "ARROW": 2,
                    "TAIL": 3}

    nodes = g.getNodes()
    p = g.getNumNodes()
    A = np.zeros((p, p), dtype=int)

    for edge in g.getEdges():
        i = nodes.indexOf(edge.getNode1())
        j = nodes.indexOf(edge.getNode2())
        A[j][i] = endpoint_map[edge.getEndpoint1().name()]
        A[i][j] = endpoint_map[edge.getEndpoint2().name()]

    columns_ = []

    for name in nodes:
        columns_.append(str(name))

    return pd.DataFrame(A, columns=columns_)


def tetrad_graph_to_causal_learn(g):
    endpoint_map = {"TAIL": Endpoint.TAIL,
                    "NULL": Endpoint.NULL,
                    "ARROW": Endpoint.ARROW,
                    "CIRCLE": Endpoint.CIRCLE,
                    "STAR": Endpoint.STAR}

    nodes = [GraphNode(str(node.getName())) for node in g.getNodes()]
    graph = GeneralGraph(nodes)

    for edge in g.getEdges():
        node1 = graph.get_node(edge.getNode1().getName())
        node2 = graph.get_node(edge.getNode2().getName())
        endpoint1 = endpoint_map[edge.getEndpoint1().name()]
        endpoint2 = endpoint_map[edge.getEndpoint2().name()]
        graph.add_edge(Edge(node1, node2, endpoint1, endpoint2))

    return graph


# PASS ME A GraphViz Graph object and call it gdot!
def write_gdot(g, gdot):
    endpoint_map = {"TAIL": "none",
                    "ARROW": "empty",
                    "CIRCLE": "odot"}

    for node in g.getNodes():
        gdot.node(str(node.getName()),
                  shape='circle',
                  fixedsize='true',
                  style='filled',
                  color='lightgray')

    for edge in g.getEdges():
        node1 = str(edge.getNode1().getName())
        node2 = str(edge.getNode2().getName())
        endpoint1 = str(endpoint_map[edge.getEndpoint1().name()])
        endpoint2 = str(endpoint_map[edge.getEndpoint2().name()])
        color = "blue"
        if (endpoint1 == "empty") and (endpoint2 == "empty"): color = "red"
        gdot.edge(node1, node2,
                  arrowtail=endpoint1,
                  arrowhead=endpoint2,
                  dir='both', color=color)

    return gdot
