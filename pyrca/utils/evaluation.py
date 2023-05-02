import pandas as pd
from typing import List

from pyrca.thirdparty.causallearn.graph.GraphNode import GraphNode
from pyrca.thirdparty.causallearn.graph.Node import Node
from pyrca.thirdparty.causallearn.graph.GeneralGraph import GeneralGraph
from pyrca.thirdparty.causallearn.graph.Endpoint import Endpoint
from pyrca.thirdparty.causallearn.graph.Edge import Edge
from pyrca.thirdparty.causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from pyrca.thirdparty.causallearn.graph.SHD import SHD

def adjmatrix_to_graph(adjmatrix : pd.DataFrame):
    nodes: List[Node] = []
    for name in adjmatrix.columns:
        node = GraphNode(name)
        nodes.append(node)
    graph = GeneralGraph(nodes)
    for i in range(adjmatrix.shape[0]):
        for j in range(i + 1, adjmatrix.shape[0]):
            if adjmatrix.values[j, i] != 0:
                graph.add_edge(Edge(nodes[j], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
    return graph

def precision(true_matrix : pd.DataFrame,
              estimated_matrix: pd.DataFrame):
    """
    Compute the precision between true graph and estimated graph
    """
    true_graph = adjmatrix_to_graph(true_matrix)
    estimated_graph = adjmatrix_to_graph(estimated_matrix)

    adj = AdjacencyConfusion(true_graph, estimated_graph)
    adjPrec = adj.get_adj_precision()
    return adjPrec

def recall(true_matrix : pd.DataFrame,
           estimated_matrix : pd.DataFrame):
    """
    Compute the recall between true graph and estimated graph
    """
    true_graph = adjmatrix_to_graph(true_matrix)
    estimated_graph = adjmatrix_to_graph(estimated_matrix)

    adj = AdjacencyConfusion(true_graph, estimated_graph)
    adjRec = adj.get_adj_recall()
    return adjRec

def f1(true_matrix : pd.DataFrame,
       estimated_matrix : pd.DataFrame):
    """
    Compute the F1 score between true graph and estimated graph
    """
    adjPrec = precision(true_matrix, estimated_matrix)
    adjRec = recall(true_matrix, estimated_matrix)
    f1 = 2.*adjPrec*adjRec/(adjPrec+ adjRec + 1e-7)
    return f1

def shd(true_matrix : pd.DataFrame,
           estimated_matrix : pd.DataFrame):
    """
    Compute the Structural Hamming Distance (SHD) between true graph and estimated graph,
    which indicates the number of wrongly directed edges
    """
    true_graph = adjmatrix_to_graph(true_matrix)
    estimated_graph = adjmatrix_to_graph(estimated_matrix)
    shd = SHD(true_graph, estimated_graph).get_shd()
    return shd