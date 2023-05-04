#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
The base class for causal discovery methods.
"""
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict
from abc import abstractmethod
from dataclasses import dataclass

from pyrca.base import BaseModel, BaseConfig
from pyrca.utils.domain import DomainParser


@dataclass
class CausalModelConfig(BaseConfig):
    """
    The configuration class for the causal discovery methods.

    :param domain_knowledge_file: The file path of the domain knowledge file.
    :param run_pdag2dag: Whether to convert a partial DAG to a DAG.
    :param max_num_points: The maximum number of data points in causal discovery.
    """

    domain_knowledge_file: str = None
    run_pdag2dag: bool = True
    max_num_points: int = 5000000


class CausalModel(BaseModel):
    @staticmethod
    def initialize():
        pass

    @staticmethod
    def finish():
        pass

    @abstractmethod
    def _train(self, df: pd.DataFrame, forbids: List, requires: List, **kwargs):
        raise NotImplementedError

    def train(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Builds the causal graph given the training dataset.

        :param df: The training dataset.
        :return: The adjacency matrix.
        """
        df = df.iloc[: self.config.max_num_points] if self.config.max_num_points is not None else df

        if self.config.domain_knowledge_file:
            parser = DomainParser(self.config.domain_knowledge_file)
            adjacency_df = self._train(
                df=df, forbids=parser.get_forbid_links(df.columns), requires=parser.get_require_links(), **kwargs
            )
        else:
            if "forbids" not in kwargs:
                kwargs["forbids"] = []
            if "requires" not in kwargs:
                kwargs["requires"] = []
            adjacency_df = self._train(df=df, **kwargs)

        var_names = adjacency_df.columns
        if self.config.run_pdag2dag:
            dag, flag = CausalModel.pdag2dag(adjacency_df.values)
            if flag is False:
                raise RuntimeError("Orientation of the undirected edges failed.")
            adjacency_df = pd.DataFrame({var_names[i]: dag[:, i] for i in range(len(var_names))}, index=var_names)
        return adjacency_df

    @staticmethod
    def pdag2dag(graph, keep_vstruct=True) -> Tuple:
        """
        Partial DAG to DAG. Implemented based on https://rdrr.io/cran/pcalg/src/R/pcalg.R
        :param graph: The partial DAG in a numpy array.
        :param keep_vstruct: Whether to keep V-structure.
        :return: The converted DAG and a flag indicating whether `pdag2dag` runs successfully.
        """

        def adj_check(g, x):
            xr, xc = g[x, :], g[:, x]
            nx = set(np.nonzero(np.logical_or(xr, xc))[0])
            un = set(np.nonzero(np.logical_and(xr, xc))[0])
            for y in un:
                adj_x = nx.difference({y})
                adj_y = set(np.nonzero(np.logical_or(g[y, :], g[:, y]))[0]).difference({x})
                if not adj_x.issubset(adj_y):
                    return False
            return True

        def find_sink(g):
            m = g - np.logical_and(g == g.T, g == 1).astype(int)
            return np.where(np.logical_and(np.sum(m, axis=1) == 0, np.sum(g, axis=0) > 0))[0]

        go_on, not_yet = True, True
        a, r = graph.copy(), graph.copy()

        while go_on and np.sum(a) > 0:
            not_yet = True
            sinks = find_sink(a)
            if len(sinks) > 0:
                index = 0
                while not_yet and index < len(sinks):
                    i = sinks[index]
                    if (not keep_vstruct) or adj_check(a, i):
                        not_yet = False
                        inc_to_i = np.nonzero(np.logical_and(a[i, :], a[:, i]))[0]
                        if len(inc_to_i) > 0:
                            r[inc_to_i, i] = 1
                            r[i, inc_to_i] = 0
                        a[i, :] = 0
                        a[:, i] = 0
                    index += 1
            go_on = not not_yet

        if not_yet:
            return graph, False
        else:
            return r, True

    @staticmethod
    def check_cycles(adjacency_df: pd.DataFrame, direct_only: bool = False) -> List:
        """
        Checks if the generated causal graph has cycles.

        :param adjacency_df: The adjacency matrix.
        :param direct_only: If True, only returns the cycles with directed edges.
        :return: A list of cycles.
        """
        graph = nx.from_pandas_adjacency(adjacency_df, create_using=nx.DiGraph)
        if not direct_only:
            return sorted(nx.simple_cycles(graph))
        else:
            node2idx = {c: i for i, c in enumerate(adjacency_df.columns)}
            adjacency_mat = adjacency_df.values
            cycles = []
            for x in sorted(nx.simple_cycles(graph)):
                flag = True
                for i in range(1, len(x)):
                    a, b = node2idx[x[i - 1]], node2idx[x[i]]
                    if adjacency_mat[a][b] == 1 and adjacency_mat[b][a] == 1:
                        flag = False
                        break
                if flag:
                    cycles.append(x)
            return cycles

    @staticmethod
    def get_parents(adjacency_df: pd.DataFrame) -> Dict:
        """
        Returns the parents of each node in the graph.

        :param adjacency_df: The adjacency matrix.
        :return: The dict of the parents, i.e., {node: [parent_1, parent_2,...]}.
        """
        var_names = adjacency_df.columns
        graph = adjacency_df.values
        parents = {name: [var_names[j] for j, v in enumerate(graph[:, i]) if v > 0] for i, name in enumerate(var_names)}
        return parents

    @staticmethod
    def get_children(adjacency_df: pd.DataFrame) -> Dict:
        """
        Returns the children of each node in the graph.

        :param adjacency_df: The adjacency matrix.
        :return: The dict of the children, i.e., {node: [child_1, child_2,...]}.
        """
        var_names = adjacency_df.columns
        graph = adjacency_df.values
        children = {
            name: [var_names[j] for j, v in enumerate(graph[i, :]) if v > 0] for i, name in enumerate(var_names)
        }
        return children

    @staticmethod
    def dump_to_tetrad_json(adjacency_df: pd.DataFrame, output_dir: str, filename: str = "graph.json"):
        """
        Dumps the graph into a Tetrad format.

        :param adjacency_df: The adjacency matrix.
        :param output_dir: The output directory.
        :param filename: The dumped file name.
        """
        var_names = [str(name) for name in adjacency_df.columns]
        g = adjacency_df.values
        edges = []
        for i in range(len(var_names)):
            for j in range(i + 1, len(var_names)):
                if g[i][j] == 1 and g[j][i] == 1:
                    edges.append((i, j, 0))
                elif g[i][j] == 1:
                    edges.append((i, j, 1))
                elif g[j][i] == 1:
                    edges.append((j, i, 1))

        graph = {
            "nodes": [],
            "edgesSet": [],
            "edgeLists": {},
            "ambiguousTriples": [],
            "underLineTriples": [],
            "dottedUnderLineTriples": [],
            "stuffRemovedSinceLastTripleAccess": False,
            "highlightedEdges": [],
            "namesHash": {},
            "pattern": False,
            "pag": False,
            "attributes": {"BIC": 0.0},
        }
        for node in var_names:
            r = {
                "nodeType": {"ordinal": 0},
                "nodeVariableType": "DOMAIN",
                "centerX": 100,
                "centerY": 100,
                "attributes": {},
                "name": node,
            }
            graph["nodes"].append(r)
            graph["namesHash"][node] = r
            graph["edgeLists"][node] = []

        for i, j, e in edges:
            a, b = var_names[i], var_names[j]
            r = {
                "node1": {
                    "nodeType": {"ordinal": 0},
                    "nodeVariableType": "DOMAIN",
                    "centerX": 100,
                    "centerY": 100,
                    "attributes": {},
                    "name": a,
                },
                "node2": {
                    "nodeType": {"ordinal": 0},
                    "nodeVariableType": "DOMAIN",
                    "centerX": 100,
                    "centerY": 100,
                    "attributes": {},
                    "name": b,
                },
                "endpoint1": {"ordinal": 0},
                "endpoint2": {"ordinal": e},
                "bold": False,
                "properties": [],
                "edgeTypeProbabilities": [],
            }
            graph["edgesSet"].append(r)
            graph["edgeLists"][a].append(r)

        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(graph, f)

    @staticmethod
    def load_from_tetrad_json(filepath: str) -> pd.DataFrame:
        """
        Loads the adjacency matrix from a Tetrad file.

        :param filepath: The file path of the dumped graph.
        :return: The adjacency matrix.
        """
        with open(filepath, "r") as f:
            g = json.load(f)

        var_names = []
        for node in g["nodes"]:
            var_names.append(node["name"])
        name_to_index = {name: i for i, name in enumerate(var_names)}
        mat = np.zeros((len(var_names), len(var_names)), dtype=np.int)

        for edge in g["edgesSet"]:
            endpoint1 = edge["endpoint1"]["ordinal"]
            endpoint2 = edge["endpoint2"]["ordinal"]
            a = edge["node1"]["name"]
            b = edge["node2"]["name"]
            mat[name_to_index[a]][name_to_index[b]] = 1
            if endpoint2 == 0:
                mat[name_to_index[b]][name_to_index[a]] = 1

        return pd.DataFrame({var_names[i]: mat[:, i] for i in range(len(var_names))}, index=var_names)

    @staticmethod
    def plot_causal_graph_networkx(adjacency_df):
        import matplotlib.pyplot as plt

        graph = nx.from_pandas_adjacency(adjacency_df, create_using=nx.DiGraph)
        print("Cycles (including undirected edges):")
        for x in sorted(nx.simple_cycles(graph)):
            print(x)

        print("Cycles (directed edges only):")
        node2idx = {c: i for i, c in enumerate(adjacency_df.columns)}
        adjacency_mat = adjacency_df.values
        for x in sorted(nx.simple_cycles(graph)):
            flag = True
            for i in range(1, len(x)):
                a, b = node2idx[x[i - 1]], node2idx[x[i]]
                if adjacency_mat[a][b] == 1 and adjacency_mat[b][a] == 1:
                    flag = False
                    break
            if flag:
                print(x)

        pos = nx.layout.circular_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
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
