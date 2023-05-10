#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
"""
Simulated Data Generation
"""
from dataclasses import dataclass
import numpy as np


def _normal_weight() -> float:
    rng = np.random.default_rng()
    weight = rng.standard_normal()
    return np.sign(weight) * (abs(weight) + 0.2)


def _uniform_weight() -> float:
    rng = np.random.default_rng()
    segments = [(-2.0, -0.5), (0.5, 2.0)]
    low, high = rng.choice(segments)
    return rng.uniform(low=low, high=high)


@dataclass
class DAGGenConfig:
    """
    The configuration class of generating DAG.

    :param num_node: Number of nodes in DAG.
    :param num_edge: Number of edges in DAG.
    :param rng: A random generator.
    """

    num_node: int = 20
    num_edge: int = 30
    rng: np.random.Generator = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()


class DAGGen:
    """
    The DAG generation.
    """

    config_class = DAGGenConfig

    def __init__(self, config: DAGGenConfig):
        self.config = config

    def gen(self):
        """
        Generate a directed acyclic graphs with a single end.
        The first node with index of 0 is the only end that does not have results.

        :return: A matrix, where matrix[i, j] != 0 means i is the cause of j.
        """
        num_edge = min(
            max(self.config.num_edge, self.config.num_node - 1),
            int(self.config.num_node * (self.config.num_node - 1) / 2),
        )
        matrix = np.zeros((self.config.num_node, self.config.num_node))

        # Make the graphs connected
        for cause in range(1, self.config.num_node):
            result = self.config.rng.integers(low=0, high=cause)
            matrix[cause, result] = 1
        num_edge -= self.config.num_node - 1

        # Add extra edges
        while num_edge > 0:
            cause = self.config.rng.integers(low=1, high=self.config.num_node)
            result = self.config.rng.integers(low=0, high=cause)
            if not matrix[cause, result]:
                matrix[cause, result] = 1
                num_edge -= 1
        return matrix


@dataclass
class DataGenConfig:
    """
    The configuration class of generating RCA Data.

    :param dag: The dependency graph.
    :param noise_type: probability distribution of each node's noise,
        it is required to be in valid_noise type list.
    :param func_type: causal function form of each node,
        it is required to be in valid_noise type list.
    :param num_samples: Number of samples.
    :param weight_generator: random generator for model weights.
    """

    dag: np.ndarray
    noise_type: str = None
    func_type: str = None
    num_samples: int = 5000
    weight_generator: str = "normal"

    _VALID_NOISE = ["normal", "exponential", "uniform", "laplace"]
    _VALID_FUNC = ["identity", "square", "sin", "tanh"]
    _VALID_WEIGHT = ["normal", "uniform"]

    # Sanity check.
    def __post_init__(self):
        assert self.dag is not None, "DAG is None"

        if self.noise_type is None:
            self.noise_type = self._VALID_NOISE[np.random.choice(len(self._VALID_NOISE))]

        if self.noise_type == "normal":
            self.noise_type = lambda size: np.random.normal(0, 1, size)
        elif self.noise_type == "exponential":
            self.noise_type = lambda size: np.random.exponential(1, size)
        elif self.noise_type == "uniform":
            self.noise_type = lambda size: np.random.uniform(-0.5, 0.5, size)
        elif self.noise_type == "laplace":
            self.noise_type = lambda size: np.random.laplace(0, 1, size)
        else:
            assert (
                self.noise_type in self._VALID_NOISE
            ), f"{self.noise_type} is not supported. The valid value is {self._VALID_NOISE}"

        if self.func_type is None:
            self.func_type = self._VALID_FUNC[np.random.choice(len(self._VALID_FUNC))]

        if self.func_type == "identity":
            self.func_type = lambda x: x
        elif self.func_type == "square":
            self.func_type = lambda x: np.power(x, 2)
        elif self.func_type == "sin":
            self.func_type = lambda x: np.sin(x)
        elif self.func_type == "tanh":
            self.func_type = lambda x: np.tanh(x)
        else:
            assert (
                self.func_type in self._VALID_FUNC
            ), f"{self.func_type} is not supported. The valid value is {self._VALID_FUNC}"

        if self.weight_generator is None:
            self.weight_generator = self._VALID_WEIGHT[np.random.choice(len(self._VALID_WEIGHT))]

        if self.weight_generator == "normal":
            self.weight_generator = _normal_weight
        elif self.weight_generator == "uniform":
            self.weight_generator = _uniform_weight
        else:
            assert (
                self.weight_generator in self._VALID_WEIGHT
            ), f"{self.weight_generator} is not supported. The valid value is {self._VALID_WEIGHT}"


class DataGen:
    """
    Normal data generation.

    Generates data (n_samples, n_nodes) according to DAG matrix.
    """

    config_class = DataGenConfig

    def __init__(self, config: DataGenConfig):
        self.config = config

    def gen(self):
        r"""
        For each node

        .. math:: x_{i} = \sum_{x_{j} \in Pa(x_i)} A_{ij} f_i(x_j) +  \beta_i * noise_i

        where f_i indicates element-wise transformation, it is chosen from identity, square, sin, tanh,
        noise_i is chosen from Exponential(1), Normal(0,1), Uniform(-0.5,0.5), Laplace(0, 1)
        Both the weights of A_{ij} and \beta_i are chosen from _uniform_weight or _normal_weight.

        :returns:
            - data: (num_samples, num_node) data of each variable x_i.
            - parent_weights: (num_node, num_node) Combination weights of each variable A_{ij}.
            - noise_weights: (num_node,) noise weight for of each variable \beta_i.
        """

        num_node, _ = self.config.dag.shape
        num_samples = self.config.num_samples
        graph = self.config.dag

        data = np.zeros((num_samples, num_node))
        parent_weights = np.zeros((num_node, num_node))
        noise_weights = np.zeros(num_node)

        for i in range(num_node - 1, -1, -1):
            noise_data = self.config.noise_type(num_samples)

            parents = np.where(graph[:, i] != 0)[0]
            if parents.shape[0] > 0:
                parents_data = self.config.func_type(data[:num_samples, parents])

                noise_weights[i] = self.config.weight_generator()
                for j in parents:
                    parent_weights[j, i] = self.config.weight_generator()
                data[:num_samples, i] = parents_data @ parent_weights[parents, i] + noise_weights[i] * noise_data
            else:
                data[:num_samples, i] = noise_data

        return data, parent_weights, noise_weights, self.config.func_type, self.config.noise_type


@dataclass
class AnomalyDataGenConfig:
    """
    The configuration class of generating RCA Data.

    :param parent_weights: The weights of parents of each node.
    :param noise_weights: The noise weights of each node.
    :param noise_type: probability distribution of each node's noise,
        it is required to be in valid_noise type list.
    :param func_type: causal function form of each node,
        it is required to be in valid_noise type list.
    :param baseline: baseline of normal data.
    :param threshold: threshold to differentiate anomaly data from stats-based anomaly detector.
    :param num_samples: Number of anomaly samples.
    :param anomaly_type: 0 change the weight of noise term, 1 add a constant shift to noise term,
        2 change the weight of parent nodes.
    :param weight_generator: random generator for model weights.
    """

    parent_weights: np.array
    noise_weights: np.array
    noise_type: str
    func_type: str
    baseline: float
    threshold: float
    num_samples: int = 5000
    anomaly_type: str = 0
    weight_generator: str = "normal"

    _VALID_NOISE = ["normal", "exponential", "uniform", "laplace"]
    _VALID_FUNC = ["identity", "square", "sin", "tanh"]
    _VALID_WEIGHT = ["normal", "uniform"]

    # Sanity check.
    def __post_init__(self):
        assert self.parent_weights is not None, "parent_weights is None"
        assert self.noise_weights is not None, "noise_weights is None"
        assert self.threshold is not None, "threshold is None"
        assert self.baseline is not None, "baseline is None"
        assert self.anomaly_type in np.arange(3), "anomaly type {self.anomaly_type} is not supported"

        if self.weight_generator is None:
            self.weight_generator = self._VALID_WEIGHT[np.random.choice(len(self._VALID_WEIGHT))]

        if self.weight_generator == "normal":
            self.weight_generator = _normal_weight
        elif self.weight_generator == "uniform":
            self.weight_generator = _uniform_weight
        else:
            assert (
                self.weight_generator in self._VALID_WEIGHT
            ), f"{self.weight_generator} is not supported. The valid value is {self._VALID_WEIGHT}"


class AnomalyDataGen:
    """
    Anomaly data generation.

    Generates anomaly data (n_samples, n_nodes).
    """

    config_class = AnomalyDataGenConfig

    def __init__(self, config: DataGenConfig):
        self.config = config

    def gen(self):
        """
        Generate anomaly data by randomly choose the root cause nodes.

        :returns:
            - data: (num_samples, num_node) data of each variable x_i in anomaly phase.
            - root causes: (num_node) root causes of anomaly data.
        """
        assert self.config.parent_weights.shape[0] == self.config.noise_weights.shape[0]
        num_node = self.config.parent_weights.shape[0]

        data = np.zeros((self.config.num_samples, num_node))

        # # Inject a fault
        rng = np.random.default_rng()
        num_causes = min(rng.poisson(1) + 1, num_node)

        if self.config.anomaly_type != 2:
            # support type 0, 1 injection noise (weight for noise, constant)
            causes = rng.choice(np.arange(1, num_node), size=num_causes, replace=False)
            fault = np.zeros(num_node)
        else:
            # support type 2 injection noise (weight for parents)
            nodeid_haveparents = np.where((self.config.parent_weights != 0).sum(axis=0) != 0)[0]
            nodeid_haveparents = np.delete(nodeid_haveparents, 0)
            col_id = rng.choice(nodeid_haveparents, size=num_causes, replace=False)
            row_id = []
            for target in col_id:
                parent_node = rng.choice(
                    np.where(self.config.parent_weights[:, target] != 0)[0], size=1, replace=False
                )[0]
                row_id.append(parent_node)
            row_id = np.array(row_id)
            fault = np.zeros((num_node, num_node))
        alpha = rng.standard_exponential(size=num_causes)
        while True:
            if self.config.anomaly_type != 2:
                fault[causes] = alpha
            else:
                fault[row_id, col_id] = alpha
            for i in range(num_node - 1, -1, -1):
                noise_data = self.config.noise_type(self.config.num_samples)
                parents = np.where(self.config.parent_weights[:, i] != 0)[0]
                if parents.shape[0] > 0:
                    parents_data = self.config.func_type(data[:, parents])
                    if self.config.anomaly_type == 0:
                        # 0. anomaly event indicate the change of constant
                        data[: self.config.num_samples, i] = (
                            parents_data @ self.config.parent_weights[parents, i]
                            + self.config.noise_weights[i] * noise_data
                            + fault[i]
                        )
                    elif self.config.anomaly_type == 1:
                        # 1. anomaly event indicate the change of noise weight
                        data[: self.config.num_samples, i] = (
                            parents_data @ self.config.parent_weights[parents, i]
                            + (self.config.noise_weights[i] + fault[i]) * noise_data
                        )
                    else:
                        # 2. anomaly event indicate the change of parent weights
                        data[: self.config.num_samples, i] = (
                            parents_data @ (self.config.parent_weights[parents, i] + fault[parents, i])
                            + self.config.noise_weights[i] * noise_data
                        )
                else:
                    if self.config.anomaly_type == 0:
                        # 0. anomaly event indicate the change of constant
                        data[: self.config.num_samples, i] = fault[i] + noise_data
                    elif self.config.anomaly_type == 1:
                        # 1. anomaly event indicate the change of noise weight
                        data[: self.config.num_samples, i] = (1 + fault[i]) * noise_data
                    else:
                        # 3. anomaly event indicate the change of parent weights, but node does not have parents
                        data[: self.config.num_samples, i] = noise_data
            if (
                abs(data[: self.config.num_samples, i] - self.config.baseline) > self.config.threshold
            ).sum() == self.config.num_samples:  # > int(num_samples / 2):
                break
            alpha *= 2

            if alpha.max() > 1e10:
                print(alpha)
                raise "Current data generation model can not trigger anomalies, " "please adjust parameters or try it again"
        return data, fault
