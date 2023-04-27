# PyRCA: A Python library for Root Cause Analysis
<div align="center">
  <a href="#">
  <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9-blue">
  </a>
  <a href="https://pypi.python.org/pypi/sfr-pyrca">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/sfr-pyrca.svg"/>
  </a>
  <a href="https://fuzzy-disco-r42n6p1.pages.github.io">
  <img alt="Documentation" src="https://github.com/salesforce/PyRCA/actions/workflows/docs.yml/badge.svg"/>
  </a>
  <a href="https://pepy.tech/project/sfr-pyrca">
  <img alt="Downloads" src="https://pepy.tech/badge/sfr-pyrca">
  </a>
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
  <img alt="DOI" src="https://zenodo.org/badge/DOI/10.48550/ARXIV.xxxx.xxxxx.svg"/>
  </a>
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Documentation](https://fuzzy-disco-r42n6p1.pages.github.io/)
5. [Tutorial](https://github.com/salesforce/PyRCA/tree/main/examples)
6. [Example](#application-example)
7. [Benchmarks](#benchmarks)
8. [How to Contribute](#how-to-contribute)

## Introduction

With the rapidly growing adoption of microservices architectures, multi-service applications become the standard 
paradigm in real-world IT applications. A multi-service application usually contains hundreds of interacting 
services, making it harder to detect service failures and identify the root causes. Root cause analysis (RCA) 
methods usually leverage the KPI metrics, traces or logs monitored on those services to determine the root causes 
when a system failure is detected, helping engineers and SREs in the troubleshooting process.

PyRCA is a Python machine-learning library designed for root cause analysis, offering multiple state-of-the-art RCA
algorithms and an end-to-end pipeline for building RCA solutions. Currently, PyRCA mainly focuses on metric-based RCA 
including two types of algorithms: 1. Identifying anomalous metrics in parallel with the observed anomaly via 
metric data analysis, e.g., ε-diagnosis, and 2. Identifying root causes based a topology/causal graph representing 
the causal relationships between the observed metrics, e.g., Bayesian inference, Random Walk. Besides, PyRCA 
provides a convenient tool for building causal graphs from the observed time series data and domain knowledge, 
helping users to develop graph-based solutions quickly. PyRCA also provides a benchmark for evaluating 
various RCA methods, which is valuable for industry and academic research.

The following list shows the supported RCA methods in our library:
1. ε-Diagnosis
2. Bayesian Inference-based RCA (BI)
3. Random Walk-based RCA (RW)
4. Root Casue Discovery method (RCD)
5. Hypothesis Testing-based RCA (HT)

We will continue improving this library to make it more comprehensive in the future. In the future, 
PyRCA will support trace and log based RCA methods as well.

## Installation

You can install ``pyrca`` from PyPI by calling ``pip install sfr-pyrca``. You may install from source by
cloning the PyRCA repo, navigating to the root directory, and calling
``pip install .``, or ``pip install -e .`` to install in editable mode. You may install additional dependencies:

- **For plotting & visualization**: Calling ``pip install sfr-pyrca[plot]``, or ``pip install .[plot]`` from the
  root directory of the repo.
- **Install all the dependencies**: Calling ``pip install sfr-pyrca[all]``, or ``pip install .[all]`` from the
  root directory of the repo.

## Getting Started

PyRCA provides a unified interface for training RCA models and finding root causes. To apply
a certain RCA method, you only need to specify: 

- **The select RCA method**: e.g., ``BayesianNetwork``, ``EpsilonDiagnosis``.
- **The RCA configuration**: e.g., ``BayesianNetworkConfig``, ``EpsilonDiagnosisConfig``.
- **Time series data for initialization or training**: e.g., A time series data in a 
  pandas dataframe.
- **Some detected anomalous KPI metrics**: Some RCA methods require the anomalous KPI metrics detected by
  certain anomaly detector.

Let's take ``BayesianNetwork`` as an example. Suppose that ``graph_df`` is a pandas dataframe of
the graph representing causal relationships between metrics (how to construct such causal graph
will be discussed later), and ``df`` is a pandas dataframe containing the historical observed time series 
data (e.g., the index is the timestamp and each column represents one monitored metric). To train a 
``BayesianNetwork``, you can simply run the following code:

```python
from pyrca.analyzers.bayesian import BayesianNetwork
model = BayesianNetwork(config=BayesianNetwork.config_class(graph=graph_df))
model.train(df)
model.save("model_folder")
```

After the model is trained, you can use it to find root causes of an incident given a list of anomalous
metrics detected by a certain anomaly detector, e.g.,

```python
from pyrca.analyzers.bayesian import BayesianNetwork
model = BayesianNetwork.load("model_folder")
results = model.find_root_causes(["observed_anomalous_metric", ...])
print(results.to_dict())
```

For other RCA methods, you can write similar code as above for finding root causes. For example, if you want
to try ``EpsilonDiagnosis``, you can initalize ``EpsilonDiagnosis`` as follows:

```python
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis
model = EpsilonDiagnosis(config=EpsilonDiagnosis.config_class(alpha=0.01))
model.train(normal_data)
```

Here ``normal_data`` is the historically observed time series data without anomalies. To identify root causes,
you can run:

```python
results = model.find_root_causes(abnormal_data)
print(results.to_dict())
```

where ``abnormal_data`` is the time series data collected in an incident window.

As mentioned above, some RCA methods such as ``BayesianNetwork`` require causal graphs as their inputs. To construct such causal
graphs from the observed time series data, you can utilize our tool by running ``python -m pyrca.tools``.
This command will launch a Dash app for time series data analysis and causal discovery.
![alt text](https://github.com/salesforce/PyRCA/raw/main/docs/_static/dashboard.png)

The dashboard allows you to try different causal discovery methods, adjust causal discovery parameters,
add domain knowledge constraints (e.g., root/leaf nodes, forbidden/required links), and visualize
the generated causal graphs. It makes easier for manually revising causal graphs based on domain knowledge.
You can download the graph generated by this tool if you satisfy with it. The graph can be used by the RCA 
methods supported in PyRCA.

Instead of using this dashboard, you can also write code for building such graphs. The package 
``pyrca.graphs.causal`` includes several popular causal discovery methods you can use. All of these methods
support domain knowledge constraints. Suppose ``df`` is the observed time series data
and you want to apply the PC algorithm for building causal graphs, then the following code will help:

```python
from pyrca.graphs.causal.pc import PC
model = PC(PC.config_class())
graph_df = model.train(df)
```

If you have some domain knowledge constraints, you may run:

```python
from pyrca.graphs.causal.pc import PC
model = PC(PC.config_class(domain_knowledge_file="file_path"))
graph_df = model.train(df)
```

The domain knowledge file has a YAML format, e.g.,

```yaml
causal-graph:
  root-nodes: ["A", "B"]
  leaf-nodes: ["E", "F"]
  forbids:
    - ["A", "E"]
  requires: 
    - ["A", "C"]
```

This domain knowledge file states that: 
1. Metrics A and B must the root nodes, 
2. Metrics E and F must be the leaf nodes,
3. There is no connection from A to E, and 
4. There is a connection from A to C. 

You can write your domain knowledge file based on this template for generating more reliable causal
graphs.

## Application Example

[Here](https://github.com/salesforce/PyRCA/tree/main/pyrca/applications/example) is a real-world example
of applying ``BayesianNetwork`` to build a solution for RCA, which is adapted from our internal use cases. 
The "config" folder includes the settings for the stats-based anomaly detector and the domain knowledge. 
The "models" folder stores the causal graph and the trained Bayesian network. The ``RCAEngine`` class in the "rca.py" 
file implements the methods for building causal graphs, training Bayesian networks and finding root causes 
by utilizing the modules provided by PyRCA. You can directly use this class if the stats-based anomaly detector 
and Bayesian inference are suitable for your RCA problems. For example, given a time series dataframe ``df``, 
you can build and train a Bayesian network via the following code:

```python
from pyrca.applications.example.rca import RCAEngine
engine = RCAEngine()
engine.build_causal_graph(
    df=df,
    run_pdag2dag=True,
    max_num_points=5000000,
    verbose=True
)
bn = engine.train_bayesian_network(dfs=[df])
bn.print_probabilities()
```

After the Bayesian network is constructed, you can use it directly for finding root causes:

```python
engine = RCAEngine()
result = engine.find_root_causes_bn(anomalies=["conn_pool", "apt"])
pprint.pprint(result)
```

The inputs of ``find_root_causes_bn`` is a list of the anomalous metrics detected by the stats-based
anomaly detector. This method will estimate the probability of a node being a root cause and extract
the paths from a potential root cause node to the leaf nodes.

## Benchmarks

The following table summarizes the RCA performance of different methods on the simulated dataset.
How to generate the simulated dataset can be found [here](https://github.com/salesforce/PyRCA/blob/main/examples/DataGeneration.ipynb),
and how to test different RCA methods can be found [here](https://github.com/salesforce/PyRCA/blob/main/examples/Root%20Cause%20Analysis.ipynb).

<div align="center">

|                             |  Recall@1   |  Recall@3   |  Recall@5   |
:---------------------------:|:-----------:|:-----------:|:-----------:
|         ε-Diagnosis         | 0.06 ± 0.02 | 0.16 ± 0.04 | 0.16 ± 0.04 |
|             RCD             | 0.28 ± 0.05 | 0.29 ± 0.05 | 0.30 ± 0.05 |
|          Local-RCD          | 0.44 ± 0.05 | 0.70 ± 0.05 | 0.70 ± 0.05 |
|         Random Walk         | 0.07 ± 0.03 | 0.20 ± 0.04 | 0.24 ± 0.04 |
|      Random Walk (PC)       | 0.06 ± 0.02 | 0.17 ± 0.04 | 0.21 ± 0.04 |
|     Bayesian Inference      | 0.15 ± 0.04 | 0.35 ± 0.05 | 0.43 ± 0.05 |
|   Bayesian Inference (PC)   | 0.11 ± 0.03 | 0.30 ± 0.05 | 0.40 ± 0.05 |
|     Hypothesis-testing      | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
|   Hypothesis-testing (PC)   | 0.95 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 |
|  Hypothesis-testing (ADJ)   | 0.95 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Hypothesis-testing (ADJ-PC) | 0.77 ± 0.04 | 0.92 ± 0.03 | 0.92 ± 0.03 |

</div>

ε-Diagnosis and RCD are one-phase RCA methods, while the rest methods are two-phase RCA methods. 
Local-RCD denotes the RCD algorithm with localized learning. The Bayesian Inference algorithm 
computes the root cause scores by estimating each structural causal model. Hypothesis-testing (ADJ) denotes 
the hypothesis-testing algorithm with descendant adjustment. For the two-phase models, the algorithms 
without suffix indicate that the root cause localization algorithm use the true causal graph for model 
training. The algorithms with suffix "PC" indicate the causal graph is estimated via PC algorithm.

## How to Contribute

We welcome the contribution from the open-source community to improve the library!
Before you get started, clone this repo, run `pip install pre-commit`, and run `pre-commit install` 
from the root directory of the repo. This will ensure all files are formatted correctly and contain 
the appropriate license headers whenever you make a commit. 

To add a new RCA method into the library, you may follow the steps below:
1. Create a new python script file for this RCA method in the ``pyrca/analyzers`` folder.
2. Create the configuration class inheriting from ``pyrca.base.BaseConfig``.
3. Create the method class inheriting from ``pyrca.analyzers.base.BaseRCA``. The constructor for the new 
method takes the new configuration instance as its input.
4. Implement the ``train`` function that trains or initializes the new method.
5. Implement the ``find_root_causes`` function that returns a ``pyrca.analyzers.base.RCAResults`` 
instance for root cause analysis results.

To add a new causal discovery method, you may follow the following steps:
1. Create a new python script file for this RCA method in the ``pyrca/graphs/causal`` folder.
2. Create the configuration class that inherits from ``pyrca.graphs.causal.base.CausalModelConfig``.
3. Create the method class that inherits from ``pyrca.graphs.causal.base.CausalModel``. 
The constructor for the new method takes the new configuration instance as its input.
4. Implement the ``_train`` function that returns the discovered casual graph. The input parameters
of ``_train`` are the time series dataframe, the lists of forbidden and required links, and other
additional parameters.

## Contact Us
If you have any questions, comments or suggestions, please do not hesitate to contact us at pyrca@salesforce.com.

## License
[BSD 3-Clause License](LICENSE)
