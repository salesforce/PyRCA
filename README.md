# PyRCA: A Python library for Root Cause Analysis
<div align="center">
  <a href="#">
  <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9, 3.10-blue">
  </a>
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Documentation](https://xxx)
5. [Tutorials](https://xxx)
6. [How to Contribute](#how-to-contribute)

## Introduction

With the rapidly growing adoption of microservices architectures, multi-service applications become the standard 
paradigm in real-world IT applications. A multi-service application usually contains hundreds of interacting 
services, making it harder to detect service failures and identify the root causes. Root cause analysis (RCA) 
methods leverage the KPI metrics monitored on those services to determine the root causes when a system failure 
is detected, helping engineers and SREs in the troubleshooting process.

PyRCA is a Python machine-learning library designed for metric-based RCA, offering multiple state-of-the-art RCA
algorithms and an end-to-end pipeline for building RCA solutions. PyRCA includes two types of algorithms: 1. 
Identifying anomalous metrics in parallel with the observed anomaly via metric data analysis, e.g., ε-diagnosis,
and 2. Identifying root causes based a topology/causal graph representing the causal relationships between 
the observed metrics, e.g., Bayesian inference, Random Walk. Besides, PyRCA provides a convenient tool
for building causal graphs from the observed time series data and domain knowledge, helping users to develop
topology/causal graph based solutions quickly. PyRCA also provides a benchmark for evaluating various RCA
methods, which is valuable for industry and academic research.

The following list shows the supported RCA methods and features in our library:
1. ε-Diagnosis
2. Bayesian Inference-based Root Cause Analysis
3. Random Walk-based Root Cause Analysis
4. Ψ-PC-based Root Cause Analysis
5. Causal Inference-based Root Cause Analysis (CIRCA)

We will continue improving this library to make it more comprehensive in the future.

## Installation

You can install ``pyrca`` from PyPI by calling ``pip install pyrca``. You may install from source by
cloning the PyRCA repo, navigating to the root directory, and calling
``pip install .``, or ``pip install -e .`` to install in editable mode. You may install additional dependencies:

- **For plotting & visualization**: Calling ``pip install pyrca[plot]``, or ``pip install .[plot]`` from the
  root directory of the repo.
- **Install all the dependencies**: Calling ``pip install pyrca[all]``, or ``pip install .[all]`` from the
  root directory of the repo.

## Getting Started

PyRCA provides a unified interface for training RCA models and finding root causes, you only need
to specify 

- **The select RCA method**: e.g., ``BayesianNetwork``, ``EpsilonDiagnosis``.
- **The RCA configuration**: e.g., ``BayesianNetworkConfig``, ``EpsilonDiagnosisConfig``.
- **Time series data for initialization or training**: e.g., A time series data in a 
  pandas dataframe.
- **Some detected anomalous KPI metrics**: Some RCA methods require the anomalous KPI metrics detected by
  certain anomaly detector.

Let's take ``BayesianNetwork`` as an example. Suppose that ``graph_df`` is a pandas dataframe encoding
the causal graph representing causal relationships between metrics (how to construct such causal graph
will be discussed later), and ``df`` is a pandas dataframe containing the historical observed time series 
data (e.g., the index is the timestamp and each column represents one monitored metric). To train a 
``BayesianNetwork``, you can simply run the following code:

```python
from pyrca.analyzers.bayesian import BayesianNetwork
model = BayesianNetwork(config=BayesianNetwork.config_class(graph=graph_df))
model.train(df)
model.save("model_folder")
```

After the model is trained, you can use it for root cause analysis given a list of detected anomalous
metrics by a certain anomaly detector, e.g.,

```python
from pyrca.analyzers.bayesian import BayesianNetwork
model = BayesianNetwork.load("model_folder")
results = model.find_root_causes(["observed_anomalous_metric", ...])
print(results.to_dict())
```

## How to Contribute

## Contact Us
If you have any questions, comments or suggestions, please do not hesitate to contact us at pyrca@salesforce.com.

## License
[BSD 3-Clause License](LICENSE)
