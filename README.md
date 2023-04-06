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
the observed metrics, e.g., Bayesian inference, Random Walk. Besides, PyRCA also provides a convenient tool
for building causal graphs from the observed time series data and domain knowledge, helping users to develop
topology/causal graph based solutions quickly.

## Installation

You can install ``pyrca`` from PyPI by calling ``pip install pyrca``. You may install from source by
cloning the PyRCA repo, navigating to the root directory, and calling
``pip install .``, or ``pip install -e .`` to install in editable mode. You may install additional dependencies:

- **For plotting & visualization**: Calling ``pip install pyrca[plot]``, or ``pip install .[plot]`` from the
  root directory of the repo.
- **Install all the dependencies**: Calling ``pip install pyrca[all]``, or ``pip install .[all]`` from the
  root directory of the repo.

## Getting Started

## How to Contribute

## Contact Us
If you have any questions, comments or suggestions, please do not hesitate to contact us at pyrca@salesforce.com.

## License
[BSD 3-Clause License](LICENSE)
