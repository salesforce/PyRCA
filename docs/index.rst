.. PyRCA documentation master file, created by
   sphinx-quickstart on Wed Oct  6 13:42:55 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRCA's documentation!
========================================

Introduction
############

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

Installation
############

You can install :py:mod:`pyrca` from PyPI by calling ``pip install pyrca``. You may install from source by
cloning the PyRCA repo, navigating to the root directory, and calling
``pip install .``, or ``pip install -e .`` to install in editable mode. You may install additional dependencies:

- **For plotting & visualization**: Calling ``pip install pyrca[plot]``, or ``pip install .[plot]`` from the
  root directory of the repo.
- **Install all the dependencies**: Calling ``pip install pyrca[all]``, or ``pip install .[all]`` from the
  root directory of the repo.

Getting Started
###############

PyRCA provides a unified interface for training RCA models and finding root causes, you only need
to specify

- **The select RCA method**: e.g., :py:mod:`pyrca.analyzers.bayesian.BayesianNetwork`,
  :py:mod:`pyrca.analyzers.epsilon_diagnosis.EpsilonDiagnosis`.
- **The RCA configuration**: e.g., :py:mod:`pyrca.analyzers.bayesian.BayesianNetworkConfig`,
  :py:mod:`pyrca.analyzers.epsilon_diagnosis.EpsilonDiagnosisConfig`.
- **Time series data for initialization or training**: e.g., A time series data in a
  pandas dataframe.
- **Some detected anomalous KPI metrics**: Some RCA methods require the anomalous KPI metrics detected by
  certain anomaly detector.

Let's take ``BayesianNetwork`` as an example. Suppose that ``graph_df`` is a pandas dataframe encoding
the causal graph representing causal relationships between metrics (how to construct such causal graph
will be discussed later), and ``df`` is a pandas dataframe containing the historical observed time series
data (e.g., the index is the timestamp and each column represents one monitored metric). To train a
``BayesianNetwork``, you can simply run the following code:

.. code-block:: python

   from pyrca.analyzers.bayesian import BayesianNetwork
   model = BayesianNetwork(config=BayesianNetwork.config_class(graph=graph_df))
   model.train(df)
   model.save("model_folder")

After the model is trained, you can use it for root cause analysis given a list of detected anomalous
metrics by a certain anomaly detector, e.g.,

.. code-block:: python

   from pyrca.analyzers.bayesian import BayesianNetwork
   model = BayesianNetwork.load("model_folder")
   results = model.find_root_causes(["observed_anomalous_metric", ...])
   print(results.to_dict())

For other RCA methods, you can use similar code for discovering root causes. For example, if you want
to try ``EpsilonDiagnosis``, you can initalize ``EpsilonDiagnosis`` as follows:

.. code-block:: python

   from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis
   model = EpsilonDiagnosis(config=EpsilonDiagnosis.config_class(alpha=0.01))
   model.train(normal_data)

Here ``normal_data`` is the historical observed time series data without anomalies. To find root causes,
you can run:

.. code-block:: python

   results = model.find_root_causes(abnormal_data)
   print(results.to_dict())

where ``abnormal_data`` is the time series data in an incident window.

As mentioned above, some RCA methods require causal graphs as their inputs. To construct such causal
graphs from the observed time series data, you can utilize our tool by running ``python -m pyrca.tools``.
This command will launch a Dash app for time series data analysis and causal discovery.

.. image:: _static/dashboard.png

The dashboard allows you to try different causal discovery methods, change causal discovery parameters,
add domain knowledge constraints (e.g., root/leaf nodes, forbidden/required links), and visualize
the generated causal graph. It makes easier for manually updating causal graphs with domain knowledge.
If you satisfy with the results after several iterations, you can download the results that can be
used by the RCA methods supported in PyRCA.

Instead of using this dashboard, you can also write code for causal discovery. The package
:py:mod:`pyrca.graphs.causal` includes several causal discovery methods you can use. All of these methods
are adjusted to support domain knowledge constraints. Suppose ``df`` is the monitored time series data
and you want to apply PC for discovering causal graphs, then the following code will help:

.. code-block:: python

   from pyrca.graphs.causal.pc import PC
   model = PC(PC.config_class())
   graph_df = model.train(df)

If you have some domain knowledge constraints, you may run:

.. code-block:: python

   from pyrca.graphs.causal.pc import PC
   model = PC(PC.config_class(domain_knowledge_file="file_path"))
   graph_df = model.train(df)

The domain knowledge file has a YAML format, e.g.,

.. code-block:: yaml

   causal-graph:
     root-nodes: ["A", "B"]
     leaf-nodes: ["E", "F"]
     forbids:
       - ["A", "E"]
     requires:
       - ["A", "C"]

This domain knowledge file states that:

1. Metrics A and B must the root nodes,
2. Metrics E and F must be the leaf nodes,
3. There is no connection from A to E, and
4. There is a connection from A to C.

You can modify this file according to your domain knowledge for generating more reliable causal
graphs.

Library Design
##############

PyRCA has four key subpackages:

- :py:mod:`pyrca.graph`: This package contains the methods for discovering causal graphs from time series data.
- :py:mod:`pyrca.outliers`: This package contains some statistical based anomaly detectors used for detecting
  anomalous metrics. We will include more methods in this package in the future.
- :py:mod:`pyrca.analyzers`: This package contains all the supported RCA methods, e.g., Bayesian inference,
  Random walk, etc.
- :py:mod:`pyrca.tools`: This includes the dashboard tools for analyzing time series and building
  causal graphs.

The following figure demonstrates the main architecture of the library.

.. image:: _static/design.png
   :width: 600px

How to Contribute
#################

We welcome the contribution from the open-source community to improve the library!
Before you get started, clone this repo, run ``pip install pre-commit``, and run ``pre-commit install``
from the root directory of the repo. This will ensure all files are formatted correctly and contain
the appropriate license headers whenever you make a commit.

To add a new RCA method into the library, you may follow the steps below:
1. Create a new python script file for this RCA method in the ``pyrca/analyzers`` folder.
2. Create the configuration class that inherits from ``pyrca.base.BaseConfig``.
3. Create the method class that inherits from :py:mod:`pyrca.analyzers.base.BaseRCA`. The constructor for the new
method takes the new configuration instance as its input.
4. Implement the ``train`` function that trains or initializes the new method.
5. Implement the ``find_root_causes`` function that returns a :py:mod:`pyrca.analyzers.base.RCAResults`
instance storing root cause analysis results.

To add a new causal discovery method, you may follow the following steps:
1. Create a new python script file for this RCA method in the ``pyrca/graphs/causal`` folder.
2. Create the configuration class that inherits from :py:mod:`pyrca.graphs.causal.base.CausalModelConfig`.
3. Create the method class that inherits from :py:mod:`pyrca.graphs.causal.base.CausalModel`.
The constructor for the new method takes the new configuration instance as its input.
4. Implement the ``_train`` function that returns the discovered casual graph. The input parameters
of ``_train`` are the time series dataframe, the lists of forbidden and required links, and other
additional parameters.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   pyrca

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
