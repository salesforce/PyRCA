#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
from setuptools import setup, find_namespace_packages

extras_require = {
    "plot": ["plotly>=4", "dash>=2.0", "dash_bootstrap_components>=1.0", "jupyter-dash>=0.4", "dash[diskcache]", "dash_cytoscape"]
}
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="sfr-pyrca",
    version="1.0.1",
    author="Salesforce Research Warden AIOps",
    description="PyRCA: A Python library for Root Cause Analysis",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="AIOps RCA root cause analysis",
    url="https://github.com/salesforce/PyRCA",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="pyrca.*"),
    package_dir={"pyrca": "pyrca"},
    package_data={"pyrca": ["tools/dashboard/assets/*"]},
    install_requires=[
        "numpy>=1.17",
        "pandas>=1.1.0",
        "scikit-learn>=0.24,<1.2",
        "networkx>=2.6",
        "matplotlib",
        "pyyaml",
        "schema",
        "pyparsing",
        "dill",
        "tqdm",
        "wheel",
        "packaging",
        "JPype1"
    ],
    extras_require=extras_require,
    python_requires=">=3.7,<4",
    zip_safe=False,
)
