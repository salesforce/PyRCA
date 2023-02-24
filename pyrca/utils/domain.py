import yaml
import pprint
from schema import Schema, And, Or, Optional, SchemaError

_larger_than = (lambda target: lambda val: val > target)
_larger_than_zero = _larger_than(0)
_less_than = (lambda target: lambda val: val < target)
_less_than_one = _less_than(1)

SCHEMA = Schema(
    {
        Optional("causal-graph"): {
            "root-nodes": Or(list, None),
            "leaf-nodes": Or(list, None),
            "forbids": Or(list, None),
            "requires": Or(list, None)
        },

        Optional("root-causes"): [{
            "name": str,
            "P(r=1)": And(float, _larger_than_zero, _less_than_one),
            "metrics": [
                {
                    "name": str,
                    "P(m=0|r=0)": And(float, _larger_than_zero, _less_than_one),
                    "P(m=0|r=1)": And(float, _larger_than_zero, _less_than_one)
                }
            ]
        }]
    }
)


class DomainParser:

    def __init__(self, file_path):
        if file_path is None:
            self.config = None
        else:
            with open(file_path, 'r') as f:
                self.config = yaml.safe_load(f)
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise RuntimeError(
                    "The domain knowledge config does not fit the required schema."
                ) from e

    def get_forbid_links(self, graph_nodes=None):
        if self.config is None or "causal-graph" not in self.config:
            return []

        other_forbids = []
        # Constraints for leaf nodes
        leaf_nodes = self.config["causal-graph"]["leaf-nodes"]
        if leaf_nodes is not None:
            assert graph_nodes is not None, "`graph_nodes` cannot be None."
            for u in leaf_nodes:
                assert u in graph_nodes, f"{u} is not in `all_nodes`"
                for v in graph_nodes:
                    if u != v:
                        other_forbids.append([u, v])

        # Constraints for root nodes
        root_nodes = self.config["causal-graph"]["root-nodes"]
        if root_nodes is not None:
            assert graph_nodes is not None, "`graph_nodes` cannot be None."
            for u in root_nodes:
                assert u in graph_nodes, f"{u} is not in `all_nodes`"
                for v in graph_nodes:
                    if u != v:
                        other_forbids.append([v, u])

        # Additional forbidden links
        forbids = self.config["causal-graph"]["forbids"]
        if len(other_forbids) == 0:
            return forbids
        else:
            return other_forbids if forbids is None \
                else forbids + other_forbids

    def get_require_links(self):
        return [] if self.config is None or "causal-graph" not in self.config \
            else self.config["causal-graph"]["requires"]

    def get_root_causes(self):
        return [] if self.config is None or "root-causes" not in self.config \
            else self.config["root-causes"]

    def print(self):
        pprint.pprint(self.config)
