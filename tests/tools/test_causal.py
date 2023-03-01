import unittest
from pyrca.tools.dashboard.models.causal import CausalDiscovery


class TestCausal(unittest.TestCase):

    def test(self):
        methods = CausalDiscovery.get_supported_methods()
        print(methods)


if __name__ == "__main__":
    unittest.main()
