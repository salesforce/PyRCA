import unittest
from pyrca.tools.dashboard.models.causal import CausalDiscovery


class TestCausal(unittest.TestCase):

    def test(self):
        causal = CausalDiscovery(folder=None)
        methods = causal.get_supported_methods()
        print(methods)

        params = causal.get_parameter_info("PC")
        print(params)


if __name__ == "__main__":
    unittest.main()
