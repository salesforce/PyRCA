from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("sfr-pyrca")
except DistributionNotFound:
    __version__ = "Please install PyRCA with setup.py"
else:
    __version__ = dist.version
